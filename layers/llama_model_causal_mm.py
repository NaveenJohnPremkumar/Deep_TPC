import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from layers.llama_mm_block import LlamaDecoderLayerWithMM

# Custom LlamaModel that will be used inside LlamaForCausalLMWithMM
class LlamaModelWithMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        
        # Number of learnable fusion tokens to append
        self.num_fusion_tokens = getattr(config, 'num_fusion_tokens', 20)
        config.num_fusion_tokens = self.num_fusion_tokens  # Ensure this is stored in config
        
        # Learnable fusion tokens
        self.fusion_tokens = nn.Parameter(torch.zeros(1, self.num_fusion_tokens, config.hidden_size))
        # Initialize fusion tokens
        nn.init.normal_(self.fusion_tokens, std=0.02)
        
        # Replace the standard LlamaDecoderLayers with our MM-enhanced layers
        self.layers = nn.ModuleList([
            LlamaDecoderLayerWithMM(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Final layer norm
        from transformers.models.llama.modeling_llama import LlamaRMSNorm
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, config=None, **kwargs):
        """Load pretrained weights into our custom MM model"""
        
        # If config is provided, use it; otherwise load from pretrained path
        if config is None:
            config = LlamaConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create our custom model with the config
        model = cls(config)
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load pretrained weights directly to target device
        pretrained_state_dict = torch.load(
            f"{pretrained_model_name_or_path}/pytorch_model.bin", 
            map_location=device  # Load directly to GPU if available
        )
        
        # Load with strict=False to ignore missing/extra keys
        missing_keys, unexpected_keys = model.load_state_dict(
            pretrained_state_dict, 
            strict=False
        )
        
        print(f"Loaded pretrained weights from {pretrained_model_name_or_path}")
        print(f"Missing keys: {len(missing_keys)}")
        print(f"Unexpected keys: {len(unexpected_keys)}")
        
        # Initialize custom components that weren't in pretrained model
        model._initialize_custom_components()
        
        return model
    
    def _initialize_custom_components(self):
        """Initialize custom MM components that weren't in pretrained weights"""
        # Fusion tokens are already initialized in LlamaModelWithMM.__init__
        
        # Initialize any MM block components
        for name, module in self.named_modules():
            if 'mm_block' in name or 'fusion_tokens' in name:
                if hasattr(module, 'weight') and 'fusion_tokens' not in name:
                    nn.init.normal_(module.weight, std=0.02)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_embeddings: Optional[torch.FloatTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            # position_ids = position_ids.unsqueeze(0)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)  # → [B, T]


        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Add fusion tokens
        fusion_tokens_expanded = self.fusion_tokens.expand(batch_size, -1, -1)
        hidden_states = torch.cat([inputs_embeds, fusion_tokens_expanded], dim=1)

        # --- CUSTOM ATTENTION MASK FOR LLAMA + FUSION TOKENS ---
        T = seq_length  # Original sequence length
        F = self.num_fusion_tokens
        total_len = T + F
        device = hidden_states.device

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, T), dtype=torch.bool, device=device)
        
        # Extend attention mask for fusion tokens (fusion tokens are always "attended to")
        fusion_mask = torch.ones((batch_size, F), dtype=attention_mask.dtype, device=device)
        extended_attention_mask = torch.cat([attention_mask, fusion_mask], dim=1)

        # Create 4D attention mask for causal attention with fusion tokens
        # Shape: [batch_size, 1, total_len, total_len]
        causal_mask = torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1)
        
        # Create full attention mask: [total_len, total_len]
        full_mask = torch.zeros((total_len, total_len), device=device, dtype=torch.bool)
        
        # Fill in the causal mask for language tokens
        full_mask[:T, :T] = causal_mask
        
        # Language tokens cannot attend to fusion tokens (set to True to mask)
        full_mask[:T, T:] = True
        
        # Fusion tokens can attend to all tokens (already zeros, so no masking)
        
        # Convert to 4D and apply padding mask
        full_mask = full_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, total_len, total_len)
        
        # Apply padding mask - where extended_attention_mask is 0, we should mask (set to True)
        padding_mask = extended_attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, total_len]
        padding_mask = (1.0 - padding_mask.float()).bool()
        
        # Broadcast padding mask and combine with causal mask
        full_mask = full_mask | padding_mask.expand(-1, -1, total_len, -1)
        
        # Convert to the format expected by LLaMA (0 for attend, large negative for mask)
        attention_mask = torch.where(full_mask, torch.finfo(hidden_states.dtype).min, 0.0)
        # --- END CUSTOM MASK ---

        # Update position_ids for fusion tokens
        fusion_position_ids = torch.arange(
            seq_length, seq_length + F, dtype=torch.long, device=device
        ).unsqueeze(0).expand(batch_size, -1)
        position_ids = torch.cat([position_ids, fusion_position_ids], dim=1)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions, None)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    patch_embeddings=patch_embeddings,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Split hidden states back into language and fusion tokens
        language_hidden_states = hidden_states[:, :-self.num_fusion_tokens, :]
        fusion_hidden_states = hidden_states[:, -self.num_fusion_tokens:, :]

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [language_hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)

        return {
            'last_hidden_state': language_hidden_states,
            'fusion_hidden_states': fusion_hidden_states,
            'past_key_values': next_cache,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attns,
        }


# Modified LlamaForCausalLM to work with patch embeddings and MM blocks
class LlamaForCausalLMWithMM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        
        # Replace the model with our MM-enhanced version
        self.model = LlamaModelWithMM(config)
        
        # Keep the original lm_head
        # self.lm_head is already initialized by parent class
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        patch_embeddings: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            patch_embeddings=patch_embeddings,
        )

        if return_dict:
            hidden_states = outputs['last_hidden_state']
            fusion_hidden_states = outputs['fusion_hidden_states']
        else:
            hidden_states = outputs[0]
            fusion_hidden_states = None

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs.get('past_key_values'),
            'hidden_states': outputs.get('hidden_states'),
            'attentions': outputs.get('attentions'),
            'fusion_hidden_states': fusion_hidden_states,
        }