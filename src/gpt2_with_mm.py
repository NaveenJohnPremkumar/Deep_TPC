import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union, List
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Block, GPT2Attention, GPT2MLP
from transformers import GPT2Tokenizer, GPT2Config

# Gated Cross-Attention module for MM blocks
class GatedCrossAttention(nn.Module):
    """
    Gated Cross-Attention module for multimodal fusion as described in DeepMLF paper.
    This module allows fusion tokens to attend to patch embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        # Cross-attention projections
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.o_proj = nn.Linear(self.embed_dim, self.embed_dim)
        
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Gating parameter (initialized to 0.5 as per paper)
        self.gate_param = nn.Parameter(torch.ones(1) * 0.5)
        
    def _attn(self, query, key, value, attention_mask=None):
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        attn_weights = attn_weights / math.sqrt(self.head_dim)
        
        # Apply the attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
            
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
            
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
        
    def forward(
        self,
        query_states: torch.Tensor,  # Fusion tokens
        key_value_states: torch.Tensor,  # Patch embeddings
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        batch_size = query_states.size(0)
        
        # Project query (from fusion tokens)
        query = self.q_proj(query_states)
        # Project key/value (from patch embeddings)
        key = self.k_proj(key_value_states)
        value = self.v_proj(key_value_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Project output
        attn_output = self.o_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        # Apply sigmoid gating as per paper
        gate = torch.sigmoid(self.gate_param)
        gated_output = gate * attn_output
        
        outputs = (gated_output,)
        if output_attentions:
            outputs += (attn_weights,)
            
        return outputs

# MM Block component that will be added to GPT2Block
class MMBlock(nn.Module):
    """
    Multimodal Block that combines Gated Cross-Attention with a Feed-Forward Network.
    This is added to GPT2Block to enable cross-modal fusion with patch embeddings.
    """
    def __init__(self, config):
        super().__init__()
        self.ln_cross_attn = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.cross_attention = GatedCrossAttention(config)
        
        # Feed Forward layer after cross-attention (initialized from GPT2MLP)
        self.ln_ff = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4 * config.hidden_size, config)
        
        # Gating parameter for FFW (initialized to 0.5 as per paper)
        self.gate_param_ff = nn.Parameter(torch.ones(1) * 0.5)
        
        # Store the number of fusion tokens for processing  
        self.num_fusion_tokens = config.num_fusion_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_embeddings: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # Process only fusion tokens with cross-attention
        # Assuming hidden_states = [language_tokens, fusion_tokens]
        # First split the hidden states
        language_tokens = hidden_states[:, :-self.num_fusion_tokens, :]
        fusion_tokens = hidden_states[:, -self.num_fusion_tokens:, :]
        
        # Apply cross-attention only to fusion tokens
        cross_attn_norm = self.ln_cross_attn(fusion_tokens)
        cross_attn_outputs = self.cross_attention(
            cross_attn_norm,
            patch_embeddings,
            attention_mask=None,
            output_attentions=output_attentions,
        )
        cross_attn_output = cross_attn_outputs[0]
        
        # Apply residual connection
        fusion_tokens = fusion_tokens + cross_attn_output
        
        # Combine back with language tokens
        hidden_states = torch.cat([language_tokens, fusion_tokens], dim=1)
        
        # Apply FFW to entire hidden states with gating
        residual = hidden_states
        ff_output = self.mlp(self.ln_ff(hidden_states))
        
        # Apply sigmoid gating to FFW
        gate = torch.sigmoid(self.gate_param_ff)
        gated_ff_output = gate * ff_output
        
        # Residual connection
        hidden_states = residual + gated_ff_output
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += cross_attn_outputs[1:]
            
        return outputs

# Modified GPT2Block to include MM components
class GPT2BlockWithMM(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)
        
        # Flag to determine if this block contains MM components
        self.has_mm_block = False
        
        # Add MM Block if this layer is in the mm_layers list
        if hasattr(config, 'mm_layers') and layer_idx in config.mm_layers:
            self.has_mm_block = True
            self.mm_block = MMBlock(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_embeddings: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ):
        # Call the original GPT2Block forward pass
        outputs = super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        
        # Extract the hidden states from outputs
        hidden_states = outputs[0]
        
        # Apply MM block if available and patch embeddings are provided
        if self.has_mm_block and patch_embeddings is not None:
            mm_outputs = self.mm_block(
                hidden_states,
                patch_embeddings,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
            hidden_states = mm_outputs[0]
            if output_attentions and len(mm_outputs) > 1:
                # Add cross-attention outputs to the existing outputs
                outputs = outputs + (mm_outputs[1],)
        
        # Update the hidden states in the outputs
        outputs = (hidden_states,) + outputs[1:]
        
        return outputs

# Modified GPT2Model to work with patch embeddings and MM blocks
class GPT2ModelWithMM(GPT2Model):
    def __init__(self, config):
        # Initialize with the parent class constructor
        super().__init__(config)
        
        # Number of learnable fusion tokens to append
        self.num_fusion_tokens = getattr(config, 'num_fusion_tokens', 20)
        config.num_fusion_tokens = self.num_fusion_tokens  # Ensure this is stored in config
        
        # Learnable fusion tokens
        self.fusion_tokens = nn.Parameter(torch.zeros(1, self.num_fusion_tokens, config.hidden_size))
        # Initialize fusion tokens
        nn.init.normal_(self.fusion_tokens, std=0.02)
        
        # Replace the standard GPT2Blocks with our MM-enhanced blocks
        self.h = nn.ModuleList([
            GPT2BlockWithMM(config, layer_idx=i) 
            for i in range(config.num_hidden_layers)
        ])
        
        # Re-initialize weights for the model
        self.init_weights()
        
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=self.device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        position_embeds = self.wpe(position_ids)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        # Add fusion tokens
        fusion_tokens_expanded = self.fusion_tokens.expand(batch_size, -1, -1)
        hidden_states_with_fusion = torch.cat([hidden_states, fusion_tokens_expanded], dim=1)

        # --- CUSTOM ATTENTION MASK FOR PROMPT + FUSION TOKENS ---
        prompt_len = input_shape[-1]
        fusion_len = self.num_fusion_tokens
        total_len = prompt_len + fusion_len

        # Causal mask for prompt tokens
        causal_mask = torch.triu(
            torch.ones((prompt_len, prompt_len), device=self.device), diagonal=1
        )
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))

        # Fusion tokens can attend to all tokens
        fusion_mask = torch.zeros((fusion_len, total_len), device=self.device)

        # Full causal + fusion mask: [total_len x total_len]
        full_causal_mask = torch.cat([
            torch.cat([causal_mask, torch.full((prompt_len, fusion_len), float('-inf'), device=self.device)], dim=1),
            fusion_mask
        ], dim=0)

        # Expand to [B, 1, T, T]
        full_causal_mask = full_causal_mask.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, total_len, total_len)

        # Padding mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, prompt_len]
            fusion_pad_mask = torch.ones((batch_size, 1, 1, fusion_len), device=attention_mask.device)
            pad_mask = torch.cat([attention_mask, fusion_pad_mask], dim=-1)  # [B, 1, 1, total_len]
            full_causal_mask = full_causal_mask + (1.0 - pad_mask) * -10000.0

        attention_mask = full_causal_mask  # Final attention mask

        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None

        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_with_fusion,)

            outputs = block(
                hidden_states_with_fusion,
                patch_embeddings=patch_embeddings,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states_with_fusion = outputs[0]

            if use_cache:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

        hidden_states_with_fusion = self.ln_f(hidden_states_with_fusion)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states_with_fusion,)

        language_hidden_states = hidden_states_with_fusion[:, :-self.num_fusion_tokens, :]
        fusion_hidden_states = hidden_states_with_fusion[:, -self.num_fusion_tokens:, :]

        if not return_dict:
            return tuple(
                v for v in [language_hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return {
            'last_hidden_state': language_hidden_states,
            'fusion_hidden_states': fusion_hidden_states,
            'past_key_values': presents,
            'hidden_states': all_hidden_states,
            'attentions': all_self_attentions,
            'cross_attentions': all_cross_attentions,
        }


# Example usage in a time series model
class TimeSeriesGPT2WithMM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_len = config.token_len
        
        # Initialize the GPT2 model with MM blocks
        base_config = GPT2Config.from_pretrained(config.llm_ckp_dir)
        base_config.mm_layers = config.mm_layers
        base_config.num_fusion_tokens = config.num_fusion_tokens
        base_config.add_cross_attention = True
        base_config.num_attention_heads = config.num_attention_heads
        base_config.hidden_size = config.hidden_size
        base_config.num_hidden_layers = config.num_hidden_layers
        base_config.layer_norm_epsilon = config.layer_norm_epsilon
        base_config.attn_pdrop = config.attn_pdrop
        base_config.resid_pdrop = config.resid_pdrop
        base_config.embd_pdrop = config.embd_pdrop

        # Pass modified config directly to the model
        self.gpt2 = GPT2ModelWithMM(base_config)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm_ckp_dir)
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze GPT2 parameters except for MM blocks
        for name, param in self.gpt2.named_parameters():
            if 'mm_block' not in name and 'fusion_tokens' not in name:
                param.requires_grad = False
                
        # Time series tokenizer (encoder) and detokenizer (decoder)
        if config.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.gpt2.config.hidden_size)
            self.decoder = nn.Linear(self.gpt2.config.hidden_size, self.token_len)
        else:
            from layers.mlp import MLP  # Import your MLP implementation
            self.encoder = MLP(
                self.token_len, 
                self.gpt2.config.hidden_size, 
                config.mlp_hidden_dim, 
                config.mlp_hidden_layers, 
                config.dropout, 
                config.mlp_activation
            )
            self.decoder = MLP(
                self.gpt2.config.hidden_size, 
                self.token_len,
                config.mlp_hidden_dim, 
                config.mlp_hidden_layers,
                config.dropout, 
                config.mlp_activation
            )
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()    
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        bs, _, n_vars = x_enc.shape
        # x_enc: [bs x nvars x seq_len]
        x_enc = x_enc.permute(0, 2, 1)
        # x_enc: [bs * nvars x seq_len]
        x_enc = x_enc.reshape(x_enc.shape[0] * x_enc.shape[1], -1)
        # fold_out: [bs * n_vars x token_num x token_len]
        fold_out = x_enc.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = fold_out.shape[1]
        denorm_len = token_num * self.token_len
        
        # Create time embeddings (patch embeddings)
        times_embeds = self.encoder(fold_out)
        
        # Create a generic prompt for all samples in batch
        batch_size = times_embeds.size(0)
        generic_prompt = ["Forecast the future of this time series"] * batch_size

        # Tokenize and embed the prompt
        tokens = self.tokenizer(generic_prompt, return_tensors='pt', padding=True, truncation=True).to(times_embeds.device)
        prompt_embeds = self.gpt2.wte(tokens['input_ids'])  # shape: [B, prompt_len, D]

        # Process through GPT2 with MM blocks
        outputs = self.gpt2(
            inputs_embeds=prompt_embeds,
            patch_embeddings=times_embeds,
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True
        )
        
        # Get the language hidden states (not fusion tokens)
        last_hidden_state = outputs['last_hidden_state']
        
        # Decode back to time series domain
        dec_out = self.decoder(last_hidden_state)
        
        # Reshape and permute to match target shape
        dec_out = dec_out.reshape(bs, n_vars, -1)
        dec_out = dec_out.permute(0, 2, 1)
        
        # Ensure output length matches target length
        target_len = x_enc.shape[1]  # This should be 672
        if dec_out.shape[1] > target_len:
            dec_out = dec_out[:, :target_len, :]
        elif dec_out.shape[1] < target_len:
            # Pad with zeros if shorter
            pad_len = target_len - dec_out.shape[1]
            dec_out = torch.nn.functional.pad(dec_out, (0, 0, 0, pad_len, 0, 0))

        # Denormalize
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).expand(-1, target_len, -1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).expand(-1, target_len, -1)
        
        return dec_out
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)