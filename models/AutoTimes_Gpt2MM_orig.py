import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model, GPT2Tokenizer
from layers.gpt2_model_mm import GPT2ModelWithMM


class Model(nn.Module):
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


        self.mix = config.mix_embeds
        
        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))

        # Pass modified config directly to the model
        self.gpt2 = GPT2ModelWithMM.from_pretrained(config.llm_ckp_dir, config=base_config)
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.llm_ckp_dir)
        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Freeze GPT2 parameters except for MM blocks
        for name, param in self.gpt2.named_parameters():
            if 'mm_block' not in name and 'fusion_tokens' not in name:
                param.requires_grad = False

            # if 'ln' in name or 'wpe' in name or 'bias' in name:
            #     param.requires_grad = True
            # else:
            #     param.requires_grad = False
        
        
                
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

        for param in self.encoder.parameters():
            param.requires_grad = True

        for param in self.decoder.parameters():
            param.requires_grad = True

        
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text):
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

        # Optional mix_embeds fusion
        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        
        # Tokenize the prompt
        tokens = self.tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True).to(times_embeds.device)
        prompt_embeds = self.gpt2.wte(tokens['input_ids'])  # shape: [B, prompt_len, D]
        
        # Concatenate prompt embeddings with patch embeddings
        # Reshape times_embeds to match batch size
        times_embeds = times_embeds.view(bs, n_vars * token_num, -1)  # [B, n_vars*token_num, D]
        combined_embeds = torch.cat([prompt_embeds, times_embeds], dim=1)  # [B, prompt_len + n_vars*token_num, D]

        # Process through GPT2 with MM blocks, now using combined embeddings
        outputs = self.gpt2(
            inputs_embeds=combined_embeds,
            patch_embeddings=times_embeds,  # Still provide patch embeddings for MM blocks
            output_attentions=False,
            output_hidden_states=False,
            use_cache=False,
            return_dict=True
        )
        
        # Get the language hidden states (not fusion tokens)
        last_hidden_state = outputs['last_hidden_state']
        
        # Take only the relevant portion of output for decoding
        # Skip the prompt part and take only the patch embeddings part
        patch_outputs = last_hidden_state[:, prompt_embeds.size(1):, :]
        
        # Reshape back to match original patch embedding shape
        patch_outputs = patch_outputs.reshape(bs * n_vars, token_num, -1)
        
        # Decode back to time series domain
        dec_out = self.decoder(patch_outputs)
        
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
    
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text)