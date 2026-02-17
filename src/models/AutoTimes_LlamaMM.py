# import torch
# import torch.nn as nn
# from transformers import LlamaConfig
# from layers.llama_model_causal_mm import LlamaForCausalLMWithMM
# from layers.mlp import MLP


# class Model(nn.Module):
#     """Time-series ⇄ LLaMA (with MM blocks)"""

#     def __init__(self, cfg):
#         super().__init__()

#         self.token_len = cfg.token_len
#         self.mix       = cfg.mix_embeds

#         base_config = LlamaConfig.from_pretrained(cfg.llm_ckp_dir)
#         base_config.mm_layers           = cfg.mm_layers
#         base_config.num_fusion_tokens  = cfg.num_fusion_tokens
#         base_config.add_cross_attention = True
#         # base_config.num_attention_heads = cfg.num_attention_heads
#         # base_config.hidden_size         = cfg.hidden_size
#         # base_config.num_hidden_layers   = cfg.num_hidden_layers
#         base_config.layer_norm_epsilon  = cfg.layer_norm_epsilon
#         base_config.attn_pdrop          = cfg.attn_pdrop
#         base_config.resid_pdrop         = cfg.resid_pdrop
#         base_config.embd_pdrop          = cfg.embd_pdrop

#         # Load model with custom config, then load weights
#         # self.llama = LlamaForCausalLMWithMM(base_config)
#         # self.llama.load_state_dict(
#         #     torch.load(f"{cfg.llm_ckp_dir}/pytorch_model.bin", map_location="cpu"),
#         #     strict=False
#         # )
#         self.llama = LlamaForCausalLMWithMM.from_pretrained(
#             cfg.llm_ckp_dir,
#             config=base_config
#         )

#         self.hidden_dim_of_llama = self.llama.model.config.hidden_size
#         self.add_scale = nn.Parameter(torch.ones([]))

#         # Freeze base LLaMA weights except MM parts
#         for name, param in self.llama.named_parameters():
#             if "mm_block" not in name and "fusion_tokens" not in name:
#                 param.requires_grad = False
#         for i, layer in enumerate(self.llama.model.layers):
#             has_mm = hasattr(layer, "mm_block") and layer.mm_block is not None
#             print(f"Layer {i}: MM block = {has_mm}")        

#         if cfg.mlp_hidden_layers == 0:
#             self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
#             self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
#         else:
#             self.encoder = MLP(
#                 self.token_len,
#                 self.hidden_dim_of_llama,
#                 cfg.mlp_hidden_dim,
#                 cfg.mlp_hidden_layers,
#                 cfg.dropout,
#                 cfg.mlp_activation,
#             )
#             self.decoder = MLP(
#                 self.hidden_dim_of_llama,
#                 self.token_len,
#                 cfg.mlp_hidden_dim,
#                 cfg.mlp_hidden_layers,
#                 cfg.dropout,
#                 cfg.mlp_activation,
#             )

#     def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):
#         means = x_enc.mean(1, keepdim=True).detach()
#         stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
#         x_norm = (x_enc - means) / stdev

#         B, T, N = x_norm.shape
#         x_var_first = x_norm.permute(0, 2, 1)           # [B, N, T]
#         x_flat = x_var_first.reshape(B * N, -1)         # [B·N, T]
#         patches = x_flat.unfold(dimension=-1, size=self.token_len, step=self.token_len)
#         token_num = patches.size(1)

#         times_embeds = self.encoder(patches)            # [B·N, token_num, D]

#         x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
#         mark_tokens = self.add_scale * x_mark_enc       # [B·N, token_num, D]

#         llama_out = self.llama.model(
#             inputs_embeds=times_embeds,
#             patch_embeddings=mark_tokens,
#             use_cache=False,
#             return_dict=True,
#         )

#         hidden = llama_out["last_hidden_state"]         # [B·N, token_num, D]
#         output = self.decoder(hidden)                   # [B·N, token_num, token_len]
#         output = output.reshape(B, N, -1).permute(0, 2, 1)  # [B, T', N]

#         output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)
#         output = output + means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)

#         return output

#     def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):
#         return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text)

import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaConfig
from layers.llama_model_causal_mm import LlamaForCausalLMWithMM
from layers.mlp import MLP


class Model(nn.Module):
    """Time-series ⇄ LLaMA (with MM blocks + prompt tokens)"""

    def __init__(self, cfg):
        super().__init__()

        self.token_len = cfg.token_len
        self.mix       = cfg.mix_embeds

        base_config = LlamaConfig.from_pretrained(cfg.llm_ckp_dir)
        base_config.mm_layers           = cfg.mm_layers
        base_config.num_fusion_tokens  = cfg.num_fusion_tokens
        base_config.add_cross_attention = True
        base_config.layer_norm_epsilon  = cfg.layer_norm_epsilon
        base_config.attn_pdrop          = cfg.attn_pdrop
        base_config.resid_pdrop         = cfg.resid_pdrop
        base_config.embd_pdrop          = cfg.embd_pdrop

        self.llama = LlamaForCausalLMWithMM.from_pretrained(
            cfg.llm_ckp_dir,
            config=base_config
        )

        self.hidden_dim_of_llama = self.llama.model.config.hidden_size
        self.add_scale = nn.Parameter(torch.ones([]))

        # Tokenizer
        self.tokenizer = LlamaTokenizer.from_pretrained(cfg.llm_ckp_dir)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # make sure padding is handled

        # Freeze all except MM
        for name, param in self.llama.named_parameters():
            if "mm_block" not in name and "fusion_tokens" not in name:
                param.requires_grad = False

        for i, layer in enumerate(self.llama.model.layers):
            has_mm = hasattr(layer, "mm_block") and layer.mm_block is not None
            print(f"Layer {i}: MM block = {has_mm}")

        # Encoder/Decoder
        if cfg.mlp_hidden_layers == 0:
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_llama)
            self.decoder = nn.Linear(self.hidden_dim_of_llama, self.token_len)
        else:
            self.encoder = MLP(
                self.token_len, self.hidden_dim_of_llama,
                cfg.mlp_hidden_dim, cfg.mlp_hidden_layers,
                cfg.dropout, cfg.mlp_activation
            )
            self.decoder = MLP(
                self.hidden_dim_of_llama, self.token_len,
                cfg.mlp_hidden_dim, cfg.mlp_hidden_layers,
                cfg.dropout, cfg.mlp_activation
            )

    def forecast(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):
        means = x_enc.mean(1, keepdim=True).detach()
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x_enc - means) / stdev

        B, T, N = x_norm.shape
        x_var_first = x_norm.permute(0, 2, 1)           # [B, N, T]
        x_flat = x_var_first.reshape(B * N, -1)         # [B·N, T]
        patches = x_flat.unfold(dimension=-1, size=self.token_len, step=self.token_len)
        token_num = patches.size(1)

        times_embeds = self.encoder(patches)            # [B·N, token_num, D]

        x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
        mark_tokens = self.add_scale * x_mark_enc       # [B·N, token_num, D]

        # ---------- NEW: Prompt Token Embedding ----------
        if prompt_text is not None:
            tokens = self.tokenizer(prompt_text, return_tensors='pt', padding=True, truncation=True).to(times_embeds.device)
            prompt_embeds = self.llama.model.embed_tokens(tokens['input_ids'])  # [B, prompt_len, D]

            # expand to match [B·N, ...]
            # prompt_embeds = prompt_embeds.repeat_interleave(N, dim=0)  # [B·N, prompt_len, D]

            # Concatenate prompts with time embeddings
            times_embeds = torch.cat([prompt_embeds, times_embeds], dim=1)  # [B·N, prompt+token_num, D]
        else:
            prompt_embeds = None

        # ---------- Forward Pass ----------
        llama_out = self.llama.model(
            inputs_embeds=times_embeds,
            patch_embeddings=mark_tokens,
            use_cache=False,
            return_dict=True,
        )

        hidden = llama_out["last_hidden_state"]         # [B·N, prompt+token_num, D]

        # ---------- Remove Prompt ----------
        if prompt_embeds is not None:
            hidden = hidden[:, prompt_embeds.size(1):, :]  # Remove prompt from hidden

        output = self.decoder(hidden)                   # [B·N, token_num, token_len]
        output = output.reshape(B, N, -1).permute(0, 2, 1)  # [B, T', N]

        output = output * stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)
        output = output + means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1)

        return output

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, prompt_text=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text)

