import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel
from layers.mlp import MLP

from peft import LoraConfig, get_peft_model, TaskType

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.token_len = configs.token_len
        if configs.use_multi_gpu:
            self.device = f"cuda:{configs.local_rank}"
        else:
            self.device = f"cuda:{configs.gpu}"
        print(self.device)
        self.gpt2 = GPT2LMHeadModel.from_pretrained(configs.llm_ckp_dir) 
        self.hidden_dim_of_gpt2 = 768 # change this to 768 if using GPT2-small
        self.mix = configs.mix_embeds

        if getattr(configs, "use_lora", False):
            lora_cfg = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=getattr(configs, "lora_r", 16),
                lora_alpha=getattr(configs, "lora_alpha", 32),
                lora_dropout=getattr(configs, "lora_dropout", 0.05),
                target_modules=["attn.c_attn", "attn.c_proj", "mlp.c_fc", "mlp.c_proj"],
            )
            self.gpt2 = get_peft_model(self.gpt2, lora_cfg)
            for n, p in self.gpt2.named_parameters():
                p.requires_grad = ("lora_" in n)
            try:
                self.gpt2.print_trainable_parameters()
            except Exception:
                pass
        else:
            for _, p in self.gpt2.named_parameters():
                p.requires_grad = False

        if self.mix:
            self.add_scale = nn.Parameter(torch.ones([]))
        
            
        # for i in range(7):
        #     block = self.gpt2.h[-(i + 1)]
        #     for param in block.parameters():
        #         param.requires_grad = True

        if configs.mlp_hidden_layers == 0:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use linear as tokenizer and detokenizer")
            self.encoder = nn.Linear(self.token_len, self.hidden_dim_of_gpt2)
            self.decoder = nn.Linear(self.hidden_dim_of_gpt2, self.token_len)
        else:
            if not configs.use_multi_gpu or (configs.use_multi_gpu and configs.local_rank == 0):
                print("use mlp as tokenizer and detokenizer")
            self.encoder = MLP(self.token_len, self.hidden_dim_of_gpt2, 
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers, 
                            configs.dropout, configs.mlp_activation)
            self.decoder = MLP(self.hidden_dim_of_gpt2, self.token_len,
                            configs.mlp_hidden_dim, configs.mlp_hidden_layers,
                            configs.dropout, configs.mlp_activation) 
    
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text=None):
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
        # times_embeds: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        times_embeds = self.encoder(fold_out)



        if self.mix:
            times_embeds = times_embeds / times_embeds.norm(dim=2, keepdim=True)
            x_mark_enc = x_mark_enc / x_mark_enc.norm(dim=2, keepdim=True)
            times_embeds = times_embeds + self.add_scale * x_mark_enc
        # outputs: [bs * n_vars x token_num x hidden_dim_of_gpt2]
        outputs = self.gpt2.transformer(
            inputs_embeds=times_embeds).last_hidden_state
        # dec_out: [bs * n_vars x token_num x token_len]
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(bs, n_vars, -1)
        # dec_out: [bs x token_num * token_len x n_vars]
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * \
            (stdev[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        dec_out = dec_out + \
            (means[:, 0, :].unsqueeze(1).repeat(1, token_num * self.token_len, 1))
        
        return dec_out
    
    # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text):
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, prompt_text=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)