import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from layers.mm_block import MMBlock

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