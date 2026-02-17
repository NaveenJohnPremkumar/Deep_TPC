import torch
from torch import nn
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from layers.mm_block_llama import MMBlockLlama

# Modified LlamaDecoderLayer to include MM components
class LlamaDecoderLayerWithMM(LlamaDecoderLayer):
    def __init__(self, config, layer_idx=None):
        # super().__init__(config, layer_idx)
        super().__init__(config)

        # self.layer_idx = layer_idx
        # Flag to determine if this block contains MM components
        self.has_mm_block = False
        
        # Add MM Block if this layer is in the mm_layers list
        if hasattr(config, 'mm_layers') and layer_idx in config.mm_layers:
            self.has_mm_block = True
            self.mm_block = MMBlockLlama(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        patch_embeddings: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        # Call the original LlamaDecoderLayer forward pass
        outputs = super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
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
