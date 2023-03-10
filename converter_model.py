import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from diffusers import ModelMixin
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class ConverterConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()


class ConverterModel(PreTrainedModel):
    def __init__(self, config, dim_in=1024, dim_out=768):
        super().__init__(config)
        self.config = config
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.convert_layer = nn.Linear(self.dim_in,self.dim_out)
        self.post_init()

    def forward(self, x):
        return self.convert_layer(x)
    
    def _init_weights(self, module):
        return
