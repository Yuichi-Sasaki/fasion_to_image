import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.configuration_utils import PretrainedConfig

class ConverterConfig(PretrainedConfig):
    def __init__(self):
        super().__init__()


class ConverterModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.convert_layer = nn.Linear(1024,768)
        self.post_init()

    def forward(self, x):
        return self.convert_layer(x)
    
    def _init_weights(self, module):
        return
