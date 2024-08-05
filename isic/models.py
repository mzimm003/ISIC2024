import torch
from torch import nn

from pathlib import Path

from typing import (
    Union,
    Dict,
    Any,
    Type
)

from isic.registry import (
    Registry,
    ActivationReg
)

class Classifier(nn.Module):
    def __init__(
            self,
            embedding_dim:int = 64,
            feature_reducer_path:Union[str, Path] = None,
            hidden_dim:int = 256,
            nhead:int = 8,
            layers:int = 4,
            dim_feedforward:int = 1024,
            norm_first:bool = False,
            activation:Union[str, ActivationReg, nn.Module] = None,
            activation_kwargs:Dict[str, Any] = None,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = ActivationReg.initialize(activation, activation_kwargs)
        self.feature_reducer = Registry.load_model(feature_reducer_path)
        self.feature_embedding = nn.Sequential(
            nn.LazyLinear(embedding_dim),
            self.activation,
            nn.Linear(embedding_dim, embedding_dim))
        #TODO
        self.query = nn.Linear(1,1)
        #TODO
        self.positional_embedding_x = nn.Embedding(1,1)
        self.positional_embedding_y = nn.Embedding(1,1)
        self.img_color_embedding = nn.Linear(3,1)
        #END TODO
        self.img_flatten = nn.Flatten(-3,-2)
        self.transformer = nn.Transformer(
            d_model = hidden_dim,
            nhead = nhead,
            num_encoder_layers = layers,
            num_decoder_layers = layers,
            dim_feedforward = dim_feedforward,
            activation = self.activation,
            batch_first = True,
            norm_first = norm_first,
        )
    
    def forward(self, img:torch.Tensor, fet:torch.Tensor):
        fet = self.feature_reducer(fet)

        param_ref = next(self.transformer.parameters())
        fet = fet.to(dtype=param_ref.dtype, device=param_ref.device)
        img = img.to(dtype=param_ref.dtype, device=param_ref.device)

        fet = self.feature_embedding(fet)
        fet = self.query(fet).reshape(2,-1)

        width = img.shape[-3]
        height = img.shape[-2]
        x_emb = self.positional_embedding_x(torch.arange(width))
        y_emb = self.positional_embedding_y(torch.arange(height))
        pos_emb = (
            torch.tile(x_emb[None,:], (height,1)) +
            torch.tile(y_emb[:,None], (1, width))
            )
        im_emb = self.img_color_embedding(img)
        img = self.img_flatten(im_emb + pos_emb)
        self.transformer(img, fet)

class ModelReg(Registry):
    Classifier = Classifier