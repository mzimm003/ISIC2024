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

class Dummy(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.least = nn.LazyLinear(2)
    
    def forward(self, img:torch.Tensor, fet:torch.Tensor):
        return self.least(fet)

class Classifier(nn.Module):
    def __init__(
            self,
            embedding_dim:int = 64,
            img_height:int = 125,
            img_width:int = 125,
            patch_size:int = 5,
            nhead:int = 8,
            layers:int = 4,
            dim_feedforward:int = 1024,
            norm_first:bool = False,
            activation:Union[str, ActivationReg, nn.Module] = None,
            activation_kwargs:Dict[str, Any] = None,
            *args,
            **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.activation = ActivationReg.initialize(
            activation, activation_kwargs if activation_kwargs else {})
        self.feature_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            self.activation,
            nn.Linear(embedding_dim, embedding_dim))
        
        self.img_height = img_height
        self.img_width = img_width
        self.patch_size = patch_size
        assert (self.img_height % self.patch_size == 0 and
                self.img_width % self.patch_size == 0)
        self.positional_embedding_x = nn.Embedding(
            self.img_width//self.patch_size, embedding_dim)
        self.positional_embedding_y = nn.Embedding(
            self.img_height//self.patch_size, embedding_dim)
        self.img_reshape_x_mask = (
            torch.arange(self.patch_size).repeat(self.patch_size)+
            (torch.arange(self.img_width//self.patch_size)*self.patch_size)[:,None]
        ).repeat(self.img_height//self.patch_size, 1).long()
        self.img_reshape_y_mask = (
            torch.arange(self.patch_size).repeat_interleave(self.patch_size)+
            (torch.arange(self.img_height//self.patch_size)*self.patch_size)[:,None]
            ).repeat_interleave(self.img_width//self.patch_size, 0).long()
        self.reshape_img = lambda x: x[..., self.img_reshape_y_mask, self.img_reshape_x_mask, :].flatten(-2)
        self.img_patch_embedding = nn.LazyLinear(embedding_dim)
        self.img_flatten = nn.Flatten(-3,-2)
        self.transformer = nn.Transformer(
            d_model = embedding_dim,
            nhead = nhead,
            num_encoder_layers = layers,
            num_decoder_layers = layers,
            dim_feedforward = dim_feedforward,
            activation = self.activation,
            batch_first = True,
            norm_first = norm_first,
        )
        self.is_malignant = nn.Sequential(
            nn.Linear(embedding_dim, 2),
            self.activation,
            nn.Flatten(-2),
            nn.LazyLinear(2)
        )
    
    def forward(self, img:torch.Tensor, fet:torch.Tensor):
        param_ref = next(self.transformer.parameters())
        fet = fet.to(dtype=param_ref.dtype, device=param_ref.device)
        img = img.to(dtype=param_ref.dtype, device=param_ref.device)

        fet = self.feature_embedding(fet[...,None])

        width = img.shape[-3]
        height = img.shape[-2]
        x_emb = self.positional_embedding_x(
            torch.arange(width//self.patch_size, device=param_ref.device))
        y_emb = self.positional_embedding_y(
            torch.arange(height//self.patch_size, device=param_ref.device))
        pos_emb = (
            torch.tile(x_emb[None,:], (height//self.patch_size,1,1)) +
            torch.tile(y_emb[:,None], (1,width//self.patch_size,1))
            )
        pos_emb = self.img_flatten(pos_emb)
        img = self.reshape_img(img)
        im_emb = self.img_patch_embedding(img)
        trans_logits = self.transformer(im_emb + pos_emb, fet)
        logits = self.is_malignant(trans_logits)
        return logits

    def name(self):
        return "test"

class ModelReg(Registry):
    Classifier = Classifier
    Dummy = Dummy