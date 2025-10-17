#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FocusOnDepth-LSTM Integrated Model
----------------------------------
Combines a Vision Transformer backbone (via timm) with LSTM-based temporal encoding
and hierarchical feature fusion for depth estimation or multimodal feature extraction.
Assumes external FOD modules exist: Composition, Fusion, HeadDepth.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange

# External dependencies (must exist in your environment)
from FOD.Composition import Composition
from FOD.Fusion import Fusion
from FOD.Head import HeadDepth

torch.manual_seed(0)


# ======================================================================
#                           LSTM ENCODER
# ======================================================================
class LSTM(nn.Module):
    def __init__(self, lstm_layer=2, input_dim=1, hidden_size=8):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_layer = lstm_layer
        self.emb_layer = nn.Linear(input_dim, hidden_size)
        self.out_layer = nn.Linear(hidden_size, input_dim)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=self.lstm_layer,
            batch_first=True
        )

    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = (
            torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device),
            torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device)
        )
        return init_h

    def forward(self, x, h=None):
        # x: (batch, seq_len)
        x = x.unsqueeze(2)  # (B, T, 1)
        h = h if h else self.init_hidden(x)
        x = self.emb_layer(x)
        output, hidden = self.lstm(x, h)
        out = self.out_layer(output[:, -1, :]).squeeze(1)
        return out


# ======================================================================
#                        FOCUS ON DEPTH MODEL
# ======================================================================
class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size=(3, 384, 384),
                 patch_size=16,
                 emb_dim=1024,
                 resample_dim=256,
                 read='projection',
                 num_layers_encoder=12,
                 hooks=[5, 11, 17, 23],
                 reassemble_s=[4, 8, 16, 32],
                 transformer_dropout=0,
                 nclasses=2,
                 type="depth",
                 model_timm="vit_base_patch16_384"):
        """
        Focus on Depth (FOD) Model
        Args:
            image_size : (C, H, W)
            patch_size : square patch size
            emb_dim    : embedding dimension for ViT
            resample_dim : feature map fusion dim
            read       : fusion read mode
            hooks      : transformer block indices for skip connections
            type       : {"full", "depth", "segmentation"}
        """
        super().__init__()

        # Patch embedding
        channels, H, W = image_size
        assert H % patch_size == 0 and W % patch_size == 0, "Image dims must be divisible by patch size."
        num_patches = (H // patch_size) * (W // patch_size)
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, emb_dim),
        )

        # CLS token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        # LSTM encoder (optional temporal branch)
        self.nlstmlayers = 10
        self.ninp = 1000
        self.lstm_encoder = nn.LSTM(self.ninp, self.ninp, self.nlstmlayers)

        # Vision Transformer backbone
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)

        # Optional parallel backbone
        self.resnet50 = timm.create_model('resnet50', pretrained=True)

        # Hooks registration
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        # Multi-scale fusion
        self.reassembles = nn.ModuleList([
            Composition(image_size, read, patch_size, s, emb_dim, resample_dim)
            for s in reassemble_s
        ])
        self.fusions = nn.ModuleList([
            Fusion(resample_dim)
            for _ in reassemble_s
        ])

        # Output head
        self.type_ = type
        if type in ["full", "depth"]:
            self.head_depth = HeadDepth(resample_dim)
        else:
            self.head_depth = None

    def forward(self, img, hidden=None):
        t = self.transformer_encoders(img)
        print("######### Encoder starts #########")
        print("Transformer output:", t.shape)
        print("Input image:", img.shape)

        previous_stage = None
        for i in np.arange(len(self.fusions) - 1, -1, -1):
            hook_name = f"t{self.hooks[i]}"
            activation_result = self.activation[hook_name]
            reassemble_result = self.reassembles[i](activation_result)
            fusion_result = self.fusions[i](reassemble_result, previous_stage)
            previous_stage = fusion_result

        if self.head_depth is not None:
            out_depth = self.head_depth(previous_stage)
            return out_depth
        else:
            return previous_stage

    # Hook registration
    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook

        for h in hooks:
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation(f"t{h}"))

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.nlstmlayers, bsz, self.ninp),
                weight.new_zeros(self.nlstmlayers, bsz, self.ninp))


# ======================================================================
#                           TESTING ENTRY POINT
# ======================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FocusOnDepth().to(device)
    dummy_input = torch.randn(1, 3, 384, 384).to(device)

    with torch.no_grad():
        output = model(dummy_input)
    print("Final output shape:", output.shape)
