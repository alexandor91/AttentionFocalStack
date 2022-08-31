import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from FOD.Reassemble import Reassemble
from FOD.Fusion import Fusion
from FOD.Head import HeadDepth, HeadSeg

import numpy
import cv2
import os
torch.manual_seed(0)
from torchvision import transforms
#class LSTMFocusOnDepth(nn.Module):
from PIL import Image

class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 resample_dim       = 256,
                 read               = 'projection',
                 num_layers_encoder = 12,
                 hooks              = [5, 11, 17, 23],
                 reassemble_s       = [4, 8, 16, 32],
                 transformer_dropout= 0,
                 nclasses           = 2,
                 type               = "depth",
                 model_timm         = "vit_base_patch16_384"):
        """
        Focus on Depth
        type : {"full", "depth", "segmentation"}
        image_size : (c, h, w)
        patch_size : *a square*
        emb_dim <=> D (in the paper)
        resample_dim <=> ^D (in the paper)
        read : {"ignore", "add", "projection"}
        """
        super().__init__()

        #Splitting img into patches
        channels, image_height, image_width = image_size
        assert image_height % patch_size == 0 and image_width % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_height // patch_size) * (image_width // patch_size)
        patch_dim = channels * patch_size * patch_size
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, emb_dim),
        )
        # #Embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_dim))

        # LSTM
        self.nlstmlayers = 10
        self.ninp = 1000
        self.lstm_encoder = nn.LSTM(self.ninp, self.ninp, self.nlstmlayers)

        #Transformer
        # encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=nhead, dropout=transformer_dropout, dim_feedforward=emb_dim*4)
        # self.transformer_encoders = nn.TransformerEncoder(encoder_layer, num_layers=num_layers_encoder)
        self.transformer_encoders = timm.create_model(model_timm, pretrained=True)
        self.type_ = type
        self.resnet50 = timm.create_model('resnet50', pretrained=True)

        #Register hooks
        self.activation = {}
        self.hooks = hooks
        self._get_layers_from_hooks(self.hooks)

        #Reassembles Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Reassemble(image_size, read, patch_size, s, emb_dim, resample_dim))
            self.fusions.append(Fusion(resample_dim))
        self.reassembles = nn.ModuleList(self.reassembles)
        self.fusions = nn.ModuleList(self.fusions)

        #Head
        if type == "full":
            self.head_depth = HeadDepth(resample_dim)
            #self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)
        elif type == "depth":
            self.head_depth = HeadDepth(resample_dim)
            #self.head_segmentation = None
        else:
            self.head_depth = None
            #self.head_segmentation = HeadSeg(resample_dim, nclasses=nclasses)

    def forward(self, img, hidden=None):
        # print(img.shape) (1,1,384,384)
        t = self.transformer_encoders(img)
        #print(t.shape, hidden.shape) #(1,1000)
        t, hidden = self.lstm_encoder(t, hidden)
        count = 0
        previous_stage = None
        for i in np.arange(len(self.fusions)-1, -1, -1):
            hook_to_take = 't'+str(self.hooks[i])

            activation_result = self.activation[hook_to_take]
            reassemble_result = self.reassembles[i](activation_result)

            fusion_result = self.fusions[i](reassemble_result, previous_stage)


            # out = self.head_depth(fusion_result)
            # print(i, out.shape)
            # original_size = (1080, 1080)
            #output = transforms.ToPILImage()(out.squeeze(0).float()).resize(original_size, resample=Image.BICUBIC)
            #image = numpy.array(output)
            #image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            #image = cv2.Laplacian(image, cv2.CV_16S, ksize=1)

            #image = cv2.cvtColor(image, cv2.COLORMAP_HOT)
            #cv2.applyColorMap(image, cv2.COLORMAP_COOL, image)

            #path_dir = '/home/eavise/ShuhuaYang/FocusOnDepth-LSTM/output/hooks'
            #cv2.imwrite(os.path.join(path_dir, str(i) + '.jpg'), image)

            previous_stage = fusion_result
        out_depth = None
        # out_segmentation = None
        #print(previous_stage.shape)
        if self.head_depth != None:
            out_depth = self.head_depth(previous_stage)
        #if self.head_segmentation != None:
        #    out_segmentation = self.head_segmentation(previous_stage)
        return out_depth, hidden

    def _get_layers_from_hooks(self, hooks):
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        for h in hooks:
            #self.transformer_encoders.layers[h].register_forward_hook(get_activation('t'+str(h)))
            self.transformer_encoders.blocks[h].register_forward_hook(get_activation('t'+str(h)))
            # print (self.transformer_encoders.blocks[h].register_forward_hook)

    def init_hidden(self, bsz):
        weight = next(self.parameters())


        return weight.new_zeros(self.nlstmlayers, bsz, self.ninp)
