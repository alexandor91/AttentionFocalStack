import numpy as np
import torch
import torch.nn as nn
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from FOD.Composition import Composition
from FOD.Fusion import Fusion
from FOD.Head import HeadDepth

import numpy
import cv2
import os
torch.manual_seed(0)
from torchvision import transforms
#class LSTMFocusOnDepth(nn.Module):
from PIL import Image

# class RNN(nn.Module):
#     def __init__(self, rnn_layer=2, input_size=1, hidden_size=4):
#         super(RNN, self).__init__()
#         self.rnn_layer = rnn_layer
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnn = nn.RNN(
#             input_size = self.input_size,  #每个字母的向量长度
#             hidden_size=self.hidden_size,  # RNN隐藏神经元个数
#             num_layers=self.rnn_layer,  # RNN隐藏层个数
#             batch_first=True
#         )
#         self.fc = nn.Linear(self.hidden_size, 1)
#     def init_hidden(self, x):
#         batch_size = x.shape[0]
#         init_h = torch.zeros(self.rnn_layer, batch_size, self.hidden_size, device=x.device).requires_grad_()
#         return init_h
#     def forward(self, x, h=None):
#         x = x.unsqueeze(2)
#         h = h if h else self.init_hidden(x)
#         out, h = self.rnn(x, h)
#         out = self.fc(out[:,-1,:]).squeeze(1)
#         return out
class EarlyMultiConvolutionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EarlyMultiConvolutionModule, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
        self.conv7x7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3)
        
        self.depth_wise = nn.Conv2d(out_channels * 3, out_channels, kernel_size=3, padding=1, groups=out_channels)
        self.conv1x1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        out7x7 = self.conv7x7(x)
        
        out = torch.cat([out3x3, out5x5, out7x7], dim=1)
        out = self.depth_wise(out)
        out = self.conv1x1(out)
        
        return out

class LSTM(nn.Module):
    def __init__(self, lstm_layer=2, input_dim=1, hidden_size=8):
        super(LSTM, self).__init__()
        self.hidden_size=hidden_size
        self.lstm_layer = lstm_layer
        self.emb_layer = nn.Linear(input_dim, hidden_size)
        self.out_layer = nn.Linear(hidden_size, input_dim)
        self.lstm = nn.LSTM(input_size=1024, hidden_size=hidden_size, num_layers=self.lstm_layer, batch_first=True)
    
    def init_hidden(self, x):
        batch_size = x.shape[0]
        init_h = (torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device),
                torch.zeros(self.lstm_layer, batch_size, self.hidden_size, device=x.device))
        return init_h

    def forward(self, x, h=None):
        # batch x stack size x dim
        x = x.unsqueeze(2)
        h = h if h else self.init_hidden(x)
        x = self.emb_layer(x)        
        output, hidden = self.lstm(x, h)        
        out = self.out_layer(output[:,-1,:]).squeeze(1)
        return out
    
class FrameFusionLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(FrameFusionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, input_tensor):
        batch_size, num_frames, token_dim = input_tensor.size()
        num_tokens = input_tensor.size(1)

        # Reshape input tensor to (batch_size * num_frames, num_tokens, token_dim)
        input_tensor = input_tensor.view(-1, num_tokens, token_dim)

        # Initialize hidden and cell states
        h_0 = torch.zeros(self.lstm.num_layers, batch_size * num_frames, self.hidden_size, device=input_tensor.device)
        c_0 = torch.zeros(self.lstm.num_layers, batch_size * num_frames, self.hidden_size, device=input_tensor.device)

        # Pass input through LSTM
        output, _ = self.lstm(input_tensor, (h_0, c_0))

        # Reshape output to (batch_size, num_frames, num_tokens, hidden_size)
        output = output.view(batch_size, num_frames, num_tokens, self.hidden_size)

        # Fuse output tokens along the frame dimension
        fused_output = output.max(dim=1)[0]

        return fused_output  


class FocusOnDepth(nn.Module):
    def __init__(self,
                 image_size         = (3, 384, 384),
                 patch_size         = 16,
                 emb_dim            = 1024,
                 stack_size         = 9, ###########5, 7, ddff 12 scene
                 ninp       = 1024,
                 threshold = 0.4
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
        self.counter = 0
        self.cached_tokens = []
        # LSTM
        self.nlstmlayers = stack_size
        self.ninp = ninp
        self.threshold = threshold
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


        # Fusion
        self.reassembles = []
        self.fusions = []
        for s in reassemble_s:
            self.reassembles.append(Composition(image_size, read, patch_size, s, emb_dim, resample_dim))
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
        parallel_conv = EarlyMultiConvolutionModule(in_channels=3, out_channels=3)
        output_feature_map = parallel_conv(img)
        t = self.transformer_encoders(output_feature_map)

        print('#########encoder starts!!!!##########')
        print(t.shape)
        print(img.shape)
        #print(t.shape, hidden.shape) #(1,1000)
        # t, hidden = self.lstm_encoder(t, hidden)
        count = 0
        previous_stage = None
        # Example usageeavise1234
        
        # input_tensor = torch.randn(5, 172, 1024)  # (num_frames, num_tokens, token_dim)
        batch_size, num_frames, token_dim = t.size()
        num_tokens = t.size(1)

        # Compute token norms
        token_norms = t.norm(dim=2)

        # Split tokens based on the threshold
        high_norms = t[token_norms >= self.threshold]
        low_norms = t[token_norms < self.threshold]

        self.cached_tokens.append(high_norms)
        self.counter +=1 
        # Fuse high-norm tokens using FrameFusionLSTM
        if self.counter == self.nlstmlayers: 
            self.counter = 0 #########reset the counter########### 
            token_features = torch.stack(self.cached_tokens, dim=1)
            token_features = token_features.view(batch_size, num_tokens, num_frames, token_dim).transpose(1, 2)
            lstm_model = FrameFusionLSTM(input_size=token_dim, hidden_size=512, num_layers=2)
            fused_high_norms = lstm_model(token_features)

            # Fuse low-norm tokens using mean pooling
            fused_low_norms = low_norms.view(batch_size, num_frames, -1, token_dim).mean(dim=2)

            # Merge the fused outputs
            fused_output = torch.zeros(batch_size, num_tokens, token_dim, device=t.device)
            fused_output[token_norms > self.threshold] = fused_high_norms
            fused_output[token_norms <= self.threshold] = fused_low_norms
            t = fused_output
            # final fused shape
            print(fused_output.shape)  # Output: torch.Size([172, 1024])  

            for i in np.arange(len(self.fusions)-1, -1, -1):
                hook_to_take = 't'+str(self.hooks[i])
                print('###########hook starts!!!!!!###########')

                activation_result = self.activation[hook_to_take]
                reassemble_result = self.reassembles[i](activation_result)
                print(activation_result.shape)

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
            return out_depth #, hidden

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
