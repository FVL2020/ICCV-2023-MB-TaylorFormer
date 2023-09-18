## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
from pdb import set_trace as stx
import numbers
import math
from torchstat import stat
from einops import rearrange
import numpy as np
import torchvision


freqs_dict = dict()

##########################################################################
## Layer Norm
##rotary_pos_embed


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class refine_att(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv=nn.Conv2d(
                cur_head_split * Ch*2,
                cur_head_split,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split,
            )



            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch*2 for x in self.head_splits]

    def forward(self, q,k, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        k_img = k
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        q_img = rearrange(q_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        k_img = rearrange(k_img, "B h Ch (H W) -> B h Ch H W", H=H, W=W)
        qk_concat=torch.cat((q_img,k_img),2)
        qk_concat= rearrange(qk_concat, "B h Ch H W -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        qk_concat_list = torch.split(qk_concat, self.channel_splits, dim=1)
        qk_att_list = [
            conv(x) for conv, x in zip(self.conv_list, qk_concat_list)
        ]

        qk_att = torch.cat(qk_att_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        qk_att = rearrange(qk_att, "B (h Ch) H W -> B h (H W) Ch", h=h)

        return qk_att

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias,shared_refine_att=None,qk_norm=1):
        super(Attention, self).__init__()
        self.norm=qk_norm
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        #self.Leakyrelu=nn.LeakyReLU(negative_slope=0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if num_heads == 8:
            crpe_window = {
                3: 2,
                5: 3,
                7: 3
            }
        elif num_heads == 1:
            crpe_window = {
                3: 1,
            }
        elif num_heads == 2:
            crpe_window = {
                3: 2,
            }
        elif num_heads == 4:
            crpe_window = {
                3: 2,
                5: 2,
            }
        self.refine_att = refine_att(Ch=dim // num_heads,
                                     h=num_heads,
                                     window=crpe_window)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        q_norm=torch.norm(q,p=2,dim=-1,keepdim=True)/self.norm
        q=torch.div(q,q_norm)
        k_norm=torch.norm(k,p=2,dim=-2,keepdim=True)/self.norm
        k=torch.div(k,k_norm)
        #k = torch.nn.functional.normalize(k, dim=-2)

        refine_weight = self.refine_att(q,k, v, size=(h, w))
        #refine_weight=self.Leakyrelu(refine_weight)
        refine_weight = self.sigmoid(refine_weight)
        attn = k@v
        #attn = attn.softmax(dim=-1)

        #print(torch.sum(k, dim=-1).unsqueeze(3).shape)
        out_numerator = torch.sum(v, dim=-2).unsqueeze(2)+(q@attn)
        out_denominator = torch.full((h*w,c//self.num_heads),h*w).to(q.device)\
                          +q@torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads)+1e-6

        #out=torch.div(out_numerator,out_denominator)*self.temperature*refine_weight
        out = torch.div(out_numerator, out_denominator) * self.temperature
        out = out* refine_weight
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder comprised of `MHCA`
    blocks."""

    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1
    ):
        super().__init__()

        self.num_layers = num_layers
        self.MHCA_layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm
            ) for idx in range(self.num_layers)
        ])

    def forward(self, x, size):
        """foward function"""
        H, W = size
        B = x.shape[0]

        # return x's shape : [B, N, C] -> [B, C, H, W]
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for layer in self.MHCA_layers:
            x = layer(x)

        return x


class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
       # self.act0 = act_layer()
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        #self.norm = norm_layer(hidden_features)
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
                
    def forward(self, x):
        """foward function"""
        identity = x
        #x=self.act0(x)
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        #feat = self.norm(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat


class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage comprised of `MHCAEncoder`
    layers."""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=8,
            ffn_expansion_factor=2.66,
            num_path=4,
            bias=False,
            LayerNorm_type='BiasFree',
            qk_norm=1

    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type,
                qk_norm=qk_norm

            ) for _ in range(num_path)
        ])

        self.aggregate = SKFF(embed_dim,height=num_path)



        #self.InvRes = ResBlock(in_features=embed_dim, out_features=embed_dim)
       # self.aggregate = Conv2d_aggregate(embed_dim * (num_path + 1),
        #                           out_embed_dim,
        #                           act_layer=nn.Hardswish)

    def forward(self, inputs):
        """foward function"""
        #att_outputs = [self.InvRes(inputs[0])]
        att_outputs = []

        for x, encoder in zip(inputs, self.mhca_blks):
            # [B, C, H, W] -> [B, N, C]
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        #out_concat = torch.cat(att_outputs, dim=1)
        out = self.aggregate(att_outputs)

        return out



##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class Conv2d_BN(nn.Module):


    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)
        #self.bn = norm_layer(out_ch)

        #torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        #torch.nn.init.constant_(self.bn.bias, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
    
        x = self.conv(x)
        #x = self.bn(x)
        x = self.act_layer(x)

        return x


class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class DWConv2d_BN(nn.Module):

    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            bn_weight_init=1,
            offset_clamp=(-1,1)
    ):
        super().__init__()
        # dw
        # self.conv=torch.nn.Conv2d(in_ch,out_ch,kernel_size,stride,(kernel_size - 1) // 2,bias=False,)
       # self.mask_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
       #                                                 stride=1, padding=1, bias=False, groups=in_ch),
       #                                       nn.Conv2d(in_channels=in_ch, out_channels=9,
       #                                                 kernel_size=1,
       #                                                 stride=1, padding=0, bias=False)
        #                                      )
        self.offset_clamp=offset_clamp
        self.offset_generator=nn.Sequential(nn.Conv2d(in_channels=in_ch,out_channels=in_ch,kernel_size=3,
                                                      stride= 1,padding= 1,bias= False,groups=in_ch),
                                            nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                      kernel_size=1,
                                                      stride=1, padding=0, bias=False)

                                            )
        self.dcn=DeformConv2d(
                    in_channels=in_ch,
                    out_channels=in_ch,
                    kernel_size=3,
                    stride= 1,
                    padding= 1,
                    bias= False,
                    groups=in_ch
                    )#.cuda(7)
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)


        #self.bn = norm_layer(out_ch)
        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
                # print(m)

         #   elif isinstance(m, nn.BatchNorm2d):
           #     m.weight.data.fill_(bn_weight_init)
          #      m.bias.data.zero_()

    def forward(self, x):

        # x=self.conv(x)
        #x = self.bn(x)
        #x = self.act(x)
        #mask= torch.sigmoid(self.mask_generator(x))
        #print('1')
        offset = self.offset_generator(x)
        #print('2')
        if self.offset_clamp:
            offset=torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])#.cuda(7)1
        #print(offset)
        #print('3')
        #x=x.cuda(7)
        x = self.dcn(x,offset)
        #x=x.cpu()
        #print('4')
        x=self.pwconv(x)
       # print('5')
        #x = self.bn(x)
        x = self.act(x)
        return x


class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding layer Image to Patch
    Embedding."""

    def __init__(self,
                 in_chans=3,
                 embed_dim=768,
                 patch_size=16,
                 stride=1,
                 idx=0,
                 act_layer=nn.Hardswish,
                 offset_clamp=(-1,1)):
        super().__init__()


        self.patch_conv = DWConv2d_BN(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=stride,
                act_layer=act_layer,
                offset_clamp=offset_clamp
            )
        """
        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer,
        )
        """

    def forward(self, x):
        """foward function"""
        x = self.patch_conv(x)

        return x


class Patch_Embed_stage(nn.Module):
    """Depthwise Convolutional Patch Embedding stage comprised of
    `DWCPatchEmbed` layers."""

    def __init__(self, in_chans, embed_dim, num_path=4, isPool=False,offset_clamp=(-1,1)):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=in_chans if idx == 0 else embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1,
                idx=idx,
                offset_clamp=offset_clamp
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        """foward function"""
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)

        return att_inputs


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)
        # self.proj_dw = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1, padding=1,groups=in_c, bias=bias)
        # self.proj_pw = nn.Conv2d(in_c, embed_dim, kernel_size=1, stride=1, padding=0, bias=bias)

        #self.bn=nn.BatchNorm2d(embed_dim)
        #self.act=nn.Hardswish()

    def forward(self, x):
        x = self.proj(x)
        # x = self.proj_dw(x)
        # x= self.proj_pw(x)
        #x=self.bn(x)
        #x=self.act(x)
        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
           # nn.BatchNorm2d(n_feat // 2),
           # nn.Hardswish(),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, input_feat,out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(  # nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            # dw
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            # pw-linear
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
           # nn.BatchNorm2d(n_feat*2),
           # nn.Hardswish(),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)





##########################################################################
##---------- Restormer -----------------------
class MB_TaylorFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 model_name=1,
                 dim=[24,48,72,96],
                 num_blocks=[2,3,3,4],
                 num_refinement_blocks=2,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',  ## Other option 'BiasFree'
                 dual_pixel_task=False,
                 num_path=[2,2,2,2],  ## True for dual-pixel defocus deblurring only. Also set inp_channels=6
                 qk_norm=1,
                 offset_clamp=(-1,1)
                 ):

        super(MB_TaylorFormer, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim[0])
        self.patch_embed_encoder_level1 = Patch_Embed_stage(dim[0], dim[0], num_path=num_path[0], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level1 = MHCA_stage(dim[0], dim[0], num_layers=num_blocks[0], num_heads=heads[0],
                                         ffn_expansion_factor=2.66, num_path=num_path[0],
                                         bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm)
        
        self.down1_2 = Downsample(dim[0],dim[1])  ## From Level 1 to Level 2

        self.patch_embed_encoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1], isPool=False,offset_clamp=offset_clamp)
        self.encoder_level2 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
                                         ffn_expansion_factor=2.66,
                                         num_path=num_path[1], bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm)

        self.down2_3 = Downsample(dim[1],dim[2])  ## From Level 2 to Level 3

        self.patch_embed_encoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.encoder_level3 = MHCA_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
                                         ffn_expansion_factor=2.66,
                                         num_path=num_path[2], bias=False, LayerNorm_type='BiasFree',qk_norm=qk_norm)

        self.down3_4 = Downsample(dim[2],dim[3])  ## From Level 3 to Level 4

        self.patch_embed_latent = Patch_Embed_stage(dim[3], dim[3], num_path=num_path[3],
                                                    isPool=False,offset_clamp=offset_clamp)
        self.latent = MHCA_stage(dim[3], dim[3], num_layers=num_blocks[3], num_heads=heads[3],
                                 ffn_expansion_factor=2.66, num_path=num_path[3], bias=False,
                                 LayerNorm_type='BiasFree',qk_norm=qk_norm)


        self.up4_3 = Upsample(int(dim[3]),dim[2])  ## From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(dim[2]*2, dim[2], 1, 1, 0, bias=bias),
            #nn.BatchNorm2d(dim * 2**2),
           # nn.Hardswish(),
        )

        self.patch_embed_decoder_level3 = Patch_Embed_stage(dim[2], dim[2], num_path=num_path[2],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level3 = MHCA_stage(dim[2], dim[2], num_layers=num_blocks[2], num_heads=heads[2],
                                         ffn_expansion_factor=2.66, num_path=num_path[2], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm)

        self.up3_2 = Upsample(int(dim[2]),dim[1])  ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(dim[1]*2, dim[1], 1, 1, 0, bias=bias),
            #nn.BatchNorm2d( dim * 2),
            #nn.Hardswish(),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[1],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level2 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[1], num_heads=heads[1],
                                         ffn_expansion_factor=2.66, num_path=num_path[1], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm)



        self.up2_1 = Upsample(int(dim[1]),dim[0])  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)

        self.patch_embed_decoder_level1 = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[0],
                                                            isPool=False,offset_clamp=offset_clamp)
        self.decoder_level1 = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[0], num_heads=heads[0],
                                         ffn_expansion_factor=2.66, num_path=num_path[0], bias=False,
                                         LayerNorm_type='BiasFree',qk_norm=qk_norm)


        self.patch_embed_refinement = Patch_Embed_stage(dim[1], dim[1], num_path=num_path[0],
                                                        isPool=False,offset_clamp=offset_clamp)
        self.refinement = MHCA_stage(dim[1], dim[1], num_layers=num_blocks[0], num_heads=heads[0],
                                     ffn_expansion_factor=2.66, num_path=num_path[0], bias=False,
                                     LayerNorm_type='BiasFree',qk_norm=qk_norm)


        #### For Dual-Pixel Defocus Deblurring Task ####
        self.dual_pixel_task = dual_pixel_task
        if self.dual_pixel_task:
            self.skip_conv = nn.Conv2d(dim[0], dim[1], kernel_size=1, bias=bias)
        ###########################

        # self.output = nn.Conv2d(dim*2**1, 3, kernel_size=3, stride=1, padding=1, bias=False)

        self.output = nn.Sequential(  # nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(dim*2),
            # nn.Hardswish(),
            nn.Conv2d(dim[1], 3, kernel_size=3, stride=1, padding=1, bias=False, ),

        )

    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)

        inp_enc_level1_list = self.patch_embed_encoder_level1(inp_enc_level1)

        out_enc_level1 = self.encoder_level1(inp_enc_level1_list) + inp_enc_level1

        #out_enc_level1 = self.encoder_level1(inp_enc_level1_list)
        inp_enc_level2 = self.down1_2(out_enc_level1)
        
        inp_enc_level2_list = self.patch_embed_encoder_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2_list) + inp_enc_level2
        inp_enc_level3 = self.down2_3(out_enc_level2)

        inp_enc_level3_list = self.patch_embed_encoder_level3(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3_list) + inp_enc_level3
        inp_enc_level4 = self.down3_4(out_enc_level3)

        inp_latent = self.patch_embed_latent(inp_enc_level4)
        latent = self.latent(inp_latent) + inp_enc_level4

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3_list = self.patch_embed_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3_list) + inp_dec_level3

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)

        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        inp_dec_level2_list = self.patch_embed_decoder_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2_list) + inp_dec_level2

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        

        inp_dec_level1_list = self.patch_embed_decoder_level1(inp_dec_level1)

        out_dec_level1 = self.decoder_level1(inp_dec_level1_list) + inp_dec_level1

        inp_latent_list = self.patch_embed_refinement(out_dec_level1)

        out_dec_level1 = self.refinement(inp_latent_list) + out_dec_level1

        # nn.Hardswish()

        #### For Dual-Pixel Defocus Deblurring Task ####
        if self.dual_pixel_task:
            out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
            out_dec_level1 = self.output(out_dec_level1)
        ###########################
        else:
            out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1

def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


if __name__ == "__main__":
    from thop import profile
    model = MB_TaylorFormer()
    model.eval()
    print("params", count_param(model))
    inputs = torch.randn(1, 3, 256, 256)
    output=model(inputs)
    print(stat(model, (3, 256, 256)))
