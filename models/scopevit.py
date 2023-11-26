# Copyright 2021 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import trunc_normal_, DropPath, create_conv2d, Mlp, _assert, to_2tuple
from timm.models.registry import register_model
from einops import rearrange
from typing import Optional, Tuple, List

try:
    from mmcv.runner import _load_checkpoint
except ImportError:
    print("If for dense prediction tasks, please install mmcv-full first.")

try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    has_mmseg = True
except ImportError:
    print("If for semantic segmentation, please install mmsegmentation first.")
    has_mmseg = False

try:
    from mmdet.models.builder import BACKBONES as det_BACKBONES
    from mmdet.utils import get_root_logger
    has_mmdet = True
except ImportError:
    print("If for detection, please install mmdetection first.")
    has_mmdet = False

try:
    from mmpose.models.builder import BACKBONES as pose_BACKBONES
    from mmpose.utils import get_root_logger
    has_mmpose = True
except ImportError:
    print("If for detection, please install mmdetection first.")
    has_mmpose = False


def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU', 'Sigmoid']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    else:
        return nn.GELU()


def build_norm_layer(norm_type, embed_dims):
    """Build normalization layer."""
    assert norm_type in ['BN', 'GN', 'LN2d', 'SyncBN']
    if norm_type == 'GN':
        return nn.GroupNorm(embed_dims, embed_dims, eps=1e-5)
    if norm_type == 'LN2d':
        return LayerNorm2d(embed_dims, eps=1e-6)
    if norm_type == 'SyncBN':
        return nn.SyncBatchNorm(embed_dims, eps=1e-5)
    else:
        return nn.BatchNorm2d(embed_dims, eps=1e-5)


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-6,
                 data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        assert self.data_format in ["channels_last", "channels_first"] 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self, in_chs, rd_ratio=0.25, rd_channels=None, act_layer='ReLU',
            gate_layer='Sigmoid', force_act_layer=None, rd_round_fn=None):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = build_act_layer(act_type=act_layer)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = build_act_layer(act_type=gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """
    def __init__(
            self, in_chs, out_chs, dw_kernel_size=3, stride=1, dilation=1, group_size=1, pad_type='',
            noskip=False, exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1, act_layer='ReLU',
            norm_layer='BN', se_layer=None, conv_kwargs=None, drop_path_rate=0.):
        super(InvertedResidual, self).__init__()
        # hyperparameters
        conv_kwargs = conv_kwargs or {}
        mid_chs = int(in_chs * exp_ratio)
        groups = mid_chs // group_size
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = nn.Sequential(
            build_norm_layer(norm_type=norm_layer, embed_dims=mid_chs),
            build_act_layer(act_type=act_layer),
        )

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            groups=groups, padding=pad_type, **conv_kwargs)
        self.bn2 = nn.Sequential(
            build_norm_layer(norm_type=norm_layer, embed_dims=mid_chs),
            build_act_layer(act_type=act_layer),
        )

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = build_norm_layer(norm_type=norm_layer, embed_dims=out_chs)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', hook_type='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

class MultiScalePatchEmbed(nn.Module):
    """An implementation of Multi-Scale Group Patch Embedding. """
    def __init__(self,
                 in_chs,
                 out_chs,
                 exp_ratio=4.0,
                 act_type='GELU',
                 stride=1,
                 split_list=[],
                 ):
        super(MultiScalePatchEmbed, self).__init__()
        assert len(split_list) == 3, "split_list must be equal to 3"
        self.cpe = InvertedResidual(in_chs=in_chs, out_chs=out_chs, exp_ratio=exp_ratio) # se_layer=SqueezeExcite
        self.split_list = split_list

        self.large_embed = create_conv2d(split_list[0], split_list[0], 7, stride=stride, groups=split_list[0], padding='')
        self.medium_embed = create_conv2d(split_list[1], split_list[1], 5, stride=stride, groups=split_list[1], padding='') if split_list[1] != 0 else None
        self.small_embed = create_conv2d(split_list[2], split_list[2], 3, stride=stride, groups=split_list[2], padding='') if split_list[2] != 0 else None

    def tensor_reshape(self, x, state, size=None):
        if state in 'bchw2bnc':
            return rearrange(x, 'b c h w -> b (h w) c')
        elif state in 'bnc2bchw':
            return rearrange(x, 'b (h w) c -> b c h w', h=size[0], w=size[1])

    def forward(self, x, size):
        x = self.cpe(self.tensor_reshape(x, state='bnc2bchw', size=size))

        if self.medium_embed != None and self.small_embed != None:
            large_group, medium_group, small_group = torch.split(x, split_size_or_sections=self.split_list, dim=1)
            multi_scale_embed = torch.cat([self.large_embed(large_group), self.medium_embed(medium_group), self.small_embed(small_group)], dim=1)
        elif self.medium_embed != None and self.small_embed == None:
            large_group, medium_group = torch.split(x, split_size_or_sections=self.split_list[:-1], dim=1)
            multi_scale_embed = torch.cat([self.large_embed(large_group), self.medium_embed(medium_group)], dim=1)
        elif self.medium_embed == None and self.small_embed == None:
            multi_scale_embed = self.large_embed(x)

        global_query, key_value = self.tensor_reshape(x, state='bchw2bnc'), self.tensor_reshape(multi_scale_embed, state='bchw2bnc')
        return global_query, key_value


class MultiScaleAttention(nn.Module):
    """An implementation of Multi-Scale Self-Attention. """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., stride=1, split_list=[]):
        super(MultiScaleAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.mspe = MultiScalePatchEmbed(in_chs=dim, out_chs=dim, stride=stride, split_list=[x * self.dim_head for x in split_list])

        self.query = nn.Linear(dim, dim, bias=qkv_bias)
        self.key_value = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, size=None):
        restore_shape = x.shape[:-1]
        global_query, key_value = self.mspe(x, size)  # b, n, c
        global_query, key_value = self.query(global_query), self.key_value(key_value)

        # split multi-head
        global_query = rearrange(global_query, 'b n (nh hd) -> b nh n hd', nh=self.num_heads)
        key_value = rearrange(key_value, 'b n (ng nh hd) -> ng b nh n hd', ng=2, nh=self.num_heads)
        key, value = key_value.unbind(0)

        attn = (global_query @ key.transpose(-2, -1)) * self.scale  # b, nh, n, n
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # merge multi-head
        x = (attn @ value).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MultiScaleAttentionBlock(nn.Module):
    """An implementation of Multi-Scale Self-Attention Block. """
    def __init__(self, dim, num_heads, mlp_ratios=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, stride=1, split_list=[]):
        super(MultiScaleAttentionBlock, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            stride=stride,
            split_list=split_list
        )
        self.ls1 = LayerScale(dim, init_values=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratios),
            act_layer=act_layer)

        self.ls2 = LayerScale(dim, init_values=1e-6)

    def forward(self, x, size=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), size=size)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class RelPosMlp(nn.Module):
    def __init__(
            self,
            window_size,
            num_heads=8,
            hidden_dim=128,
            prefix_tokens=0,
            mode='cr',
            pretrained_window_size=(0, 0)
    ):
        super().__init__()
        self.window_size = window_size
        self.window_area = self.window_size[0] * self.window_size[1]
        self.prefix_tokens = prefix_tokens
        self.num_heads = num_heads
        self.bias_shape = (self.window_area,) * 2 + (num_heads,)
        if mode == 'swin':
            self.bias_act = nn.Sigmoid()
            self.bias_gain = 16
            mlp_bias = (True, False)
        elif mode == 'rw':
            self.bias_act = nn.Tanh()
            self.bias_gain = 4
            mlp_bias = True
        else:
            self.bias_act = nn.Identity()
            self.bias_gain = None
            mlp_bias = True

        self.mlp = Mlp(
            2,  # x, y
            hidden_features=hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(0.125, 0.)
        )

        self.register_buffer(
            "relative_position_index",
            gen_relative_position_index(window_size),
            persistent=False)

        # get relative_coords_table
        self.register_buffer(
            "rel_coords_log",
            gen_relative_log_coords(window_size, pretrained_window_size, mode=mode),
            persistent=False)

    def get_bias(self) -> torch.Tensor:
        relative_position_bias = self.mlp(self.rel_coords_log)
        if self.relative_position_index is not None:
            relative_position_bias = relative_position_bias.view(-1, self.num_heads)[
                self.relative_position_index.view(-1)]  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.view(self.bias_shape)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)
        relative_position_bias = self.bias_act(relative_position_bias)
        if self.bias_gain is not None:
            relative_position_bias = self.bias_gain * relative_position_bias
        if self.prefix_tokens:
            relative_position_bias = F.pad(relative_position_bias, [self.prefix_tokens, 0, self.prefix_tokens, 0])
        return relative_position_bias.unsqueeze(0).contiguous()

    def forward(self, attn, shared_rel_pos: Optional[torch.Tensor] = None):
        return attn + self.get_bias()


def gen_relative_position_index(
        q_size: Tuple[int, int],
        k_size: Tuple[int, int] = None,
        class_token: bool = False) -> torch.Tensor:
    # Adapted with significant modifications from Swin / BeiT codebases
    # get pair-wise relative position index for each token inside the window
    q_coords = torch.stack(torch.meshgrid([torch.arange(q_size[0]), torch.arange(q_size[1])])).flatten(1)  # 2, Wh, Ww
    if k_size is None:
        k_coords = q_coords
        k_size = q_size
    else:
        # different q vs k sizes is a WIP
        k_coords = torch.stack(torch.meshgrid([torch.arange(k_size[0]), torch.arange(k_size[1])])).flatten(1)
    relative_coords = q_coords[:, :, None] - k_coords[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0)  # Wh*Ww, Wh*Ww, 2
    _, relative_position_index = torch.unique(relative_coords.view(-1, 2), return_inverse=True, dim=0)

    if class_token:
        # handle cls to token & token 2 cls & cls to cls as per beit for rel pos bias
        # NOTE not intended or tested with MLP log-coords
        max_size = (max(q_size[0], k_size[0]), max(q_size[1], k_size[1]))
        num_relative_distance = (2 * max_size[0] - 1) * (2 * max_size[1] - 1) + 3
        relative_position_index = F.pad(relative_position_index, [1, 0, 1, 0])
        relative_position_index[0, 0:] = num_relative_distance - 3
        relative_position_index[0:, 0] = num_relative_distance - 2
        relative_position_index[0, 0] = num_relative_distance - 1

    return relative_position_index.contiguous()


def gen_relative_log_coords(
        win_size: Tuple[int, int],
        pretrained_win_size: Tuple[int, int] = (0, 0),
        mode='swin',
):
    assert mode in ('swin', 'cr', 'rw')
    # as per official swin-v2 impl, supporting timm specific 'cr' and 'rw' log coords as well
    relative_coords_h = torch.arange(-(win_size[0] - 1), win_size[0], dtype=torch.float32)
    relative_coords_w = torch.arange(-(win_size[1] - 1), win_size[1], dtype=torch.float32)
    relative_coords_table = torch.stack(torch.meshgrid([relative_coords_h, relative_coords_w]))
    relative_coords_table = relative_coords_table.permute(1, 2, 0).contiguous()  # 2*Wh-1, 2*Ww-1, 2
    if mode == 'swin':
        if pretrained_win_size[0] > 0:
            relative_coords_table[:, :, 0] /= (pretrained_win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (pretrained_win_size[1] - 1)
        else:
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            1.0 + relative_coords_table.abs()) / math.log2(8)
    else:
        if mode == 'rw':
            # cr w/ window size normalization -> [-1,1] log coords
            relative_coords_table[:, :, 0] /= (win_size[0] - 1)
            relative_coords_table[:, :, 1] /= (win_size[1] - 1)
            relative_coords_table *= 8  # scale to -8, 8
            relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
                1.0 + relative_coords_table.abs())
            relative_coords_table /= math.log2(9)   # -> [-1, 1]
        else:
            # mode == 'cr'
            relative_coords_table = torch.sign(relative_coords_table) * torch.log(
                1.0 + relative_coords_table.abs())

    return relative_coords_table

class GlobalScaleAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., dilation=8):
        super().__init__()
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scale = self.dim_head ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dilation_size = to_2tuple(dilation)
        self.rel_pos = RelPosMlp(window_size=self.dilation_size, num_heads=num_heads, hidden_dim=512)

    def attn(self, x):
        restore_shape = x.shape[:-1]
        # split multi-head
        q, k, v = rearrange(self.qkv(x), 'b n (ng nh hd) -> ng b nh n hd', ng=3, nh=self.num_heads).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.rel_pos is not None:
            attn = self.rel_pos(attn)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(restore_shape + (-1,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward(self, x, size=None):  # size=(H, W)
        x = rearrange(x, 'b (dh hp dw wp) c -> (b hp wp) (dh dw) c', dh=self.dilation_size[0], dw=self.dilation_size[1],
                      hp=size[0] // self.dilation_size[0], wp=size[1] // self.dilation_size[1])
        x = self.attn(x)
        x = rearrange(x, '(b hp wp) (dh dw) c -> b (dh hp dw wp) c ', dh=self.dilation_size[0], dw=self.dilation_size[1],
                      hp=size[0] // self.dilation_size[0], wp=size[1] // self.dilation_size[1])
        return x


class GlobalScaleAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratios=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, dilation=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            dilation=dilation
        )
        self.ls1 = LayerScale(dim, init_values=1e-6)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratios),
            act_layer=act_layer)

        self.ls2 = LayerScale(dim, init_values=1e-6)

    def forward(self, x, size=None):
        x = x + self.drop_path(self.ls1(self.attn(self.norm1(x), size=size)))
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))))
        return x


class ScopeViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratios=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, mssa_stride=1, mssa_split=[], gsda_dilation=8, reshape_state=0):
        super().__init__()

        self.reshape_state = reshape_state

        self.mssa = MultiScaleAttentionBlock(dim, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, stride=mssa_stride, split_list=mssa_split)
        self.gsda = GlobalScaleAttentionBlock(dim, num_heads=num_heads, mlp_ratios=mlp_ratios, qkv_bias=qkv_bias,
                 drop_path=drop_path, act_layer=act_layer, norm_layer=norm_layer, dilation=gsda_dilation)

    def tensor_reshape(self, x, state, size=None):
        if state in 'bchw2bnc':
            return rearrange(x, 'b c h w -> b (h w) c')
        elif state in 'bnc2bchw':
            return rearrange(x, 'b (h w) c -> b c h w', h=size[0], w=size[1])

    def forward(self, x, size=None):
        if self.reshape_state == 0 or -2:
            x = self.tensor_reshape(x, state='bchw2bnc')
        x = self.mssa(x, size=size)
        x = self.gsda(x, size=size)
        if self.reshape_state == -1 or -2:
            x = self.tensor_reshape(x, state='bnc2bchw', size=size)
        return x


class ConvPatchEmbed(nn.Module):
    """An implementation of Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of PatchEmbed.
            Defaults to 3.
        stride (int): The conv stride of PatchEmbed. Defaults to 2.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 norm_type='BN'):
        super(ConvPatchEmbed, self).__init__()

        self.projection = nn.Conv2d(
            in_channels, embed_dims, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2)
        self.norm = build_norm_layer(norm_type, embed_dims)

    def forward(self, x):
        x = self.projection(x)
        x = self.norm(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class StackConvPatchEmbed(nn.Module):
    """An implementation of Stack Conv patch embedding layer.

    Args:
        in_features (int): The feature dimension.
        embed_dims (int): The output dimension of PatchEmbed.
        kernel_size (int): The conv kernel size of stack patch embedding.
            Defaults to 3.
        stride (int): The conv stride of stack patch embedding.
            Defaults to 2.
        act_type (str): The activation in PatchEmbed. Defaults to 'GELU'.
        norm_type (str): The type of normalization layer. Defaults to 'BN'.
    """

    def __init__(self,
                 in_channels,
                 embed_dims,
                 kernel_size=3,
                 stride=2,
                 act_type='GELU',
                 norm_type='BN'):
        super(StackConvPatchEmbed, self).__init__()

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dims // 2, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims // 2),
            build_act_layer(act_type),
            nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=kernel_size,
                stride=stride, padding=kernel_size // 2),
            build_norm_layer(norm_type, embed_dims),
        )

    def forward(self, x):
        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        return x, out_size


class ScopeViT(nn.Module):
    arch_zoo = {
        **dict.fromkeys(['xt', 'x-tiny', 'xtiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [1, 2, 4, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mssa_stride':[4, 2, 1, 1],
                         'mssa_split':[[1, 1, 0],[2, 1, 1],[4, 3, 1],[8, 6, 2]],
                         'gsda_dilation': [8, 8, 8, 8]}),
        **dict.fromkeys(['t', 'tiny'],
                        {'embed_dims': [32, 64, 128, 256],
                         'depths': [2, 2, 5, 2],
                         'num_heads': [2, 4, 8, 16],
                         'mssa_stride':[4, 2, 1, 1],
                         'mssa_split':[[1, 1, 0],[2, 1, 1],[4, 3, 1],[8, 6, 2]],
                         'gsda_dilation': [8, 8, 8, 8]}),
        **dict.fromkeys(['s', 'small'],
                        {'embed_dims': [64, 128, 256, 512],
                         'depths': [1, 2, 3, 1],
                         'num_heads': [2, 4, 8, 16],
                         'mssa_stride': [4, 2, 1, 1],
                         'mssa_split': [[1, 1, 0], [2, 1, 1], [4, 3, 1], [8, 6, 2]],
                         'gsda_dilation': [8, 8, 8, 8]}),
    }  # yapf: disable

    def __init__(self,
                 arch='tiny',
                 in_channels=3,
                 num_classes=10,
                 mlp_ratios=4.,
                 drop_path_rate=0.,
                 head_init_scale=1.,
                 patch_sizes=[3, 3, 3, 3],
                 stem_norm_type='BN',
                 conv_norm_type='BN',
                 patchembed_types=['ConvEmbed', 'Conv', 'Conv', 'Conv',],
                 fork_feat=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {'embed_dims', 'depths', 'num_heads', 'mssa_stride', 'mssa_split', 'gsda_dilation'}
            assert isinstance(arch, dict) and set(arch) == essential_keys, \
                f'Custom arch needs a dict with keys {essential_keys}'
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']
        self.depths = self.arch_settings['depths']
        self.num_heads = self.arch_settings['num_heads']
        self.mssa_stride = self.arch_settings['mssa_stride']
        self.mssa_split = self.arch_settings['mssa_split']
        self.gsda_dilation = self.arch_settings['gsda_dilation']

        self.num_stages = len(self.depths)

        self.use_layer_norm = stem_norm_type == 'LN'
        assert len(patchembed_types) == self.num_stages
        self.fork_feat = fork_feat
        self.frozen_stages = frozen_stages

        total_depth = sum(self.depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]  # stochastic depth decay rule

        cur_block_idx = 0
        for i, depth in enumerate(self.depths):
            if i == 0 and patchembed_types[i] == "ConvEmbed":
                assert patch_sizes[i] <= 3
                patch_embed = StackConvPatchEmbed(
                    in_channels=in_channels,
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    act_type='GELU',
                    norm_type=conv_norm_type,
                )
            else:
                patch_embed = ConvPatchEmbed(
                    in_channels=in_channels if i == 0 else self.embed_dims[i - 1],
                    embed_dims=self.embed_dims[i],
                    kernel_size=patch_sizes[i],
                    stride=patch_sizes[i] // 2 + 1,
                    norm_type=conv_norm_type)

            blocks = nn.ModuleList([
                ScopeViTBlock(
                    dim=self.embed_dims[i],
                    num_heads=self.num_heads[i],
                    mlp_ratios=mlp_ratios,
                    qkv_bias=False,
                    drop_path=dpr[cur_block_idx + j],
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                    mssa_stride=self.mssa_stride[i],
                    mssa_split=self.mssa_split[i],
                    gsda_dilation=self.gsda_dilation[i],
                    reshape_state=-2 if j+1==depth and j==0 else (-1 if j+1==depth else j) ,
                ) for j in range(depth)
            ])
            cur_block_idx += depth
            norm = build_norm_layer(stem_norm_type, self.embed_dims[i])

            self.add_module(f'patch_embed{i + 1}', patch_embed)
            self.add_module(f'blocks{i + 1}', blocks)
            self.add_module(f'norm{i + 1}', norm)

        if self.fork_feat:
            self.head = nn.Identity()
        else:
            # Classifier head
            self.num_classes = num_classes
            self.head = nn.Linear(self.embed_dims[-1], num_classes) \
                if num_classes > 0 else nn.Identity()

            # init for classification
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights(pretrained)

    def _init_weights(self, m):
        """ Init for timm image classification """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        """ Init for mmdetection or mmsegmentation by loading pre-trained weights """
        logger = get_root_logger()
        if self.init_cfg is None and pretrained is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            pass
        else:
            if self.init_cfg is not None:
                assert 'checkpoint' in self.init_cfg, f'Only support specify ' \
                                                      f'`Pretrained` in `init_cfg` in ' \
                                                      f'{self.__class__.__name__} '
                ckpt_path = self.init_cfg['checkpoint']
            elif pretrained is not None:
                ckpt_path = pretrained

            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            state_dict = _state_dict
            missing_keys, unexpected_keys = \
                self.load_state_dict(state_dict, False)
            # show for debug
            # print('missing_keys: ', missing_keys)
            # print('unexpected_keys: ', unexpected_keys)

    def _freeze_stages(self):
        for i in range(0, self.frozen_stages + 1):
            # freeze patch embed
            m = getattr(self, f'patch_embed{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze blocks
            m = getattr(self, f'blocks{i + 1}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            # freeze norm
            m = getattr(self, f'norm{i + 1}')
            m.eval()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return dict()

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        outs = []
        for i in range(self.num_stages):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            blocks = getattr(self, f'blocks{i + 1}')
            norm = getattr(self, f'norm{i + 1}')

            x, hw_shape = patch_embed(x)
            for block in blocks:
                x = block(x, size=hw_shape)
            if self.use_layer_norm:
                x = x.flatten(2).transpose(1, 2)
                x = norm(x)
                x = x.reshape(-1, *hw_shape,
                              blocks.out_channels).permute(0, 3, 1, 2).contiguous()
            else:
                x = norm(x)
            if self.fork_feat:
                outs.append(x)

        if self.fork_feat:
            # output the features of four stages for dense prediction
            return outs
        else:
            # output only the last layer for image classification
            return x

    def forward_head(self, x):
        return self.head(x.mean(dim=[2, 3]))

    def forward(self, x):
        x = self.forward_features(x)
        if self.fork_feat:
            # for dense prediction
            return x
        else:
            # for image classification
            return self.forward_head(x)




def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 256, 256),
        'crop_pct': 0.90, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }

default_cfgs = {
    'scopevit_xt': _cfg(crop_pct=0.9),
    'scopevit_t': _cfg(crop_pct=0.9),
    'scopevit_s': _cfg(crop_pct=0.9),
}

@register_model
def scopevit_xtiny(pretrained=False, **kwargs):
    model = ScopeViT(arch='xtiny', **kwargs)
    model.default_cfg = default_cfgs['scopevit_xt']
    return model

@register_model
def scopevit_tiny(pretrained=False, **kwargs):
    model = ScopeViT(arch='tiny', **kwargs)
    model.default_cfg = default_cfgs['scopevit_t']
    return model

@register_model
def scopevit_small(pretrained=False, **kwargs):
    model = ScopeViT(arch='small', **kwargs)
    model.default_cfg = default_cfgs['scopevit_s']
    return model


if has_mmdet:
    """
    The following models are for dense prediction tasks based on
    mmdetection, mmsegmentation, and mmpose.
    """
    @det_BACKBONES.register_module()
    class ScopeViT_feat(ScopeViT):
        """
        ScopeViT Model for Dense Prediction.
        """
        def __init__(self, **kwargs):
            super().__init__(fork_feat=True, **kwargs)

if has_mmseg:
    @seg_BACKBONES.register_module()
    class ScopeViT_feat(ScopeViT):
        """
        ScopeViT Model for Dense Prediction.
        """
        def __init__(self, **kwargs):
            super().__init__(fork_feat=True, **kwargs)

if has_mmpose:
    @pose_BACKBONES.register_module()
    class ScopeViT_feat(ScopeViT):
        """
        ScopeViT Model for Dense Prediction.
        """
        def __init__(self, **kwargs):
            super().__init__(fork_feat=True, **kwargs)

