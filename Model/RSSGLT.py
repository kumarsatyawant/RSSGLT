import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Overlap_Patch_Embedding(nn.Module):
    def __init__(self, in_channel, hidden_proj_dim, kernel_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(in_channel, hidden_proj_dim, kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

    def forward(self, f_map):
        x = self.proj(f_map)
        x = x.flatten(2)  # (n_samples, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (n_samples, n_patches, embed_dim)

        x = self.norm(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_proj_dim, seq_reduction_ratio, num_attention_heads, dropout_rate=0.0):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_proj_dim = hidden_proj_dim
        self.scale = (hidden_proj_dim // num_attention_heads) ** -0.5
        self.sr_ratio = seq_reduction_ratio
        
        self.attention_head_size = int(self.hidden_proj_dim / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.q = nn.Linear(self.hidden_proj_dim, self.all_head_size)
        self.kv = nn.Linear(self.hidden_proj_dim, self.all_head_size*2)
        self.proj = nn.Linear(self.hidden_proj_dim, self.all_head_size)
        
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj_drop = nn.Dropout(dropout_rate)
        
        if seq_reduction_ratio > 1:
            self.sr = nn.Conv2d(self.hidden_proj_dim, self.hidden_proj_dim, kernel_size=self.sr_ratio, stride=self.sr_ratio)
            self.norm = nn.LayerNorm(self.hidden_proj_dim)

        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)

    
    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        hidden_states = hidden_states.view(*new_shape)
        return hidden_states.permute(0, 2, 1, 3)
    
    def forward(self, x, pos_2D):
        B, N, C = x.shape
        # print(x.shape)
        q = self.transpose_for_scores(self.q(x))
        
        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = x.shape
            # Reshape to (batch_size, num_channels, height, width)
            x = x.permute(0, 2, 1).reshape(batch_size, num_channels, int(math.sqrt(N)), int(math.sqrt(N)))
            # Apply sequence reduction
            x = self.sr(x)
            # Reshape back to (batch_size, seq_len, num_channels)
            x = x.reshape(batch_size, num_channels, -1).permute(0, 2, 1)
            x = self.norm(x)
            

        kv = self.kv(x)
        kv = kv.reshape(B, kv.shape[1], 2, self.num_attention_heads, C//self.num_attention_heads).permute(2, 0, 3, 1, 4)

        k = kv[0]
        v = kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = (1 - self.alpha) * attn + self.alpha * pos_2D
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class DW_Conv(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, stride=1, padding=1, groups=in_channel)
        self.activ = nn.GELU()

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, int(math.sqrt(N)), int(math.sqrt(N)))
        x = self.dwconv(x)
        x = self.activ(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class MLP_block(nn.Module):
    def __init__(self, in_channel, out_channel, dropout_rate = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_channel, out_channel)
        self.fc2 = nn.Linear(out_channel, in_channel)
        self.drop = nn.Dropout(p=dropout_rate)

        self.dwconv = DW_Conv(out_channel)

    def forward(self, x):
        x = self.fc1(x)

        x = self.dwconv(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class Block(nn.Module):
    def __init__(self, hidden_proj_dim, seq_reduction_ratio, num_attention_heads, mlp_expan_ratio, drop_path= 0., dropout_rate=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

        self.attn = SelfAttention(hidden_proj_dim, seq_reduction_ratio, num_attention_heads, dropout_rate=dropout_rate)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = nn.LayerNorm(hidden_proj_dim, eps=1e-6)

        self.mlp = MLP_block(in_channel=hidden_proj_dim, out_channel=int(hidden_proj_dim*mlp_expan_ratio), dropout_rate=dropout_rate)

    def forward(self, x, pos_2D):
        x = x + self.drop_path(self.attn(self.norm1(x), pos_2D))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

#2D_Positional Attention
def normpdf(x, std):
    var = std**2
    denom = (2*math.pi*var)**.5
    return torch.exp(-(x)**2/(2*var))/denom

def pdist(sample_1, sample_2, norm=2, eps=0):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared).to(torch.float))

def position(H, W, sr_ratio=1, std=1):
    N = H*W
    h = int(H/sr_ratio)
    w = int(W/sr_ratio)
    n = int(h*w)
    yv, xv = torch.meshgrid([torch.arange(H), torch.arange(W)])
    grid0 = torch.stack((xv, yv), 2).view((H, W, 2)).float().transpose(0, 1).cuda()
    grid01 = grid0[:,:,0:1].permute(2,0,1)
    grid02 = grid0[:,:,1:2].permute(2,0,1)
    ymax = F.max_pool2d(grid01, kernel_size=sr_ratio,stride=sr_ratio)
    ymin = -F.max_pool2d(-grid01, kernel_size=sr_ratio,stride=sr_ratio)
    y = ((ymax+ymin) / 2).resize(n,1,1)
    xmax = F.max_pool2d(grid02, kernel_size=sr_ratio,stride=sr_ratio)
    xmin = -F.max_pool2d(-grid02, kernel_size=sr_ratio,stride=sr_ratio)
    x = ((xmax+xmin) / 2).resize(n,1,1)
    grid1 = torch.cat([y,x],2).resize(h,w,2)
    grid0 = grid0.resize(N, 2)
    grid1 = grid1.resize(n, 2)
    dist = pdist(grid0, grid1, norm=2)
    dist1 = dist / (sr_ratio*2)
    dist2 = normpdf(dist1, std)
    dist3 = 10 * dist2.softmax(dim=-1)
    return dist3

class Mix_Transformer(nn.Module):
    def __init__(self, input_channel=3, dropout_rate=0.0):
        super().__init__()
          # patch_embeds = [hidden_proj_dim, kernel_size, stride, padding]
        self.patch_embed_dim = [[64, 7, 4, 3], [128, 3, 2, 1], [320, 3, 2, 1], [512, 3, 2, 1]]

        # stages = [reduction_ratio, attention_heads, mlp_expansion_ration, num_encoder_blocks]
        self.stages_dim = [[8, 1, 4, 3], [4, 2, 4, 4], [2, 5, 4, 6], [1, 8, 4, 3]]

        self.patch_embed1 = Overlap_Patch_Embedding(input_channel, self.patch_embed_dim[0][0], self.patch_embed_dim[0][1], self.patch_embed_dim[0][2], self.patch_embed_dim[0][3])
        self.patch_embed2 = Overlap_Patch_Embedding(self.patch_embed_dim[0][0], self.patch_embed_dim[1][0], self.patch_embed_dim[1][1], self.patch_embed_dim[1][2], self.patch_embed_dim[1][3])
        self.patch_embed3 = Overlap_Patch_Embedding(self.patch_embed_dim[1][0], self.patch_embed_dim[2][0], self.patch_embed_dim[2][1], self.patch_embed_dim[2][2], self.patch_embed_dim[2][3])
        self.patch_embed4 = Overlap_Patch_Embedding(self.patch_embed_dim[2][0], self.patch_embed_dim[3][0], self.patch_embed_dim[3][1], self.patch_embed_dim[3][2], self.patch_embed_dim[3][3])

        self.norm1 = nn.LayerNorm(self.patch_embed_dim[0][0], eps=1e-6)
        self.norm2 = nn.LayerNorm(self.patch_embed_dim[1][0], eps=1e-6)
        self.norm3 = nn.LayerNorm(self.patch_embed_dim[2][0], eps=1e-6)
        self.norm4 = nn.LayerNorm(self.patch_embed_dim[3][0], eps=1e-6)

        # self.block1 = nn.ModuleList(
        #     [
        #         Block(
        #             hidden_size, 
        #             reduction_ratio, 
        #             num_attention_head, 
        #             mlp_expansion_ratio,
        #             dropout_rate=dropout_rate
        #         )
        #         for _ in range(depth)
        #     ]
        # )

        self.block1 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[0][0], 
                    self.stages_dim[0][0], 
                    self.stages_dim[0][1], 
                    self.stages_dim[0][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[0][3])
            ]
        )

        self.block2 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[1][0], 
                    self.stages_dim[1][0], 
                    self.stages_dim[1][1], 
                    self.stages_dim[1][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[1][3])
            ]
        )

        self.block3 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[2][0], 
                    self.stages_dim[2][0], 
                    self.stages_dim[2][1], 
                    self.stages_dim[2][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[2][3])
            ]
        )

        self.block4 = nn.ModuleList(
            [
                Block(
                    self.patch_embed_dim[3][0], 
                    self.stages_dim[3][0], 
                    self.stages_dim[3][1], 
                    self.stages_dim[3][2],
                    dropout_rate=dropout_rate
                )
                for _ in range(self.stages_dim[3][3])
            ]
        )

    def forward(self, x):
        # print(x.shape)
        x = self.patch_embed1(x)
        pos_2D_1 = position(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), 8, std=0.5)
        for i, block in enumerate(self.block1):
            x = block(x, pos_2D_1)
            

        x = self.norm1(x)
        block_1_out = x.transpose(1, 2)
        block_1_out = torch.reshape(block_1_out, (block_1_out.shape[0], block_1_out.shape[1], int(math.sqrt(block_1_out.shape[2])), int(math.sqrt(block_1_out.shape[2]))))
        # print(block_1_out.shape)

        x = self.patch_embed2(block_1_out)
        pos_2D_2 = position(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), 4, std=0.5)
        for i, block in enumerate(self.block2):
            x = block(x, pos_2D_2)
            

        x = self.norm2(x)
        block_2_out = x.transpose(1, 2)
        block_2_out = torch.reshape(block_2_out, (block_2_out.shape[0], block_2_out.shape[1], int(math.sqrt(block_2_out.shape[2])), int(math.sqrt(block_2_out.shape[2]))))
        # print(block_2_out.shape)

        x = self.patch_embed3(block_2_out)
        pos_2D_3 = position(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), 2, std=0.4)
        for i, block in enumerate(self.block3):
            x = block(x, pos_2D_3)
            

        x = self.norm3(x)
        block_3_out = x.transpose(1, 2)
        block_3_out = torch.reshape(block_3_out, (block_3_out.shape[0], block_3_out.shape[1], int(math.sqrt(block_3_out.shape[2])), int(math.sqrt(block_3_out.shape[2]))))
        # print(block_3_out.shape)

        x = self.patch_embed4(block_3_out)
        pos_2D_4 = position(int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), 1, std=0.4)
        for i, block in enumerate(self.block4):
            x = block(x, pos_2D_4)
            

        x = self.norm4(x)
        block_4_out = x.transpose(1, 2)
        block_4_out = torch.reshape(block_4_out, (block_4_out.shape[0], block_4_out.shape[1], int(math.sqrt(block_4_out.shape[2])), int(math.sqrt(block_4_out.shape[2]))))
        # print(block_4_out.shape)


        return block_1_out, block_2_out, block_3_out, block_4_out

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class GlobalLocalAttention(nn.Module):
    def __init__(self,
                 dim=256,
                 num_heads=16,
                 qkv_bias=False,
                 window_size=8,
                 relative_pos_embedding=True
                 ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5
        self.ws = window_size

        self.qkv = Conv(dim, 3*dim, kernel_size=1, bias=qkv_bias)
        self.local1 = ConvBN(dim, dim, kernel_size=3)
        self.local2 = ConvBN(dim, dim, kernel_size=1)
        self.local3 = ConvBN(dim, dim, kernel_size=5)
        self.proj = SeparableConvBN(dim, dim, kernel_size=window_size)

        self.attn_x = nn.AvgPool2d(kernel_size=(window_size, 1), stride=1,  padding=(window_size//2 - 1, 0))
        self.attn_y = nn.AvgPool2d(kernel_size=(1, window_size), stride=1, padding=(0, window_size//2 - 1))

        self.relative_pos_embedding = relative_pos_embedding

        if self.relative_pos_embedding:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(self.ws)
            coords_w = torch.arange(self.ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += self.ws - 1  # shift to start from 0
            relative_coords[:, :, 1] += self.ws - 1
            relative_coords[:, :, 0] *= 2 * self.ws - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

            trunc_normal_(self.relative_position_bias_table, std=.02)

    def pad(self, x, ps):
        _, _, H, W = x.size()
        if W % ps != 0:
            x = F.pad(x, (0, ps - W % ps), mode='reflect')
        if H % ps != 0:
            x = F.pad(x, (0, 0, 0, ps - H % ps), mode='reflect')
        return x

    def pad_out(self, x):
        x = F.pad(x, pad=(0, 1, 0, 1), mode='reflect')
        return x

    def forward(self, x):
        B, C, H, W = x.shape

        local = self.local2(x) + self.local1(x) + self.local3(x)

        x = self.pad(x, self.ws)
        B, C, Hp, Wp = x.shape
        qkv = self.qkv(x)

        q, k, v = rearrange(qkv, 'b (qkv h d) (hh ws1) (ww ws2) -> qkv (b hh ww) h (ws1 ws2) d', h=self.num_heads,
                            d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, qkv=3, ws1=self.ws, ws2=self.ws)

        dots = (q @ k.transpose(-2, -1)) * self.scale

        if self.relative_pos_embedding:
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
                self.ws * self.ws, self.ws * self.ws, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            dots += relative_position_bias.unsqueeze(0)

        attn = dots.softmax(dim=-1)
        attn = attn @ v

        attn = rearrange(attn, '(b hh ww) h (ws1 ws2) d -> b (h d) (hh ws1) (ww ws2)', h=self.num_heads,
                         d=C//self.num_heads, hh=Hp//self.ws, ww=Wp//self.ws, ws1=self.ws, ws2=self.ws)

        attn = attn[:, :, :H, :W]

        out = self.attn_x(F.pad(attn, pad=(0, 0, 0, 1), mode='reflect')) + \
              self.attn_y(F.pad(attn, pad=(0, 1, 0, 0), mode='reflect'))

        out = out + local
        out = self.pad_out(out)
        out = self.proj(out)
        # print(out.size())
        out = out[:, :, :H, :W]

        return out


class GLTB_Block(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Refinement_Module(nn.Module):
    def __init__(self, in_feature_dim, out_feature_dim):
        super().__init__()
        self.reduce_dim = nn.Conv2d(in_feature_dim, out_feature_dim, kernel_size=1, stride=1, bias=False)

        self.refinement_block = nn.Sequential(
            nn.BatchNorm2d(out_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out_feature_dim),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(out_feature_dim, out_feature_dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_feature_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        temp_x = self.reduce_dim(x)
        x = self.refinement_block(temp_x) * temp_x

        return x


class Fusion(nn.Module):
    def __init__(self, in_channels, decode_channels):
        super(Fusion, self).__init__()
        self.res_refinement_module = Refinement_Module(in_channels, decode_channels)
        self.decode_feature_refinement_module = Refinement_Module(decode_channels, decode_channels)

        self.conv_1 = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)
        self.conv_2 = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.decode_feature_refinement_module(x)
        res = self.res_refinement_module(res)

        fuse = x + res
        fuse_out = self.conv_2(self.conv_1(fuse)) + fuse

        return fuse_out



class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.feature_fusion = Fusion(in_channels, decode_channels)

        self.branch_1 = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.branch_2 = SeparableConvBN(decode_channels, decode_channels, kernel_size=5)
        self.branch_3 = nn.Sequential(
            ConvBNReLU(decode_channels, decode_channels, kernel_size=3),
            ConvBNReLU(decode_channels, decode_channels, kernel_size=5)
        )

        self.conv_block_1 = ConvBN(int(decode_channels*2), decode_channels, kernel_size=1)
        self.conv_block_2 = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)

        self.weight_block = nn.Sequential(
            ConvBNReLU(decode_channels, 1, kernel_size=1)
        )


    def forward(self, x, res):
        x = self.feature_fusion(x, res)
        shortcut = self.shortcut(x)

        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        concat_branch_1_2 = torch.cat((branch_1, branch_2), dim=1)
        concat_branch_1_2 = self.conv_block_2(self.conv_block_1(concat_branch_1_2))
        
        concat_branch_1_2_add = concat_branch_1_2 + shortcut + branch_3

        x = self.weight_block(concat_branch_1_2_add) * concat_branch_1_2_add

        return x


class Decoder(nn.Module):
    def __init__(self,
                 encoder_channels=(64, 128, 320, 512),
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6):
        super(Decoder, self).__init__()

        self.pre_conv = ConvBN(encoder_channels[-1], decode_channels, kernel_size=1)
        self.b4 = GLTB_Block(dim=decode_channels, num_heads=8, window_size=window_size)

        self.b3 = GLTB_Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p3 = Fusion(encoder_channels[-2], decode_channels)

        self.b2 = GLTB_Block(dim=decode_channels, num_heads=8, window_size=window_size)
        self.p2 = Fusion(encoder_channels[-3], decode_channels)


        self.p1 = FeatureRefinementHead(encoder_channels[-4], decode_channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(decode_channels, decode_channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(decode_channels, num_classes, kernel_size=1))

    def forward(self, res1, res2, res3, res4, h, w):
        x = self.b4(self.pre_conv(res4))
        x = self.p3(x, res3)
        x = self.b3(x)

        x = self.p2(x, res2)
        x = self.b2(x)

        x = self.p1(x, res1)

        x = self.segmentation_head(x)
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

        return x


class RSSGLT(nn.Module):
    def __init__(self,
                 decode_channels=64,
                 dropout=0.1,
                 window_size=8,
                 num_classes=6
                 ):
        super().__init__()

        self.mix_trans = Mix_Transformer(dropout_rate=dropout)
        encoder_channels_list = [64, 128, 320, 512]

        self.decoder = Decoder(encoder_channels_list, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        # print(x.shape)
        h, w = x.size()[-2:]
        block_1_out, block_2_out, block_3_out, block_4_out = self.mix_trans(x)

        x = self.decoder(block_1_out, block_2_out, block_3_out, block_4_out, h, w)

        return x