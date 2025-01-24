import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from helpers import DropPath

class FeatureExtraction(nn.Module):
    r""" Initial Feature Extraction
    Args:
        in_chans (tuple(int)): Input channels
    """

    def __init__(self, in_channels=[1, 16, 16, 32, 32]):
        super().__init__()
        # number of conv block
        num_blocks = len(in_channels) - 1
        
        # conv, pool, conv, pool
        self.layers = nn.ModuleList()

        # in channels
        out_channels = in_channels[1:]
        kernel_size = [(19, 17), (17, 15), (15, 13), (13, 11), (11, 9), (9, 7), (7, 5), (5, 3)]
        kernel_size = kernel_size[len(kernel_size) - num_blocks:]
        # create layers
        for i in range(num_blocks):
            conv = nn.Conv1d(in_channels[i], out_channels[i], kernel_size[i][1], padding="same")
            act = nn.ReLU()
            pool = nn.MaxPool1d(kernel_size=3, stride=2)
            
            self.layers.append(conv)
            self.layers.append(act)
            self.layers.append(pool)

    def forward(self, x):
        for layers in self.layers:
            x = layers(x)        
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)     
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FocalModulation(nn.Module):
    def __init__(self, dim, focal_window, focal_level, focal_factor=6, bias=True, proj_drop=0., normalize_modulator=False):
        super().__init__()

        self.dim = dim
        self.focal_window = focal_window
        self.focal_level = focal_level
        self.focal_factor = focal_factor
        self.normalize_modulator = normalize_modulator

        self.f = nn.Linear(dim, 2*dim + (self.focal_level+1), bias=bias)
        self.h = nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=bias, padding="same")

        self.act = nn.GELU()
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.focal_layers = nn.ModuleList()
                
        self.kernel_sizes = []
        for k in range(self.focal_level):
            kernel_size = self.focal_factor*k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=kernel_size, stride=1, 
                    groups=dim, padding=kernel_size//2, bias=False),
                    nn.GELU(),
                    )
                )              
            self.kernel_sizes.append(kernel_size)          

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, L, C)
        """
        C = x.shape[-1]

        # pre linear projection
        x = self.f(x).permute(0, 2, 1).contiguous() # (B, L, C) -> (B, C, L)
        q, ctx, self.gates = torch.split(x, (C, C, self.focal_level+1), 1)
        
        # context aggreation
        ctx_all = 0 
        for l in range(self.focal_level):         
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx*self.gates[:, l:l+1]
        ctx_global = self.act(ctx.mean(2, keepdim=True))
        ctx_all = ctx_all + ctx_global*self.gates[:,self.focal_level:]

        # normalize context
        if self.normalize_modulator:
            ctx_all = ctx_all / (self.focal_level+1)

        # focal modulation
        self.modulator = self.h(ctx_all)
        x_out = q*self.modulator
        x_out = x_out.permute(0, 2, 1).contiguous() # (B, C, L) -> (B, L, C)
        
        # post linear porjection
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
        return x_out


class FocalNetBlock(nn.Module):
    r""" Focal Modulation Network Block.
    Args:
        dim (int): Number of input channels.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        focal_level (int): Number of focal levels. 
        focal_window (int): Focal window size at first focal level
    """

    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., 
                    focal_level=1, focal_window=3,
                    normalize_modulator=False):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.focal_window = focal_window
        self.focal_level = focal_level

        self.norm1 = nn.LayerNorm(dim)
        self.modulation = FocalModulation(
            dim, proj_drop=drop, focal_window=focal_window, focal_level=self.focal_level, 
            normalize_modulator=normalize_modulator
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        self.gamma_1 = 1.0
        self.gamma_2 = 1.0    

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, L, C)
        """
        shortcut = x

        # Focal Modulation
        x = self.norm1(x)
        x = self.modulation(x)
        x = self.norm1(x)

        # FFN
        x = shortcut + self.drop_path(self.gamma_1 * x)
        x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))

        return x

class DownSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2)        

    def forward(self, x):
        x = self.pool(x.transpose(1, 2))        
        x = x.transpose(1, 2)  # B Ph*Pw C
        return x

class BasicLayer(nn.Module):
    """ A basic Focal Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at first focal level
    """

    def __init__(self, dim, depth,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 focal_level=1, focal_window=1, 
                 downsample=True, use_checkpoint=False,                  
                 normalize_modulator=False):

        super().__init__()
        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        # build blocks
        self.blocks = nn.ModuleList([
            FocalNetBlock(dim, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          focal_level=focal_level, focal_window=focal_window, normalize_modulator=normalize_modulator)
            for i in range(depth)])

        if downsample:
            self.downsample = DownSample()
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint.checkpoint(blk, x, use_reentrant=True)
            else:
                x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class FocalNet(nn.Module):
    def __init__(
            self,
            in_channels=[1, 16, 16, 32, 32, 96],
            output_len=1024,
            num_classes=5,
            depths=[2, 2, 6, 2],
            focal_levels=[2, 2, 2, 2], 
            focal_windows=[3, 3, 3, 3], 
            mlp_ratio=4.,
            downsample=True,
            drop_rate=0., 
            drop_path_rate=0.1,
            fc_drop=0.2,
            use_checkpoint=False 
    ):
        super().__init__()
        # whether to downsample
        self.downsample = downsample

        # output length
        self.output_len = output_len

        # feature extractor
        self.feature_extraction = FeatureExtraction(in_channels)

        # number of blocks
        self.num_layers = len(depths)

        # embedding dims
        embedding_dim = in_channels[-1]

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build layers
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicLayer(embedding_dim, depths[i], mlp_ratio, drop_rate, 
                                  dpr[sum(depths[:i]):sum(depths[:i + 1])], focal_levels[i], focal_windows[i], downsample, use_checkpoint)
            self.layers.append(layer)
        
        # classifier
        self.norm = nn.LayerNorm(embedding_dim)
        self.avg = nn.AdaptiveAvgPool1d(output_len)
        self.fc1 = nn.Linear(embedding_dim, embedding_dim//2)
        self.fc_drop = nn.Dropout(fc_drop)
        self.fc2 = nn.Linear(embedding_dim//2, num_classes)
    
    def forward(self, x: torch.Tensor):
        # (B, C, L) -> (B, L, C)
        x= self.feature_extraction(x)

        # pass through blcks: (B, L, C) -> (B, L, C)
        for i in range(self.num_layers):
            # pass through blocks
            x = self.layers[i](x)
        
        x = self.avg(x.transpose(-1, 1)) # (B, L, C) -> (B, C, L')
        x = self.fc1(x.transpose(-1, 1)) # (B, C, L') -> (B, L', C')
        x = nn.functional.relu(x)
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x