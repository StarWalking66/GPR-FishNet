import torch
import torch.nn as nn

from baseline.models.task_adapter import format_prediction_grid, validate_sequence_grid_input


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PFGBlock(nn.Module):
    """
    ?1D ?
    ?
    """
    def __init__(self, dim, kernel_size=11):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, dim * 2, 1)
        
        
        padding = kernel_size // 2
        self.dwconv_h = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size), padding=(0, padding), groups=dim)
        self.dwconv_w = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1), padding=(padding, 0), groups=dim)
        
        
        self.center_conv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        
        
        self.gate_conv = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)
        
        self.norm1 = nn.GroupNorm(4, dim)
        self.norm2 = nn.GroupNorm(4, dim)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        
         
        x_proj = self.proj_in(x)
        feat, gate = x_proj.chunk(2, dim=1)
        
         
        peri_h = self.dwconv_h(feat)
        peri_w = self.dwconv_w(feat)
        peripheral = peri_h + peri_w
        
         
        center = self.center_conv(feat)
        
    
        gate_weight = torch.sigmoid(self.gate_conv(gate))
        
        
        fused = peripheral - center * gate_weight
        
        x = self.proj_out(fused * gate_weight)
        x = x + shortcut
        
         
        x = x + self.mlp(self.norm2(x))
        return x


class PFGNet(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, seq_len=12, img_size=(64, 96), num_layers=4, pred_len=1):
        super().__init__()
        self.seq_len = seq_len
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.img_size = img_size
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans * seq_len, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        )
        
        
        self.translator = nn.Sequential(
            *[PFGBlock(hidden_dim, kernel_size=11) for _ in range(num_layers)]
        )
        
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, pred_len, kernel_size=1),
        )

    def forward(self, x):
        
        B, S, C, H, W = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            seq_len=self.seq_len,
            in_chans=self.in_chans,
            img_size=self.img_size,
        )
        
        
        x = x.reshape(B, S * C, H, W)
        
         
        z = self.encoder(x)
        
        
        z = self.translator(z)
        
        
        pred = self.decoder(z)
        
        
        return format_prediction_grid(pred, model_name=self.__class__.__name__)
