import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.task_adapter import format_prediction_grid, validate_sequence_grid_input


class TemporalKANLayer(nn.Module):
    """
    ?KAN ?
     SiLU  (Base function) Chebyshev ?(Polynomials) 
    ?
    """
    def __init__(self, seq_len, out_len=1, degree=3):
        super().__init__()
        self.seq_len = seq_len
        self.out_len = out_len
        self.degree = degree
        
        
        self.base_weight = nn.Parameter(torch.Tensor(out_len, seq_len))
        
        self.poly_weight = nn.Parameter(torch.Tensor(out_len, seq_len, degree + 1))
        
        
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.poly_weight, a=5**0.5)

    def forward(self, x):
       
        
        
        base_out = F.silu(x)
        
        res_base = torch.matmul(base_out, self.base_weight.t())

       
        x_norm = torch.tanh(x)
        
        
        polys = [torch.ones_like(x_norm), x_norm]
        for i in range(2, self.degree + 1):
            polys.append(2 * x_norm * polys[-1] - polys[-2])
            
        
        polys = torch.stack(polys, dim=-1)
        
      
        res_poly = torch.einsum('bchwsd,sdo->bchwo', polys, self.poly_weight.permute(1, 2, 0))

       
        return res_base + res_poly


class TimeKAN(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, seq_len=12, img_size=(64, 96), pred_len=1):
        super().__init__()
        self.in_chans = in_chans
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.img_size = img_size
        
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
        )
        
        
        self.time_kan = TemporalKANLayer(seq_len=seq_len, out_len=pred_len, degree=3)
        
        
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1),
        )

    def forward(self, x):
        
        B, S, C, H, W = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            seq_len=self.seq_len,
            in_chans=self.in_chans,
            img_size=self.img_size,
        )
        
       
        x_enc = x.view(B * S, C, H, W)
        h_enc = self.encoder(x_enc)  
        
        
        h_enc = h_enc.view(B, S, self.hidden_dim, H, W)
        
        
        h_time = h_enc.permute(0, 2, 3, 4, 1)
        
        
        h_mixed = self.time_kan(h_time)
        
        
        h_mixed = h_mixed.permute(0, 4, 1, 2, 3).reshape(B * self.pred_len, self.hidden_dim, H, W)
        
        
        pred = self.decoder(h_mixed).reshape(B, self.pred_len, H, W)
        
        
        return format_prediction_grid(pred, model_name=self.__class__.__name__)
