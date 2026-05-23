import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.task_adapter import format_prediction_grid, validate_sequence_grid_input


class LocalSpatiotemporalAttention(nn.Module):
    def __init__(self, channels, window_size=3):
        super().__init__()
        self.query_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(channels, channels, kernel_size=window_size, padding=window_size//2)
        
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)  # [B, H*W, C']
        proj_key = self.key_conv(x).view(B, -1, W * H)                       # [B, C', H*W]
        energy = torch.bmm(proj_query, proj_key)                             # [B, H*W, H*W]
        attention = F.softmax(energy, dim=-1)                                
        proj_value = self.value_conv(x).view(B, -1, W * H)                   # [B, C, H*W]

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class ConvGRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.reset_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.update_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)
        self.out_gate = nn.Conv2d(input_size + hidden_size, hidden_size, kernel_size, padding=padding)

    def forward(self, input_tensor, prev_state):
        stacked_inputs = torch.cat([input_tensor, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_tensor, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update
        return new_state


class ExPreCast(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, img_size=(64, 96), num_layers=2, pred_len=1):
        super(ExPreCast, self).__init__()
        self.in_chans = in_chans
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pred_len = pred_len
        self.h, self.w = img_size

        
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_chans, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

        
        self.rnn_cells = nn.ModuleList([
            ConvGRUCell(hidden_dim, hidden_dim, kernel_size=3) for _ in range(num_layers)
        ])

        
        self.local_attn = LocalSpatiotemporalAttention(hidden_dim, window_size=5)

       
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(4, hidden_dim // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, pred_len, kernel_size=1),
        )

    def forward(self, x):
        # x shape: [B, S, C, H, W]
        B, S, _, H, W = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            in_chans=self.in_chans,
            img_size=(self.h, self.w),
        )
        
        h_states = [torch.zeros(B, self.hidden_dim, H, W, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        
        
        for t in range(S):
            xt = self.input_layer(x[:, t])
            for i, cell in enumerate(self.rnn_cells):
                h_states[i] = cell(xt if i == 0 else h_states[i-1], h_states[i])
                
        h_out = h_states[-1]
        
       
        attn_out = self.local_attn(h_out)
        
        
        pred = self.decoder(attn_out)
        
        
        return format_prediction_grid(pred, model_name=self.__class__.__name__)
