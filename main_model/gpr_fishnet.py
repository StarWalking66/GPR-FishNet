import torch
import torch.nn as nn
import torch.nn.functional as F


class STLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super().__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0

        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, height, width]),
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, height, width]),
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width]),
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x, h, c, m):
        combined_h = torch.cat([x, h], dim=1)
        gates_h = self.conv_h(combined_h)
        i, f, g, _ = torch.split(gates_h, self.num_hidden, dim=1)

        c_next = torch.sigmoid(f + self._forget_bias) * c + torch.sigmoid(i) * torch.tanh(g)

        combined_m = torch.cat([x, m], dim=1)
        gates_m = self.conv_m(combined_m)
        f_prime, g_prime, m_prime = torch.split(gates_m, self.num_hidden, dim=1)

        m_next = torch.sigmoid(f_prime + self._forget_bias) * m + torch.sigmoid(m_prime) * torch.tanh(g_prime)

        combined_o = torch.cat([x, c_next, m_next], dim=1)
        o = torch.sigmoid(self.conv_o(combined_o))
        h_next = o * torch.tanh(self.conv_last(torch.cat([c_next, m_next], dim=1)))

        return h_next, c_next, m_next

# --- 空间与通道注意力模块 ---
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=8, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.conv_sa = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        ca_out = x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))
        sa_avg, sa_max = torch.mean(ca_out, dim=1, keepdim=True), torch.max(ca_out, dim=1, keepdim=True)[0]
        sa_out = self.conv_sa(torch.cat([sa_avg, sa_max], dim=1))
        return x + ca_out * self.sigmoid(sa_out)


class AttentionResidualPredictor(nn.Module):
    def __init__(self, fusion_dim, hidden_dim):
        super().__init__()
        self.cbam = CBAM(fusion_dim)
        self.conv_net = nn.Sequential(
            nn.Conv2d(fusion_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, kernel_size=1)
        )
        
        self.shortcut = nn.Conv2d(fusion_dim, 1, kernel_size=1)

    def forward(self, x):
        h_attn = self.cbam(x)
        return h_attn, self.conv_net(h_attn) + self.shortcut(h_attn)


class MultiScalePersistence(nn.Module):
    def __init__(self, in_channels, hidden_dim):
        super().__init__()
        mid_dim = hidden_dim // 4 
        
        self.conv_local = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=3, padding=1),
            nn.GroupNorm(4, mid_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.conv_surround = nn.Sequential(
            nn.Conv2d(in_channels, mid_dim, kernel_size=3, padding=2, dilation=2),
            nn.GroupNorm(4, mid_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )
        
        self.fuse = nn.Conv2d(mid_dim * 2, 1, kernel_size=1)

    def forward(self, h_attn, last_obs):
        ctx = torch.cat([last_obs, h_attn], dim=1)
        local_feat = self.conv_local(ctx)
        surround_feat = self.conv_surround(ctx)
        persistence_delta = self.fuse(torch.cat([local_feat, surround_feat], dim=1))
        return last_obs + persistence_delta


class ContextAwareRouter(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
            nn.Sigmoid()  
        )

    def forward(self, route_ctx, pred_gen, pred_pers):
       
        weights = self.gate(route_ctx)
        w_gen = weights[:, 0:1, :, :]
        w_pers = weights[:, 1:2, :, :]
        return w_gen * pred_gen + w_pers * pred_pers

# ====================================================================
# Full main network (GPR-FishNet)
# ====================================================================
class GPRFishNet(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, img_size=(64, 96), num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.h, self.w = img_size

        self.cell_list = nn.ModuleList([
            STLSTMCell(in_chans if i == 0 else hidden_dim, hidden_dim, self.h, self.w, 3, 1) 
            for i in range(num_layers)
        ])
        
        fusion_dim = hidden_dim * 2 if num_layers > 1 else hidden_dim
        
       
        self.predictor = AttentionResidualPredictor(fusion_dim, hidden_dim)          
        self.persister = MultiScalePersistence(1 + fusion_dim, hidden_dim)           
       
        self.router = ContextAwareRouter(fusion_dim + 1)                 

    def forward(self, x):
        batch_size, seq_len = x.shape[0], x.shape[1]
        device = x.device

        h_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device) for _ in range(self.num_layers)]
        memory = torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=device)

        for t in range(seq_len):
            for i in range(self.num_layers):
                layer_input = x[:, t] if i == 0 else h_t[i - 1]
                h_t[i], c_t[i], memory = self.cell_list[i](layer_input, h_t[i], c_t[i], memory)

        fusion_h = torch.cat([h_t[0], h_t[-1]], dim=1) if self.num_layers > 1 else h_t[-1]
        last_obs = x[:, -1, -1:] 
        
        # ---------------------------------------------------------
        # 分支 1: 生成全新渔场特征 (ARP)
        h_attn, pred_gen = self.predictor(fusion_h)
        
        # ---------------------------------------------------------
        # 分支 2: 捕捉多尺度空间驻留与扩散 (MSSP)
        pred_pers = self.persister(h_attn, last_obs)
        
        # ---------------------------------------------------------
        # 分支 3: 上下文感知路由 (CAR)
        
        route_ctx = torch.cat([h_attn, last_obs], dim=1)
        pred_final = self.router(route_ctx, pred_gen, pred_pers)
        
       
        pred_final = F.relu(pred_final)

        return pred_final.unsqueeze(1)
