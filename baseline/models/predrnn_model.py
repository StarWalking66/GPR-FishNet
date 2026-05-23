import torch
import torch.nn as nn

from baseline.models.task_adapter import format_prediction_grid, validate_sequence_grid_input


class STLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(STLSTMCell, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        self.conv_m = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden, height, width])
        )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x, h, c, m):
        combined_h = torch.cat([x, h], dim=1)
        gates_h = self.conv_h(combined_h)
        i, f, g, i_prime = torch.split(gates_h, self.num_hidden, dim=1)
        
        i = torch.sigmoid(i)
        f = torch.sigmoid(f + self._forget_bias)
        g = torch.tanh(g)
        c_next = f * c + i * g
        
        combined_m = torch.cat([x, m], dim=1)
        gates_m = self.conv_m(combined_m)
        f_prime, g_prime, m_prime = torch.split(gates_m, self.num_hidden, dim=1)
        
        f_prime = torch.sigmoid(f_prime + self._forget_bias)
        g_prime = torch.tanh(g_prime)
        m_next = f_prime * m + torch.sigmoid(m_prime) * g_prime
        
        combined_o = torch.cat([x, c_next, m_next], dim=1)
        o = torch.sigmoid(self.conv_o(combined_o))
        h_next = o * torch.tanh(self.conv_last(torch.cat([c_next, m_next], dim=1)))
        
        return h_next, c_next, m_next


class PredRNN(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, img_size=(64, 96), num_layers=2, pred_len=1):
        super(PredRNN, self).__init__()
        self.in_chans = in_chans
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.h, self.w = img_size
        
        cell_list = []
        for i in range(num_layers):
            cur_in_chans = in_chans if i == 0 else hidden_dim
            cell_list.append(STLSTMCell(cur_in_chans, hidden_dim, self.h, self.w, 3, 1))
        self.cell_list = nn.ModuleList(cell_list)
        self.predictor = nn.Conv2d(hidden_dim, pred_len, kernel_size=1)

    def forward(self, x):
        batch_size, seq_len, _, _, _ = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            in_chans=self.in_chans,
            img_size=(self.h, self.w),
        )
        
        h_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        memory = torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype)
        
        for t in range(seq_len):
            input_t = x[:, t] 
            for i in range(self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](input_t if i == 0 else h_t[i-1], h_t[i], c_t[i], memory)
        
        out = self.predictor(h_t[-1])
        return format_prediction_grid(out, model_name=self.__class__.__name__)
