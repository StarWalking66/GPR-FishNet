import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline.models.task_adapter import format_prediction_grid, validate_sequence_grid_input


class STLSTMCellV2(nn.Module):
    def __init__(self, in_channel, num_hidden, height, width, filter_size, stride):
        super(STLSTMCellV2, self).__init__()
        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        
        
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 7, height, width])
        )
        
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 4, height, width])
        )
        
        self.conv_m = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden * 3, height, width])
        )
        
        self.conv_o = nn.Sequential(
            nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            nn.LayerNorm([num_hidden, height, width])
        )
       
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0)

    def forward(self, x, h, c, m):
        x_concat = self.conv_x(x)
        h_concat = self.conv_h(h)
        m_concat = self.conv_m(m)

        
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        
        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)
        c_next = f_t * c + i_t * g_t
        
        
        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)
        m_next = f_t_prime * m + i_t_prime * g_t_prime
        
       
        mem = torch.cat([c_next, m_next], dim=1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_next = o_t * torch.tanh(self.conv_last(mem))
        
        delta_c = c_next - c
        delta_m = m_next - m
        return h_next, c_next, m_next, delta_c, delta_m



class PredRNNV2(nn.Module):
    def __init__(self, in_chans=8, hidden_dim=64, img_size=(64, 96), num_layers=2, pred_len=1):
        super(PredRNNV2, self).__init__()
        self.in_chans = in_chans
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.pred_len = pred_len
        self.h, self.w = img_size
        
        cell_list = []
        for i in range(num_layers):
            cur_in_chans = in_chans if i == 0 else hidden_dim
            
            cell_list.append(STLSTMCellV2(cur_in_chans, hidden_dim, self.h, self.w, 3, 1))
        self.cell_list = nn.ModuleList(cell_list)
        self.delta_adapter = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1, bias=False)
        self.predictor = nn.Conv2d(hidden_dim, pred_len, kernel_size=1)

    def forward(self, x, return_aux: bool = False):
        batch_size, seq_len, _, _, _ = validate_sequence_grid_input(
            x,
            model_name=self.__class__.__name__,
            in_chans=self.in_chans,
            img_size=(self.h, self.w),
        )
        
        h_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        c_t = [torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        memory = torch.zeros(batch_size, self.hidden_dim, self.h, self.w, device=x.device, dtype=x.dtype)
        decouple_terms = []

        for t in range(seq_len):
            input_t = x[:, t] 
            for i in range(self.num_layers):
                layer_in = input_t if i == 0 else h_t[i-1]
                h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](layer_in, h_t[i], c_t[i], memory)
                delta_c = F.normalize(self.delta_adapter(delta_c).flatten(2), dim=2)
                delta_m = F.normalize(self.delta_adapter(delta_m).flatten(2), dim=2)
                decouple_terms.append(torch.abs(F.cosine_similarity(delta_c, delta_m, dim=2)).mean())
        
       
        out = self.predictor(h_t[-1])
        pred = format_prediction_grid(out, model_name=self.__class__.__name__)
        if not return_aux:
            return pred

        decouple_loss = pred.new_zeros(())
        if decouple_terms:
            decouple_loss = torch.stack(decouple_terms).mean()
        return pred, {"decouple_loss": decouple_loss}
