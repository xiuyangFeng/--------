import torch.nn as nn
import torch
import torch.nn.functional as F


class get_model(nn.Module):
    def __init__(self):
        super(get_model, self).__init__()

        # Initial MLP1: [3 → 256 → 512]
        self.mlp_convs1 = nn.ModuleList()
        self.mlp_bns1   = nn.ModuleList()
        input_channel = 3
        for output_channel in [256, 512]:
            self.mlp_convs1.append(nn.Conv1d(input_channel, output_channel, 1))
            self.mlp_bns1.append(nn.BatchNorm1d(output_channel))
            input_channel = output_channel

        # Final MLP2: [1024 → 512 → 256 → 4]
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns   = nn.ModuleList()
        last_channel = 1024
        for out_channel in [512, 256, 4]:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            if out_channel != 4:
                self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

        self.dropout = nn.Dropout(0.5)


    def forward(self, xyz):
        B, _, N = xyz.shape
        x = xyz[:, 0:3, :]   # only xyz

        # MLP 1
        for i, conv in enumerate(self.mlp_convs1):
            bn = self.mlp_bns1[i]
            x = F.relu(bn(conv(x)))

        # Global feature
        y = x                           # (B, 512, N)
        x_global = torch.max(x, 2, keepdim=True)[0]  # (B, 512, 1)
        x_global = x_global.expand(-1, -1, N)        # (B, 512, N)
        x = torch.cat([x_global, x], dim=1)          # (B, 1024, N)

        # MLP 2
        for i, conv in enumerate(self.mlp_convs):
            if i < 2:
                bn = self.mlp_bns[i]
                x = F.relu(bn(conv(x)))
            else:
                x = conv(x)

        return x   # (B, 4, N)

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        e = pred - target
        mse = torch.mean(e**2)
        mae= torch.mean(torch.abs(e))
        return mse,mae
