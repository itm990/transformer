import torch
import torch.nn as nn


class MyNLLLoss(nn.Module):
    
    def __init__(self, smooth_weight=0.1, ignore_index=-100):
        super(MyNLLLoss, self).__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        ignore_mask = target.ne(self.ignore_index)
        hot_input = input.gather(dim=1, index=target.unsqueeze(-1))
        hot_input = hot_input.squeeze(-1).masked_select(ignore_mask)
        all_input = input.sum(-1).masked_select(ignore_mask)
        hot_loss = -(1 - self.smooth_weight) * hot_input
        all_loss = -(self.smooth_weight/input.size(-1)) * all_input
        loss = hot_loss + all_loss
        loss = loss.mean()

        return loss

