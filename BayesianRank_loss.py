import torch
import torch.nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class BRankLoss(nn.Module):
    def __init__(self):
        super(BRankLoss, self).__init__()
    def forward(self, res: torch.Tensor, label: torch.Tensor):
        loss = torch.neg(torch.log(res).mul(label).sum())
        return loss

            
        
