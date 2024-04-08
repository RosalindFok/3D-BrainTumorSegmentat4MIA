import torch
from torch import nn
from torch.nn.modules.loss import _Loss


class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division, 
    '''
    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true)) 
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union 
        dice_loss = 1 - dice

        return max(dice_loss, 0)


class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''
    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(torch.log(torch.mul(std, std))) - 1


class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1, type='WT'
    WT: whole tumor(1,2,4)
    TC: tumor core(2,4)
    ET: enhancing tumor(4)
    '''
    def __init__(self, k1=0.1, k2=0.1, type : str = None):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()
        self.type = type if not type is None else 'WT'

    def forward(self, seg_y_pred, seg_y_true, rec_y_pred, rec_y_true, y_mid):
        if self.type == 'WT':
            threshold = 1
        elif self.type == 'TC':
            threshold = 2
        elif self.type == 'ET':
            threshold = 4
        else:
            raise ValueError('Invalid type')
            exit(1)
        seg_y_true = torch.where(seg_y_true >= threshold, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))
        rec_y_true = torch.where(rec_y_true >= threshold, torch.tensor(1.0, dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32))

        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        dice_loss = self.dice_loss(seg_y_pred, seg_y_true)
        l2_loss = self.l2_loss(rec_y_pred, rec_y_true)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        return combined_loss

class Hausdorff_Distance(_Loss):
    '''
    Hausdorff_Distance = max(h(A,B), h(B,A)); h(A,B) = max_(a in A){min_(b in B){||a-b||}}; ||a-b|| is the Euclidean distance
    '''
    def __init__(self, *args, **kwargs):
        super(Hausdorff_Distance, self).__init__()

    def forward(self, y_pred, y_true):
        dist1 = torch.cdist(y_pred.view(-1, y_pred.size(-1)), y_true.view(-1, y_true.size(-1))).max(dim=1)[0].max()
        dist2 = torch.cdist(y_true.view(-1, y_true.size(-1)), y_pred.view(-1, y_pred.size(-1))).max(dim=1)[0].max()
        hausdorff_distance = torch.max(dist1, dist2)
        return hausdorff_distance