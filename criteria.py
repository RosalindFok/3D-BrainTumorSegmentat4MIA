import torch
import numpy as np
from torch import nn
from torch.nn.modules.loss import _Loss
from scipy.spatial.distance import directed_hausdorff

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
    def __init__(self, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, seg_y_pred, seg_y_true, rec_y_pred, rec_y_true, y_mid):
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
        y_pred = y_pred.squeeze().detach().cpu().numpy()
        y_true = y_true.squeeze().detach().cpu().numpy()
        hausdorff_distance = []
        for set1, set2 in zip(y_pred, y_true):
            # 计算从set1到set2的Hausdorff距离
            u_hausdorff = directed_hausdorff(set1, set2)[0]
            # 计算从set2到set1的Hausdorff距离
            v_hausdorff = directed_hausdorff(set2, set1)[0]
            # Hausdorff距离是这两个值中的最大值
            hausdorff_distance.append(max(u_hausdorff, v_hausdorff))
        return np.mean(hausdorff_distance)