import torch
import torch.nn as nn
from torch.nn import functional as F

class Re_Mo_Diff_Loss(nn.Module):
    def __init__(self, in_c=1024):
        super(Re_Mo_Diff_Loss, self).__init__()
        self.pre_conv_p = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.pre_conv_m = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, stride=1)
        self.change_c_m = nn.Conv2d(in_channels=in_c, out_channels=512, kernel_size=1)
        self.change_c_p = nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.Conv2d(in_channels=in_c, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1),
        )
        self.change_c_i = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=2, padding=1),
        )
        self.change_c_hs = nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1)
        self.change_to_one_1 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
        self.change_to_one_2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
        )
    def forward(self, f_p, f_ms, f_i, f_hs):
        f_p_t = self.change_c_p(f_p)
        pre_p = self.pre_conv_p(f_p_t)
        f_hs_t = self.change_c_hs(f_hs)

        pre_p = self.change_to_one_1(pre_p)
        f_hs_t = self.change_to_one_1(f_hs_t)
        f_p_t = self.change_to_one_1(f_p_t)
        pre_p_1 = pre_p
        pre_p = torch.matmul(torch.matmul(pre_p_1, f_p_t), pre_p)
        l1 = F.smooth_l1_loss(pre_p, f_hs_t, reduction='mean')

        f_ms_t = self.change_c_m(f_ms)
        pre_ms = self.pre_conv_m(f_ms_t)
        f_i_t = self.change_c_i(f_i)

        pre_ms = self.change_to_one_2(pre_ms)
        f_i_t = self.change_to_one_2(f_i_t)
        f_ms_t = self.change_to_one_2(f_ms_t)

        pre_ms_1 = pre_ms
        pre_ms = torch.matmul(torch.matmul(pre_ms_1, f_ms_t), pre_ms)
        l2 = F.smooth_l1_loss(pre_ms, f_i_t, reduction='mean')

        diff_loss = 0.5 * l1 + 0.5 * l2
        return diff_loss
