from backbone import ResBlk
from Mutual_Information import Mutual_Informations
from Reduce_modal_Diff import Re_Mo_Diff_Loss
import torch.nn as nn
import torch
from torch.nn import functional as F

class ResNet18_MI_Diff(nn.Module):
    def __init__(self):
        super(ResNet18_MI_Diff, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.blk1_1 = ResBlk(64, 64, stride=1)
        self.blk2_1 = ResBlk(64, 128, stride=1)
        self.blk3_1 = ResBlk(128, 256, stride=1)
        self.blk4_1 = ResBlk(256, 512, stride=1)

        self.blk1_2 = ResBlk(64, 64, stride=1)
        self.blk2_2 = ResBlk(64, 128, stride=1)
        self.blk3_2 = ResBlk(128, 256, stride=1)
        self.blk4_2 = ResBlk(256, 512, stride=1)

        self.Mutual_information = Mutual_Informations()
        self.Modal_Diff = Re_Mo_Diff_Loss(in_c=512)
        self.outlayer = nn.Linear(1024, 11)

    def forward(self, f_p, f_ms, f_i, f_hs, phase='train'):
        f_p_1 = F.relu(self.conv1(f_p))
        f_ms_1 = F.relu(self.conv2(f_ms))
        f_p_1, f_ms_1 = self.Mutual_information[0](f_p_1, f_ms_1)

        f_p_2 = self.blk1_1(f_p_1)
        f_ms_2 = self.blk1_2(f_ms_1)
        f_p_2, f_ms_2 = self.Mutual_information[1](f_p_2, f_ms_2)

        f_p_3 = self.blk2_1(f_p_2)
        f_ms_3 = self.blk2_2(f_ms_2)
        f_p_3, f_ms_3 = self.Mutual_information[2](f_p_3, f_ms_3)

        f_p_4 = self.blk3_1(f_p_3)
        f_ms_4 = self.blk3_2(f_ms_3)
        f_p_4, f_ms_4 = self.Mutual_information[3](f_p_4, f_ms_4)

        f_p_5 = self.blk4_1(f_p_4)
        f_ms_5 = self.blk4_2(f_ms_4)
        f_p_5, f_ms_5 = self.Mutual_information[4](f_p_5, f_ms_5)

        out = []
        if phase == 'train':
            # Loss
            loss_diff = self.Modal_Diff(f_p_5, f_ms_5, f_i, f_hs)
            out.append(loss_diff)

        f_p_fusion = torch.cat([f_p_1, f_p_2, f_p_3, f_p_4, f_p_5], 1)
        f_ms_fusion = torch.cat([f_ms_1, f_ms_2, f_ms_3, f_ms_4, f_ms_5], 1)
        f_ms_fusion = F.adaptive_avg_pool2d(f_ms_fusion, [1, 1])
        f_p_fusion = F.adaptive_avg_pool2d(f_p_fusion, [1, 1])
        s = f_ms_fusion + f_p_fusion
        s = s.view(s.size()[0], -1)
        rel = self.outlayer(s)
        out.append(rel)
        return out

