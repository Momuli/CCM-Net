import cv2
import numpy as np
import math
import torch.nn as nn
import torch
import torch.nn.functional as F

class Mu_Infor_Channel(nn.Module):
    def __init__(self):
        super(Mu_Infor_Channel, self).__init__()
        self.softmax = nn.Softmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(16, 16))

    def forward(self, f_p, f_ms):
        f_p_m = torch.mean(f_p, dim=1, keepdim=True)
        f_p_a = torch.mean(self.avg_pool(f_p), dim=1, keepdim=True)
        f_ms_n = Norm(f_ms)
        f_p_m = Norm(f_p_m)
        f_p_a = Norm(f_p_a)
        h_ms = Entropy(f_ms_n).cuda()
        h_p = Entropy(f_p_m).cuda()
        h_ms_p = Joint_entropy(f_p_a, f_ms_n).cuda()
        mutual_information = torch.sub(torch.add(h_p, h_ms), h_ms_p)
        mutual_information = self.softmax(mutual_information).cuda()
        rel = torch.add(f_ms, torch.mul(f_ms, mutual_information))
        return rel

def Norm(x):
    max_val_t = torch.max(x, 2)[0]
    max_val = torch.max(max_val_t, 2)[0]

    min_val_t = torch.min(x, 2)[0]
    min_val = torch.min(min_val_t, 2)[0]

    delta_t1 = torch.sub(max_val, min_val)
    delta_t2 = torch.unsqueeze(delta_t1, 2)
    delta = torch.unsqueeze(delta_t2, 3)

    min_val_t1 = torch.unsqueeze(min_val, 2)
    min_val = torch.unsqueeze(min_val_t1, 3)

    rel_t1 = torch.sub(x, min_val)
    rel_t2 = torch.div(rel_t1, delta)
    rel = torch.mul(rel_t2, 255).int()
    return rel

def Split(x):
    b, c, w, h = x.size()
    sp_list_b = torch.chunk(x, b, 0)
    sp_list_c = []
    for i in range(len(sp_list_b)):
        temp1 = torch.squeeze(sp_list_b[i], 0)
        sp_list_c.append(torch.chunk(temp1, c, 0))
    sp_list_tc = []
    for i in range(len(sp_list_c)):
        temp_list = []
        for j in range(len(sp_list_c[i])):
            temp2 = torch.squeeze(sp_list_c[i][j], 0)
            temp_list.append(temp2)
        sp_list_tc.append(temp_list)
    return sp_list_tc

def Entropy(x):
    B, C, W, H = x.size()
    size = W * H
    histic = torch.zeros(size=(B, C, 256))
    for i in range(256):
        eq_i = torch.eq(x, i)
        sum_t1 = torch.sum(eq_i, dim=2)
        sum = torch.sum(sum_t1, dim=2)
        histic[:, :, i] = sum
    p_ij = torch.div(histic, size)
    h_ij_t1 = torch.add(p_ij, 1e-8)
    h_ij_t2 = p_ij * torch.log(h_ij_t1)
    h_ij = -torch.sum(h_ij_t2, dim=2)
    return torch.unsqueeze(torch.unsqueeze(h_ij, 2), 3)

def Joint_entropy(x_p, x_ms):
    B, C, H, W = x_ms.size()
    temp = torch.randint(low=4, high=5, size=(B, C, W, H)).cuda()
    histic_ms_p = torch.zeros(size=(B, C, 256, 256))
    for i in range(256):
        for j in range(256):
            eq_i_t1 = torch.eq(x_ms, i).long()
            eq_i = torch.add(eq_i_t1, 1)

            eq_j_t1 = torch.eq(x_p, j).long()
            eq_j = torch.add(eq_j_t1, 1)

            eq_ms = torch.where(eq_i==2, eq_i, temp)
            eq_p = torch.where(eq_j==2, eq_j, temp)
            eq_ij = torch.eq(eq_ms, eq_p)

            sum_t1 = torch.sum(eq_ij, dim=2)
            sum = torch.sum(sum_t1, dim=2)

            histic_ms_p[:, :, i, j] = sum

    p_ms_p = torch.div(histic_ms_p, 256*256)
    h_ms_p_t1 = torch.add(p_ms_p, 1e-8)
    h_ms_p_t2 = p_ms_p * torch.log(h_ms_p_t1)
    h_ms_p_t3 = torch.sum(h_ms_p_t2, dim=2)
    h_ms_p = -torch.sum(h_ms_p_t3, dim=2)
    return torch.unsqueeze(torch.unsqueeze(h_ms_p, 2), 3)

def Upsample(x, y):
    _, _, h1, w1 = x.size()
    result = F.upsample(y, size=(h1, w1), mode='bilinear')
    return result

def Similarity(f_p, f_ms):
    f_ms_t = Upsample(f_p, f_ms)
    f_ms_1 = f_ms_t.view(f_ms_t.size()[0], f_ms_t.size()[1], -1).contiguous()
    f_p_1 = f_p.view(f_p.size()[0], f_p.size()[1], -1).contiguous()
    f_ms_delta_t1 = torch.mean(f_ms_1, dim=2)
    f_ms_delta_t2 = torch.unsqueeze(f_ms_delta_t1, 2)
    f_ms_delta = torch.sub(f_ms_1, f_ms_delta_t2)
    f_p_delta_t1 = torch.mean(f_p_1, dim=2)
    f_p_delta_t2 = torch.unsqueeze(f_p_delta_t1, dim=2)
    f_p_delta = torch.sub(f_p_1, f_p_delta_t2)
    f_p_delta_sum_t1 = torch.pow(f_p_delta, 2)
    f_p_delta_sum_t2 = torch.sum(f_p_delta_sum_t1, dim=2)
    f_p_delta_sum = torch.sqrt(f_p_delta_sum_t2)
    f_ms_delta_sum_t1 = torch.pow(f_ms_delta, 2)
    f_ms_delta_sum_t2 = torch.sum(f_ms_delta_sum_t1, dim=2)
    f_ms_delta_sum = torch.sqrt(f_ms_delta_sum_t2)

    f_p_delta = torch.transpose(f_p_delta, dim0=1, dim1=2)
    f_p_delta = f_p_delta.contiguous()
    s_m_t1 = torch.div(torch.mul(f_ms_delta_sum, f_p_delta_sum), 100)
    s_m_t2 = torch.unsqueeze(s_m_t1, 2)
    s_m_t3 = torch.matmul(f_ms_delta, f_p_delta)

    s_m= torch.div(s_m_t3 , s_m_t2)
    return s_m

def Similarity_Select(s_m):
    max_val, idx = torch.max(s_m, dim=2)
    max_val = nn.Softmax()(max_val)

    list_k = []
    list_total_score = []
    list_total_idx = []
    for i in range(idx.size()[0]):
        idx_u_t1 = torch.unique(idx[i])
        idx_u = torch.sort(idx_u_t1, descending=True)[0]
        co = idx[i].unsqueeze(0) - idx[i].unsqueeze(1)
        uniquer = co.unique(dim=0)
        list_idx = []
        for j in uniquer:
            cover = torch.arange(idx[i].size()[0])
            mask = j == 0
            idx_ = cover[mask]
            list_idx.append(idx_)
        list_sum = []
        for item in list_idx:
            f_t1 = torch.index_select(max_val[i], dim=0, index=item.cuda())
            f = torch.sum(f_t1)
            list_sum.append(f)
        list_sum = torch.stack(list_sum).contiguous()
        list_sum_sorted, list_sum_sorted_idx = torch.sort(list_sum, descending=True)
        list_total_score.append(list_sum_sorted)
        idx_u_select = torch.index_select(idx_u, dim=0, index=list_sum_sorted_idx)
        list_total_idx.append(idx_u_select)
        list_k.append(idx_u_select.size()[0])
    min_k = math.ceil(min(list_k) / 2)
    for i in range(len(list_total_score)):
        list_total_score[i], idx_i = torch.topk(list_total_score[i], k=min_k)
        list_total_idx[i] = torch.index_select(list_total_idx[i], dim=0, index=idx_i)
    list_total_score = torch.stack(list_total_score)
    list_total_idx = torch.stack(list_total_idx)
    return list_total_score, list_total_idx

class Mu_Infor_Spatial(nn.Module):
    def __init__(self):
        super(Mu_Infor_Spatial, self).__init__()
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
    def forward(self, f_p, f_ms):
        s_m = Similarity(f_p, f_ms)
        selected_score, selected_idx = Similarity_Select(s_m)
        selected_total_f_p = []
        for k in range(f_p.size()[0]):
            select_f_p = torch.index_select(f_p[k], dim=0, index=selected_idx[k])
            selected_total_f_p.append(select_f_p)
        selected_total_f_p = torch.stack(selected_total_f_p)
        selected_total_f_p = self.sigmoid(selected_total_f_p)
        list_total_score_t1 = self.softmax(selected_score)
        list_total_score_t2 = torch.unsqueeze(list_total_score_t1, 2)
        list_total_score = torch.unsqueeze(list_total_score_t2, 3)

        mask_t1 = torch.mul(list_total_score, selected_total_f_p)
        mask_t2 = torch.sum(mask_t1, dim=1)
        mask = torch.unsqueeze(mask_t2, 1)

        rel = torch.add(f_p, torch.mul(f_p, mask))
        return rel

class Mutual_Informations(nn.Module):
    def __init__(self):
        super(Mutual_Informations, self).__init__()
        self.Mu_Infor_Channel = Mu_Infor_Channel()
        self.Mu_Infor_Spatial = Mu_Infor_Spatial()

    def forward(self, f_p, f_ms):
        rel_ms = self.Mu_Infor_Channel(f_p, f_ms)
        rel_p = self.Mu_Infor_Spatial(f_p, f_ms)
        return rel_p, rel_ms



