from torch.utils.data import Dataset, DataLoader
class MyData(Dataset):
    def __init__(self, MS4, Pan, Label, P_i, MS_hs, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.train_labels = Label
        self.P_i = P_i
        self.MS_hs = MS_hs

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        # 同样用index裁剪I与HS
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)      # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        image_ms_hs = self.MS_hs[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan_i = self.P_i[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        target = self.train_labels[index]
        return image_ms, image_pan, image_ms_hs, image_pan_i, target, locate_xy

    def __len__(self):
        return len(self.gt_xy)


class MyData1(Dataset):
    def __init__(self, MS4, Pan, P_i, MS_hs, xy, cut_size):
        self.train_data1 = MS4
        self.train_data2 = Pan
        self.P_i = P_i
        self.MS_hs = MS_hs

        self.gt_xy = xy
        self.cut_ms_size = cut_size
        self.cut_pan_size = cut_size * 4

    def __getitem__(self, index):
        x_ms, y_ms = self.gt_xy[index]
        x_pan = int(4 * x_ms)  # 计算不可以在切片过程中进行
        y_pan = int(4 * y_ms)
        image_ms = self.train_data1[:, x_ms:x_ms + self.cut_ms_size,
                   y_ms:y_ms + self.cut_ms_size]

        image_pan = self.train_data2[:, x_pan:x_pan + self.cut_pan_size,
                    y_pan:y_pan + self.cut_pan_size]

        locate_xy = self.gt_xy[index]

        return image_ms, image_pan, self.MS_hs, self.P_i, locate_xy

    def __len__(self):
        return len(self.gt_xy)
