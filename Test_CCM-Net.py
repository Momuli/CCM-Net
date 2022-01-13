from Data_Process import *
import numpy as np
import torch
import torch.nn as nn
from libtiff import TIFF
import cv2
import os
from torch.utils.data import Dataset, DataLoader
from IHS import *
from torch.nn import functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import h5py
from Resnet18 import ResNet18_All
from Resnet50_All import ResNet50_All

BATCH_SIZE = 48

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
Train_Rate = 0.02
ms4_tif = TIFF.open('./Image/ms.tif', mode='r')
ms4_np = ms4_tif.read_image()

pan_tif = TIFF.open('./Image/pan.tif', mode='r')
pan_np = pan_tif.read_image()

label_np = np.load("./data/Image/label.npy")

path = './Image/'
Read_img(os.path.join(path, 'ms.tif'), os.path.join(path, 'pan.tif'))
P_i = cv2.imread(os.path.join(path, 'pan_i.png'))
MS_hs = cv2.imread(os.path.join(path, 'ms_hs.png'))

Ms4_patch_size = 16
Interpolation = cv2.BORDER_REFLECT_101
top_size, bottom_size, left_size, right_size = (int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2),
                                                int(Ms4_patch_size / 2 - 1), int(Ms4_patch_size / 2))
ms4_np = cv2.copyMakeBorder(ms4_np, top_size, bottom_size, left_size, right_size, Interpolation)
MS_hs = cv2.copyMakeBorder(MS_hs, top_size, bottom_size, left_size, right_size, Interpolation)

Pan_patch_size = Ms4_patch_size * 4
top_size, bottom_size, left_size, right_size = (int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2),
                                                int(Pan_patch_size / 2 - 4), int(Pan_patch_size / 2))
pan_np = cv2.copyMakeBorder(pan_np, top_size, bottom_size, left_size, right_size, Interpolation)
P_i = cv2.copyMakeBorder(P_i, top_size, bottom_size, left_size, right_size, Interpolation)

# label_np=label_np.astype(np.uint8)
label_np = label_np - 1

label_element, element_count = np.unique(label_np, return_counts=True)
Categories_Number = len(label_element) - 1
label_row, label_column = np.shape(label_np)

def to_tensor(image):
    max_i = np.max(image)
    min_i = np.min(image)
    image = (image - min_i) / (max_i - min_i)
    return image

ground_xy = np.array([[]] * Categories_Number).tolist()
ground_xy_allData = np.arange(label_row * label_column * 2).reshape(label_row * label_column, 2)

count = 0
for row in range(label_row):
    for column in range(label_column):
        ground_xy_allData[count] = [row, column]
        count = count + 1
        if label_np[row][column] != 255:
            ground_xy[int(label_np[row][column])].append([row, column])

for categories in range(Categories_Number):
    ground_xy[categories] = np.array(ground_xy[categories])
    shuffle_array = np.arange(0, len(ground_xy[categories]), 1)
    np.random.shuffle(shuffle_array)
    ground_xy[categories] = ground_xy[categories][shuffle_array]
shuffle_array = np.arange(0, label_row * label_column, 1)
np.random.shuffle(shuffle_array)
ground_xy_allData = ground_xy_allData[shuffle_array]

ground_xy_train = []
ground_xy_test = []
label_train = []
label_test = []

for categories in range(Categories_Number):
    categories_number = len(ground_xy[categories])
    for i in range(categories_number):
        if i < int(categories_number * Train_Rate):
            ground_xy_train.append(ground_xy[categories][i])
        else:
            ground_xy_test.append(ground_xy[categories][i])
    label_train = label_train + [categories for x in range(int(categories_number * Train_Rate))]
    label_test = label_test + [categories for x in range(categories_number - int(categories_number * Train_Rate))]

label_train = np.array(label_train)
label_test = np.array(label_test)
ground_xy_train = np.array(ground_xy_train)
ground_xy_test = np.array(ground_xy_test)

shuffle_array = np.arange(0, len(label_test), 1)
np.random.shuffle(shuffle_array)
label_test = label_test[shuffle_array]
ground_xy_test = ground_xy_test[shuffle_array]

shuffle_array = np.arange(0, len(label_train), 1)
np.random.shuffle(shuffle_array)
label_train = label_train[shuffle_array]
ground_xy_train = ground_xy_train[shuffle_array]

label_train = torch.from_numpy(label_train).type(torch.LongTensor)
label_test = torch.from_numpy(label_test).type(torch.LongTensor)
ground_xy_train = torch.from_numpy(ground_xy_train).type(torch.LongTensor)
ground_xy_test = torch.from_numpy(ground_xy_test).type(torch.LongTensor)
ground_xy_allData = torch.from_numpy(ground_xy_allData).type(torch.LongTensor)

ms4 = to_tensor(ms4_np)
pan = to_tensor(pan_np)

pan = np.expand_dims(pan, axis=0)
ms4 = np.array(ms4).transpose((2, 0, 1))
MS_hs = np.array(MS_hs).transpose((2, 0, 1))
P_i = np.array(P_i).transpose((2, 0, 1))

ms4 = torch.from_numpy(ms4).type(torch.FloatTensor)
pan = torch.from_numpy(pan).type(torch.FloatTensor)
P_i = torch.from_numpy(P_i).type(torch.FloatTensor)
MS_hs = torch.from_numpy(MS_hs).type(torch.FloatTensor)

train_data = MyData(ms4, pan, label_train, P_i, MS_hs, ground_xy_train, Ms4_patch_size)
test_data = MyData(ms4, pan, label_test, P_i, MS_hs, ground_xy_test, Ms4_patch_size)
all_data = MyData1(ms4, pan, P_i, MS_hs, ground_xy_allData, Ms4_patch_size)

train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
all_data_loader = DataLoader(dataset=all_data, batch_size=BATCH_SIZE,shuffle=False,num_workers=0)

cnn = torch.load('./Model/CCM-Net.pkl')
cnn.cuda()

l = 0
y_pred = []
cnn.eval()
for step, (data, data1, ms_hs, p_i, target, gt_xy) in enumerate(test_loader):
    l = l + 1
    ms = data.cuda()
    pan = data1.cuda()
    ms_hs = ms_hs.cuda()
    p_i = p_i.cuda()
    label = target.cuda()
    with torch.no_grad():
        output = cnn(pan, ms, p_i, ms_hs, 'test')
    pred_y = output[0].max(1, keepdim=True)[1]

    if l == 1:
        y_pred = pred_y.cpu().numpy()
    else:
        y_pred = np.concatenate((y_pred, pred_y.cpu().numpy()), axis=0)
con_mat = confusion_matrix(y_true=label_test, y_pred=y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print("con_mat:")
print(con_mat)

all_acr = 0
p = 0
column = np.sum(con_mat, axis=0)
line = np.sum(con_mat, axis=1)
for i, class_s, in enumerate(con_mat):
    precise = class_s[i]
    all_acr = precise + all_acr
    acr = precise / line[i]
    recall = precise / line[i]
    f1 = 2 * acr * recall / (acr + recall)
    temp = column[i] * line[i]
    p = p + temp

    print("第%d类： || 准确率：%.7f  || 召回率：%.7f  || F1 : %.7f" % (i, acr, recall, f1))

OA = np.trace(con_mat) / np.sum(con_mat)
print("OA", OA)

AA = np.mean(con_mat.diagonal() / np.sum(con_mat, axis=1))

print("AA", AA)

Pc = np.sum(np.sum(con_mat, axis=0) * np.sum(con_mat, axis=1)) / (np.sum(con_mat)) ** 2
kappa = (OA-Pc) / (1-Pc)
print("kappa", kappa)