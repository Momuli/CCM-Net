from Data_Process import *
from Resnet18 import ResNet18_MI_Diff
import os
from torch.nn import functional as F
import torch.optim as optim
from libtiff import TIFF
import numpy as np
import torch
import cv2
from IHS import *

EPOCH = 30
BATCH_SIZE = 48
LR = 0.001
Train_Rate = 0.02
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

model = ResNet18_MI_Diff().cuda()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train_model(model, train_loader, optimizer, epoch):
    model.train()
    correct = 0.0
    for step, (ms, pan, ms_hs, p_i, label, _) in enumerate(train_loader):
        ms, pan, ms_hs, p_i, label = ms.cuda(), pan.cuda(), ms_hs.cuda(), p_i.cuda(), label.cuda()

        optimizer.zero_grad()
        output = model(pan, ms, p_i, ms_hs, 'train')
        pred_train = output[1].max(1, keepdim=True)[1]
        correct += pred_train.eq(label.view_as(pred_train).long()).sum().item()
        loss = F.cross_entropy(output[1], label.long()) + 0.01 * output[0]
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print("Train Epoch: {} \t Loss : {:.6f} \t step: {} ".format(epoch, loss.item(), step))
    print("Train Accuracy: {:.6f}".format(correct * 100.0 / len(train_loader.dataset)))

def test_model(model, test_loader):
    model.eval()
    correct = 0.0
    test_loss = 0.0
    with torch.no_grad():
        for data, data1, ms_hs, p_i, target, _ in test_loader:
            data, data1, ms_hs, p_i, target = data.cuda(), data1.cuda(), ms_hs.cuda(), p_i.cuda(), target.cuda()
            output = model(data1, data, p_i, ms_hs, 'test')
            test_loss += F.cross_entropy(output[0], target.long()).item()
            pred = output[0].max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred).long()).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        print("test-average loss: {:.4f}, Accuracy:{:.3f} \n".format(
            test_loss, 100.0 * correct / len(test_loader.dataset)
        ))

for epoch in range(1, EPOCH+1):
    train_model(model,  train_loader, optimizer, epoch)
    test_model(model,  test_loader)
torch.save(model, './Model/CCM-Net.pkl')