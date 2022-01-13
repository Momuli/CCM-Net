# CCM-Net
This is a pytorch implementation of CCM-Net

## Requirements
1. pytorch == 1.1.0

2. cuda 8.0

3. python == 3.7

4. [opencv(CV2)](https://pypi.org/project/opencv-python/)

## Data Prepare
1. Please prepare PAN and MS data sets
2. Please prepare the label file
3. The picture format is .tiff or .tif
4. The label format is .npy or .mat
5. Please store the prepared data file in the Image file
```
CCM-Net/Image/
```
## Train
if you want to train your own dataset:
```
1. Put your data set and label file to `CCM-Net/Image/`
2. Modify parameters in `CCM-Net/Train.py` 
4. `python train_CCM-Net.py`
5. save `*.pkl` to weights, like `CCM-Net/Models/*.pkl`
```
## Performance
If you want to evaluate the performance of your model, that is, calculate OA, AA and Kappa:
```
cd CCM-Net/performance.py
modify parameters in `CCM-Net/Performance.py`
python Performance.py
```
