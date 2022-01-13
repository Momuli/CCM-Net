import numpy as np
from osgeo import gdal
from PIL import Image
import cv2
import matplotlib.pyplot as plt


def IHS(data_ms, data_pan):
    A = [[1./3., 1./3., 1./3.], [-np.sqrt(2)/6., -np.sqrt(2)/6., 2*np.sqrt(2)/6], [1./np.sqrt(2), -1./np.sqrt(2), 0.]]
    A = np.matrix(A)

    band, w, h = data_ms.shape
    band_, w_, h_ = data_pan.shape
    pixels_ms = w * h
    pixels_pan = w_ * h_
    data_ms = data_ms.reshape(3, pixels_ms)
    data_pan = data_pan.reshape(3, pixels_pan)

    pan_i = np.dot(A, np.matrix(data_pan))
    pan_i = np.array(pan_i)
    pan_i = pan_i.reshape(band_, w_, h_)
    pan_i = pan_i[0, :, :].reshape(1, w_, h_)

    ms_ihs = np.dot(A, np.matrix(data_ms))
    ms_ihs = np.array(ms_ihs)
    ms_ihs = ms_ihs.reshape(band_, w, h)
    ms_ihs = ms_ihs[1:, :, :]

    pan_min = np.min(pan_i.ravel())
    pan_max = np.max(pan_i.ravel())
    pan_visual = np.uint8((pan_i.astype(np.float) - pan_min) / (pan_max - pan_min) * 255)
    pan_visual = Image.fromarray(cv2.merge([pan_visual[0]]))

    ms_min = np.min(ms_ihs.ravel())
    ms_max = np.max(ms_ihs.ravel())
    ms_visual = np.uint8((ms_ihs.astype(np.float) - ms_min) / (ms_max - ms_min) * 255)
    ms_visual = Image.fromarray(cv2.merge([ms_visual[0], ms_visual[1]]))
    return pan_visual, ms_visual

def imresize(data_ms, data_pan):
    band, col, row = data_ms.shape
    data = np.zeros(((band, col, row)))
    for i in range(0, band):
            data[i] = np.array(Image.fromarray(data_ms[i]).resize((row, col)))

def gdal_open_MS(path):
    data = gdal.Open(path)
    col = data.RasterXSize
    row = data.RasterYSize
    data_array_r = data.GetRasterBand(1).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array_g = data.GetRasterBand(2).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array_b = data.GetRasterBand(3).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array = np.array((data_array_r, data_array_g, data_array_b))
    return data_array

def gdal_open_Pan(path):
    data = gdal.Open(path)
    col = data.RasterXSize
    row = data.RasterYSize
    data_array_r = data.GetRasterBand(1).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array_g = data.GetRasterBand(1).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array_b = data.GetRasterBand(1).ReadAsArray(0, 0, col, row).astype(np.float)
    data_array = np.array((data_array_r, data_array_g, data_array_b))
    return data_array

def Read_img(path_ms, path_pan):
    data_ms = gdal_open_MS(path_ms)
    data_pan = gdal_open_Pan(path_pan)
    pan_i, ms_hs = IHS(data_ms, data_pan)
    pan_i.save('./Image/pan_i.png', 'png')
    ms_hs.save('./Image/ms_hs.png', 'png')
    return
