import os
import time

import cv2
import h5py
import numpy as np
import scipy.io
import scipy.spatial
from scipy.ndimage import gaussian_filter
import math
import torch
import random
import xml.etree.ElementTree as ET

import glob
'''change your dataset'''
# root = '/home/dkliang/projects/synchronous/dataset/UCF-QNRF_ECCV18'
root = '/root/TransCrowdTest/dataset'
img_train_path = root + '/train_data/images/'
img_test_path = root + '/test_data/images/'


save_train_img_path = root + '/my_train_data/images/'
save_test_img_path = root + '/my_test_data/images/'

distance = 1
img_train = []
img_test = []


# path_sets = [img_train_path, img_test_path]
path_sets = [img_train_path]

print(len(path_sets))
print(path_sets)

img_paths = []
for path in path_sets:
    print(path)
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
        # print(img_path)
        img_paths.append(img_path)

img_paths.sort()

np.random.seed(0)
random.seed(0)
for img_path in img_paths:


    Img_data = cv2.imread(img_path)
    print(img_path)
    
    xml_file = img_path.replace('.jpg', 'R.xml').replace('images', 'labels')
    tree = ET.parse(xml_file)
    root = tree.getroot()
    width = int(root.find('size/width').text)
    height = int(root.find('size/height').text)
    Gt_data = []

    for obj in root.findall('object'):
        x_elem = obj.find('point/x')
        y_elem = obj.find('point/y')

        # 如果 <x> 和 <y> 元素都存在，获取其文本内容
        if x_elem is not None and x_elem.text is not None and y_elem is not None and y_elem.text is not None:
            x = int(x_elem.text)
            y = int(y_elem.text)
            # print(f"Found point: (x={x}, y={y})")
        else:
            x_elem = obj.find('bndbox/xmin')
            y_elem = obj.find('bndbox/ymin')
            x = int(x_elem.text)
            y = int(y_elem.text)
            # print("No valid point found for this object")
        
        # 将坐标点添加到 Gt_data 中，可以选择如何存储，例如列表、数组等
        Gt_data.append([x, y])

    # 将 Gt_data 转换为 numpy 数组（可选）
    Gt_data = np.array(Gt_data)
    
    print(len(Gt_data))


    if Img_data.shape[1] >= Img_data.shape[0]:
        rate_1 = 1152.0 / Img_data.shape[1]
        rate_2 = 768 / Img_data.shape[0]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_1, fy=rate_2)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_1
        Gt_data[:, 1] = Gt_data[:, 1] * rate_2

    elif Img_data.shape[0] > Img_data.shape[1]:
        rate_1 = 1152.0 / Img_data.shape[0]
        rate_2 = 768.0 / Img_data.shape[1]
        Img_data = cv2.resize(Img_data, (0, 0), fx=rate_2, fy=rate_1)
        Gt_data[:, 0] = Gt_data[:, 0] * rate_2
        Gt_data[:, 1] = Gt_data[:, 1] * rate_1
        print(img_path)


    kpoint = np.zeros((Img_data.shape[0], Img_data.shape[1]))
    for count in range(0, len(Gt_data)):
        if int(Gt_data[count][1]) < Img_data.shape[0] and int(Gt_data[count][0]) < Img_data.shape[1]:
            kpoint[int(Gt_data[count][1]), int(Gt_data[count][0])] = 1

    height, width = Img_data.shape[0], Img_data.shape[1]

    m = int(width / 384)
    n = int(height / 384)
    fname = img_path.split('/')[-1]
    print(fname)
    # root_path = img_path.split('img_')[0]
    root_path = os.path.dirname(img_path)
    # print(img_path)
    # print(root_path)
    # print(root_path.split('/')[-2])


    if root_path.split('/')[-2] == 'train_data':

        for i in range(0, m):
            for j in range(0, n):
                crop_img = Img_data[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                crop_kpoint = kpoint[j * 384: 384 * (j + 1), i * 384:(i + 1) * 384, ]
                gt_count = np.sum(crop_kpoint)

                # save_fname = str(i) + str(j) + str('_') + fname
                save_fname = fname
                save_path = root_path.replace('train_data', 'my_train_data') + save_fname
                h5_path = save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map/')

                with h5py.File(h5_path, 'w') as hf:
                    hf['gt_count'] = gt_count
                cv2.imwrite(save_path, crop_img)

                density_map = gaussian_filter(crop_kpoint, 2)
                density_map = density_map
                density_map = density_map / np.max(density_map) * 255
                density_map = density_map.astype(np.uint8)
                density_map = cv2.applyColorMap(density_map, 2)
                result = np.hstack((density_map, crop_img))
                cv2.imwrite(save_path.replace('images', 'gt_show/').replace('jpg', 'jpg'), result)
        
        print("ok")

    else:
#         save_path = root_path.replace('test', 'my_test_data') + fname
#         print(save_path)
#         cv2.imwrite(save_path, Img_data)

#         gt_count = np.sum(kpoint)
#         with h5py.File(save_path.replace('.jpg', '.h5').replace('images', 'gt_density_map'), 'w') as hf:
#             hf['gt_count'] = gt_count
            
        print("nooo")
