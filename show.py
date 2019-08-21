#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 05:34:00 2019

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide as opsl
import sys
import os
import glob

npy_dir = '/cptjack/totem/yanyiting/Focus test/predict_dir'
npy_names = glob.glob(npy_dir+'/*.npy')
picture_dir = '/cptjack/totem/yanyiting/Focus test/'
step_size = 224
patch_size = 224

result = []

for k,npy_name in enumerate(npy_names):
    prediction = np.load(npy_name)
    slide = opsl.OpenSlide(picture_dir + npy_name.split('/')[-1][:-4]+'.svs')
    slide_h = slide.level_dimensions[0][1]-patch_size
    slide_w = slide.level_dimensions[0][0]-patch_size
    init_matrix = np.zeros((slide_h-patch_size,slide_w), dtype = np.float32)
    num = 0
    for w in range((slide.level_dimensions[0][0]-patch_size)//step_size):
        for h in range((slide.level_dimensions[0][1]-patch_size)//step_size):
            if prediction[num][-1]==1:
                init_matrix[h,w] =  prediction[num][-1]
            num +=1
    
    plt.imshow(init_matrix)
    slide.close()
    result.append(np.sum(init_matrix[:,:]==1 )/(slide_h*slide_w))
    



















