#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 13:04:39 2019

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide as opsl
import sys
import os
import glob
from keras.models import load_model
os.environ['CUDA_VISIBLE_DEVICES']='0'
resnet_model = load_model('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/Resnet50_gray_jingxi224/ResNet50_gray.hdf5')
picture_dir = '/cptjack/totem/yanyiting/Focus test/'
picture_names = glob.glob(picture_dir + '*.svs')
step_size = 224
patch_size = 224
save_dir = '/cptjack/totem/yanyiting/Focus test/predict_dir'
if not os.path.exists(save_dir):os.makedirs(save_dir)


for i, picture_name in enumerate(picture_names):
    print(i)
    predictions = []
    slide = opsl.OpenSlide(picture_name)
    for w in range((slide.level_dimensions[0][0]-patch_size)//step_size):
        for h in range((slide.level_dimensions[0][1]-patch_size)//step_size):
            img = np.array(slide.read_region((step_size*w, step_size*(h)), 0, (patch_size, patch_size)))[:, :, 0:3]
            if np.sum((img[:,:,0]>=229)&(img[:,:,1]>=229)&(img[:,:,2]>=229)) /(patch_size*patch_size)>0.5:
                predictions.append((step_size*w, step_size*(w+1), 0, 2))
                continue
            img_1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            new_img = np.array([img_1,img_1,img_1])
            new_img = new_img.transpose((1,2,0))
            new_img = (new_img/255).astype(np.float32)
            new_img = np.expand_dims(new_img, axis=0)
            probability = resnet_model.predict(new_img)
            prediction = np.argmax(probability)
            predictions.append((step_size*w, step_size*(w+1), probability[0][0], prediction))
            
    np.save(save_dir + picture_name.split('/')[-1][:-4] + '.npy', predictions)
                
            