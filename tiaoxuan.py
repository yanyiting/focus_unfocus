#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 10 11:54:44 2019

@author: root
"""

import cv2
import os
from keras.models import *
import glob
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
resnet_model = load_model('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/model/Resnet50_gray224/ResNet50_gray.hdf5')
validation_folders = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val'
label_dir = sorted(os.listdir(validation_folders))
for i in range(len(label_dir)):
    focus_pngs = glob.glob(validation_folders+'/'+label_dir[i]+'/'+'*.png')
    for j,focus_png in enumerate(focus_pngs):
        img = cv2.imread(focus_png)
        #img_predcit =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_predcit = np.expand_dims(img, axis=0)
        img_predcit = img_predcit/255.
        prediction = resnet_model.predict(img_predcit)
        if np.argmax(prediction)<=1:
            if i>1:
                continue
            else:
                save_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/2_focus/focus'
        if np.argmax(prediction)>1:
            if i<=1:
                continue
            else:
                save_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/2_focus/unfocus'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        cv2.imwrite(save_dir + '/' + focus_pngs[j].split('/')[-1], img)
            
        
    
    
    
    
    
    
    