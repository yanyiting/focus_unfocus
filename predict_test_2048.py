# -*- coding: utf-8 -*-
import scipy.io as  scio
from skimage import io
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import math
import glob
from keras.models import *
import time
import openslide as opsl
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES']='0'
resnet_model = load_model('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/Resnet50_gray_jingxi224/ResNet50_gray.hdf5')
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'
save_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/prediction_gray_2048'
color_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/prediction_color_2048'
focus_names = sorted(os.listdir(directory_name))
labels = []
prediction_list = []
focus_list = []
unfocus_list = []
font = cv2.FONT_HERSHEY_SIMPLEX

for k in range(len(focus_names)):
    print(k)
    img_names = sorted(os.listdir(directory_name+'/'+focus_names[k]))
    for l in range(len(img_names)):
        img = cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l],0)
        img_3 = cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l])
        img_2 = img.copy()
#        new_img = new_img/255
        resnet_preds = resnet_model.predict(new_img)
        
        img = img/255
        patch_img = []
    for i in range(img.shape[0]//224):
        for j in range(img.shape[1]//224):
    #                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
            patch = img[224*i:224*(i+1),224*j:224*(j+1)]
            patch = cv2.resize(patch,(224,224))
            patch_img.append(patch)
            patch = np.expand_dims(patch,axis=0)
            patch_pro = Resnet_model.predict(patch)
            if labels[k]!= np.argmax(patch_pro):
                cv2.putText(img_2,str(np.round(patch_pro[0][0],2)),(224*i+112,224*j+112),font,0.5(255,0,0),2)
            else:
                cv2.putText(img_2,str(np.round(patch_pro[0][0],2)),(224*i+112,224*j+112),font,0.5(0,0,0),2)
                patch_pro.append(patch_pro)


def read_directory(directory_name,resnet_model):
    img_num = os.listdir(directory_name)
    prediction_list = []
    for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k])
        img=cv2.resize(img,(256,256))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img= cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l],0)
        img_3 = cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l]) 
        img_2 = img.copy()
        new_img = np.array([img,img,img])
        new_img = new_img.transpose((1,2,0))
        new_img = np.expand_dims(new_img, axis=0)
        img = img/255
        patch_img=[]
        for i in range(img.shape[1]//224):
            for j in range(img.shape[0]//224):
#                patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch = img[224*i:224*(i+1),224*j:224*(j+1)]
                patch = cv2.resize(patch,(224,224))
                patch_img.append(patch)
             
        patch_img = np.array(patch_img)
        predictions = resnet_model.predict_classes(patch_img)
        prediction= Counter(predictions).most_common(1)[0][0]
        print(prediction)
        prediction_list.append((img_num[k].split(".")[0],prediction))
    return prediction_list


directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/png'
prediction_list = read_directory(directory_name,resnet_model)
np.save('prediction_resnet_list.npy', prediction_list)   
