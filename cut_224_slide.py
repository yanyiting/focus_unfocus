# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import cv2
import xlrd

workbook = xlrd.open_workbook("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
workbook = pd.read_excel("/cptjack/totem/yanyiting/Eight_classification/data/DatabaseInfo.xlsx")
label_name = workbook[['Name','Slice #']]
label_name['Name'][0]
directory_name = '/cptjack/totem/yanyiting/Eight_classification/data/FocusPath'
img_num =sorted( os.listdir(directory_name))

focus = [8,9]
focus1=[7,10]
focus2=[6,11]
focus3 = [5,12]
focus4 = [4,13]
focus5 =[3,14]
focus6=[2,15]
focus7 = [1,16]

for k in range(len(img_num)):
        img = cv2.imread(directory_name+'/'+img_num[k], 0)
        new_img = np.array([img,img,img])
        new_img = new_img.transpose((1,2,0))

        if k<len(img_num)//2:         
            if label_name['Slice #'][k] in focus:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus'
            elif label_name['Slice #'][k] in focus1:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus1'
            elif label_name['Slice #'][k] in focus2:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus2'
            elif label_name['Slice #'][k] in focus3:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus3'
            elif label_name['Slice #'][k] in focus4:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus4'
            elif label_name['Slice #'][k] in focus5:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus5'
            elif label_name['Slice #'][k] in focus6:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus6'
            elif label_name['Slice #'][k] in focus7:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/train/focus7'

        else:
            if label_name['Slice #'][k] in focus:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus'
            elif label_name['Slice #'][k] in focus1:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus1'
            elif label_name['Slice #'][k] in focus2:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus2'
            elif label_name['Slice #'][k] in focus3:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus3'
            elif label_name['Slice #'][k] in focus4:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus4'
            elif label_name['Slice #'][k] in focus5:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus5'
            elif label_name['Slice #'][k] in focus6:
               save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus6'
            elif label_name['Slice #'][k] in focus7:
                save_dir = '/cptjack/totem/yanyiting/Eight_classification/gray/8_focus/val/focus7'
        if not os.path.exists(save_dir): os.makedirs(save_dir)
        
        for j in range(img.shape[1]//224):
            for i in range(img.shape[0]//224):
                    #patch=img[8+72*i+8:8+72*(i+1)+8, 8+72*j+8:8+72*(j+1)+8]
                patch= new_img[224*i:224*(i+1), 224*j:224*(j+1)]
                cv2.imwrite(save_dir+'/'+img_num[k].split('.')[0]+'_'+str(j)+str(i)+'.png',patch)   
        
        
        
        
        
        
        
        
        
       