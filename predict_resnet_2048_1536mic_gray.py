# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import scipy.io as scio
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
save_dir = '/cptjack/totem/yanyiting/Eight_classification/data/predict_png'

img_names = sorted(os.listdir(directory_name))
patch_size=224

font = cv2.FONT_HERSHEY_SIMPLEX
for k in range(len(img_names)):
    img= cv2.imread(directory_name+'/'+img_names[k], 0)
    img_1 = cv2.imread(directory_name+'/'+img_names[k])
    for w in range(img.shape[1]//patch_size):
        for h in range(img.shape[0]//patch_size):
            crop = img[patch_size*(h): patch_size*(h+1), patch_size*(w): patch_size*(w+1)]
            new_img = np.array([crop,crop,crop])
            new_img = new_img.transpose((1,2,0))
            new_img = new_img/255
            new_img = np.expand_dims(new_img, axis=0)
            resnet_preds = resnet_model.predict(new_img)

            prediction = np.argmax(resnet_preds)
            if prediction==0:
                cv2.rectangle(img_1, (patch_size*(w), patch_size*(h)), (patch_size*(w+1), patch_size*(h+1)), (0,0,255), 3)
                cv2.putText(img_1, str(round(resnet_preds[0][0],2)), (patch_size*(w)+patch_size//2, patch_size*(h)+patch_size//2), font, 2, (0, 0, 255), 1)
            else:
                cv2.rectangle(img_1, (patch_size*(w), patch_size*(h)), (patch_size*(w+1), patch_size*(h+1)), (0,0,0), 3)
                cv2.putText(img_1, str(round(resnet_preds[0][0],2)), (patch_size*(w)+patch_size//2, patch_size*(h)+patch_size//2), font, 2, (0, 0, 0), 1)



    if not os.path.exists(save_dir):os.makedirs(save_dir)
    cv2.imwrite(save_dir+'/'+img_names[k], img_1)


                     


#    
#from sklearn.metrics import classification_report
#target_name = ['focus', 'unfocus']
#print(classification_report(labels, prediction_list, target_names = target_name))
#
#import matplotlib.pyplot as plt
#import itertools
#from sklearn.metrics import confusion_matrix
#cm= confusion_matrix(labels,prediction_list)
#plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
#plt.title("Confusion matrix",size = 15)
#plt.colorbar()
#tick_marks = np.arange(2)
#plt.xticks(tick_marks,['focus', 'unfocus'],rotation = 45,size = 10)
#plt.yticks(tick_marks,['focus', 'unfocus'],rotation=45,size = 10)
#plt.tight_layout()
#plt.ylabel("Actual label",size = 15)
#plt.xlabel("Predicted",size =15)
#width,height = cm.shape
#a =[0,0]
#for i in  range(2):
#    for j in range(2):
#        a[i] = cm[i][j]+a[i]
#for x in range(width):
#    for y in range(height):
#        plt.annotate(str(np.round(cm[x][y]/a[x],2)),xy = (y,x),horizontalalignment="center",verticalalignment='center')
#
#                    
#
#import numpy as np
#import matplotlib.pyplot as plt
#x_1 = focus_list
#x_2 = unfocus_list
#y_1 = [np.arange(0,len(focus_list),1)]
#y_2 = [np.arange(0,len(unfocus_list),1)] 
#plt.scatter(y_1,x_1,label='focus')
##plt.scatter(y_2, x_2,label='unfocus')
#plt.legend()
#
#
#
##def draw_scatter(n,s):
##test=np.load('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/all_scores.npy',encoding = 'latin1')
##x1=data[:,0]
##y1 = data[:,3]
##x2 = np.random.uniform(,)
