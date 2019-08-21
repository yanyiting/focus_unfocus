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
directory_name = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/micro_png_224'

save_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/prediction_gray'
color_dir = '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/prediction_color'
focus_names = sorted(os.listdir(directory_name))

labels = []
prediction_list = []
focus_list = []
unfocus_list = []
font = cv2.FONT_HERSHEY_SIMPLEX
for k in range(len(focus_names)):
    print(k)
    img_names = sorted(os.listdir(directory_name+'/'+focus_names[k]))
    for l in range(len(img_names)):#
        img= cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l],0)
        img_3 = cv2.imread(directory_name+'/'+focus_names[k]+'/'+img_names[l]) 
        img_2 = img.copy()
        new_img = np.array([img,img,img])
        new_img = new_img.transpose((1,2,0))
        new_img = np.expand_dims(new_img, axis=0)

        new_img = new_img/255
        resnet_preds = resnet_model.predict(new_img)
#        for i in range(len(resnet_preds[0])):
#            resnet_preds[0][i] = resnet_preds[0][i]*(8-i)
#        prediction= Counter(resnet_preds).most_common(1)[0][0]
#        print(prediction)
        prediction = np.argmax(resnet_preds)
        cv2.putText(img_2, str(round(resnet_preds[0][0],2)), (112,112), font, 1, (0, 0, 0), 1)
        cv2.putText(img_3, str(round(resnet_preds[0][0],2)), (112,112), font, 1, (0, 0, 0), 1)
        prediction_list.append(prediction)
        if k==0:
            labels.append(0)
            focus_list.append(resnet_preds[0][0])
        else:
            labels.append(1)
            unfocus_list.append(resnet_preds[0][0])
        if k==0 and k!=prediction:
            if not os.path.exists(save_dir):os.makedirs(save_dir)
            if not os.path.exists(color_dir):os.makedirs(color_dir)
            cv2.imwrite(save_dir+'/'+img_names[l], img_2)
            cv2.imwrite(color_dir+'/'+img_names[l], img_3)

    #prediction_list.append(focus_list)
#np.save('prediction_resnet224_gray_mic.npy', prediction_list)
#np.save('labels_resnet224_gray_mic.npy', labels)
#np.save('pro_list_resnet224_gray_mic.npy', pro_list)                             


    
from sklearn.metrics import classification_report
target_name = ['focus', 'unfocus']
print(classification_report(labels, prediction_list, target_names = target_name))

import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(labels,prediction_list)
plt.imshow(cm,interpolation='nearest',cmap = "Pastel1")
plt.title("Confusion matrix",size = 15)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks,['focus', 'unfocus'],rotation = 45,size = 10)
plt.yticks(tick_marks,['focus', 'unfocus'],rotation=45,size = 10)
plt.tight_layout()
plt.ylabel("Actual label",size = 15)
plt.xlabel("Predicted",size =15)
width,height = cm.shape
a =[0,0]
for i in  range(2):
    for j in range(2):
        a[i] = cm[i][j]+a[i]
for x in range(width):
    for y in range(height):
        plt.annotate(str(np.round(cm[x][y]/a[x],2)),xy = (y,x),horizontalalignment="center",verticalalignment='center')

                    

import numpy as np
import matplotlib.pyplot as plt
x_1 = focus_list
x_2 = unfocus_list
y_1 = [np.arange(0,len(focus_list),1)]
y_2 = [np.arange(0,len(unfocus_list),1)] 
plt.scatter(y_1,x_1,label='focus')
#plt.scatter(y_2, x_2,label='unfocus')
plt.legend()



#def draw_scatter(n,s):
#test=np.load('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/all_scores.npy',encoding = 'latin1')
#x1=data[:,0]
#y1 = data[:,3]
#x2 = np.random.uniform(,)
