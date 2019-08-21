# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 03:06:30 2019

@author: root
"""
import pandas as pd
import numpy as np
import argparse
import datetime
import random
import keras
import glob
import time
import sys
import os

from keras.models import *
from keras.layers import Dense,GlobalAveragePooling2D,Dropout,MaxPooling2D,Flatten,BatchNormalization
from keras.callbacks import CSVLogger,ModelCheckpoint,ReduceLROnPlateau
from sklearn.metrics import classification_report,confusion_matrix
#from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
#from clssification_models.resnet import ResNet18
from keras.initializers import orthogonal
from keras.utils import to_categorical
from generators import DataGenerator
from scipy.misc import imresize
from keras.models import Model
from keras import backend as K
from keras import optimizers
from skimage import io
from keras.callbacks import CSVLogger,Callback,EarlyStopping
from keras.callbacks import ModelCheckpoint,LearningRateScheduler,ReduceLROnPlateau
from keras.callbacks import TensorBoard,ModelCheckpoint
from keras.utils import multi_gpu_model
import time
from PIL import Image
import h5py
from sklearn.model_selection import StratifiedKFold
from scipy import misc
from sklearn.model_selection import train_test_split
import cv2
import gc
import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input,decode_predictions
from PIL import Image
import os
import glob
import xlrd
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import StratifiedKFold
from scipy import misc
from sklearn.model_selection import train_test_split


os.environ['CUDA_VISIBLE_DEVICES']='0'
base = ResNet50(weights =  None,include_top = False,input_shape = (224,224,3))
base.load_weights('/cptjack/sys_software_bak/keras_models/models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5')
top_model = Sequential()
top_model.add(base)
top_model.add(MaxPooling2D(pool_size=(2,2)))
top_model.add(Flatten())
top_model.add(Dropout(0.5))
top_model.add(Dense(128,activation = 'relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2,activation='softmax',kernel_initializer=orthogonal()))
top_model.summary()

for layer in top_model.layers:
    layer.trainable = True


trainable_params = int(np.sum([K.count_params(p)for p in set(top_model.trainable_weights)]))
non_trainable_params = int(np.sum([K.count_params(p) for p in set(top_model.non_trainable_weights)]))
print("model Stats")
print("="*30)
print("Total Parameters:{:,}".format((trainable_params+non_trainable_params)))
print("Non-Trainable Parameters:{:,}".format(non_trainable_params))
print("Trainable Parameters:{:,}\n".format(trainable_params))

class Mycbk(ModelCheckpoint):
    def __init__(self, model, filepath ,monitor = 'val_acc',mode='max', save_best_only=True):
        self.single_model = model
        super(Mycbk,self).__init__(filepath, monitor, save_best_only, mode)
    def set_model(self,model):
        super(Mycbk,self).set_model(self.single_model)
        
def get_callbacks(filepath,model,j,patience=5):
    es = EarlyStopping('val_loss', patience=patience, mode="min")
    msave = Mycbk(model, './Resnet50_gray_cross_'+str(j)+'/'+filepath)
    file_dir = './Resnet50_gray_cross_'+str(j)+'/'+'log/'+ time.strftime('%Y_%m_%d',time.localtime(time.time()))
    if not os.path.exists(file_dir): os.makedirs(file_dir)
    tb_log = TensorBoard(log_dir=file_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, 
                                  patience=2, verbose=0, mode='min', epsilon=-0.95, cooldown=0, min_lr=1e-8)
    log_cv = CSVLogger('./Resnet50_gray_cross_'+str(j)+'/' + time.strftime('%Y_%m_%d',time.localtime(time.time()))  +'_log.csv', separator=',', append=True)
    return [es, msave,reduce_lr,tb_log,log_cv]

                

folders = ['/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/2_focus/focus/',
                 '/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/2_focus/unfocus/']
img_width,img_height = 224,224
batch_size_for_generators = 32
train_datagen = DataGenerator(rescale = 1./255,rotation_range=178,horizontal_flip=True,vertical_flip=True,shear_range=0.6,fill_mode='nearest',stain_transformation = True)
valid_datagen = DataGenerator(rescale = 1./255)

all_images = []
all_labels = []

for index, folder in enumerate(folders):
    files = glob.glob(folder + "*.png")
    images = io.imread_collection(files)
    images = [image for image in images] ### Reshape to (299, 299, 3) ###
    labels = [index] * len(images)
    all_images = all_images + images
    all_labels = all_labels + labels
    print("Class: %s. Size: %d" %(folder.split("/")[-2], len(images)))

all_images = np.stack(all_images)
all_images = (all_images/255).astype(np.float32) ### Standardise
#
all_labels = np.array(all_labels).astype(np.int32)
#all_labels = to_categorical(all_labels, num_classes = np.unique(all_labels).shape[0])

k=5
#num_val_samples = all_images.shape[0]//k
all_scores=[]
stratified_folder=StratifiedKFold(n_splits = 5, random_state=0, shuffle=False)
for i,(train_index, test_index) in enumerate(stratified_folder.split(all_images, all_labels)):
    print('processing fold #', i)

    train_images = all_images[train_index]
    train_labels = all_labels[train_index]
    Y_train = to_categorical(train_labels,num_classes = np.unique(train_labels).shape[0])

    
    
    valid_images = all_images[test_index]
    valid_labels = all_labels[test_index]
    Y_valid = to_categorical(valid_labels,num_classes = np.unique(valid_labels).shape[0])





    
    train_gen = train_datagen.flow(train_images, Y_train, batch_size = batch_size_for_generators)
    valid_gen = valid_datagen.flow(valid_images, Y_valid, batch_size = batch_size_for_generators)

    file_path = 'Resnet.hdf5'
    callbacks_s = get_callbacks(file_path,top_model,i,patience=5)
    train_steps = train_images.shape[0]//batch_size_for_generators
    valid_steps = valid_labels.shape[0]//batch_size_for_generators
    LearningRate = 0.001
    decay = 0.0001
    n_epochs = 30
    sgd = optimizers.SGD(lr=LearningRate,decay=LearningRate/n_epochs,momentum = 0.9,nesterov = True)
    top_model.compile(optimizer = sgd,loss = 'categorical_crossentropy',metrics=['accuracy'])

    
    top_model.fit_generator(generator=train_gen,epochs=n_epochs,steps_per_epoch=train_steps,validation_data=valid_gen,
                        validation_steps = valid_steps, callbacks=callbacks_s, verbose=1)
    
    
    
    top_model.load_weights('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/Resnet50_gray_cross_'+str(i)+'/Resnet.hdf5')
    val_loss, val_metrics=top_model.evaluate(valid_images, Y_valid, batch_size =32)
    all_scores.append((val_loss, val_metrics))
    del train_images,Y_train
    del valid_labels,Y_valid
    gc.collect()

np.save('all_scores.npy',all_scores)