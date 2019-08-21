# -*- coding: utf-8 -*-
from utils import preview
import get_colormap_img as colormap
import openslide as opsl
import numpy as np
import cv2
import time
from PIL import Image,ImageDraw
import os
from keras.models import load_model
import gc
from utils import judge_position
from util import metrics
from skimage import io
os.environ['CUDA_VISIBLE_DEVICES']='0'

def get_out_img(model,svs_file_path,name):
    step = 224
    img_size =224
    y_nd = 1
    livel = 2
    slide = opsl.OpenSlide(svs_file_path)
    Wh = np.zeros((len(slide.level_dimensions),2))
    for i in range(len(slide.level_dimensions)):
        Ds[i,0]= slide.level_downsamples[i]
        Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0])
    w_count = int(slide.level_dimensions[0][0])//step
    h_count = int(slide.level_dimensions[0][1]//step
    
    out_img = np.zeros([h_count,w_count])
    out_img1 = np.zeros([h_count,w_count])
    out_img0 = np.zeros([h_count,w_count])
    
    regions = judge_position.get_regions(xml_file,Ds[livel,0])
    
    base_dir = '/cptjack/totem/yanyiting/Focus test'
    base_path = 
                  
                  
    
