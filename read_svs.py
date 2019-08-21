# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import openslide as opsl
import sys
#
#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/58751.svs')
#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/58761.svs')
#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/1.svs')
#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/68113_.svs')
#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/68116_.svs')
slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/68119_.svs')

#slide = opsl.OpenSlide('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/data/Focus test/*.svs')
level_count = slide.level_count
print('level_count=',level_count)
level_count = 4

Wh = np.zeros((len(slide.level_dimensions),2))
for i in range(len(slide.level_dimensions)):
    print('level_dimensions:'+str(i))
    Wh[i,:] = slide.level_dimensions[i]
    print('W = %d,H = %d'%(Wh[i,0],Wh[i,1]))
    
Ds = np.zeros((len(slide.level_dimensions),2))
for i in range(len(slide.level_downsamples)):
    Ds[i,0] = slide.level_downsamples[i]
    Ds[i,1] = slide.get_best_level_for_downsample(Ds[i,0])
    print('%d of level_downsamples is %d:the best_level_for_downsample is %d'%(i,Ds[i,0],Ds[i,1]))
print(slide.level_downsamples)

plt.rcParams['figure.figsize']=15,15
slide_thumbnail = slide.get_thumbnail(slide.level_dimensions[2])
plt.imshow(slide_thumbnail)
print(slide_thumbnail.size)

slide_thumbnail = slide.get_thumbnail((1200,613))
plt.imshow(slide_thumbnail)
print(slide_thumbnail.size)
 
tiles = slide.read_region((16000,30000),0,(1024,512))   
plt.imshow(tiles)
print(tiles.size)

tiles = slide.read_region((4000,9900),1,(1024,512))
plt.imshow(tiles)


print(type(slide_thumbnail),type(tiles))
tiles_n = np.array(tiles)
print(type(tiles_n),tiles_n.shape)
tiles_n = np.array(slide.read_region((4000,9900),1,(1024,512)))[:,:,0:3]


#def(slide,patch_c,patch_r,step):
#    slide.level_dimensions[0]
#    w_count = int(slide.level_dimensions[0][0]//step)
#    h_count = int(slide.level_dimensions[0][1]//step)
#    for x in range(1,w_count-1):
#        for y in range(int(h_count)):
#            slide_region = np.array(slide.read_region((x*step,y*step),0,
#                                                      (patch_c,patch_r)))[:,:,0:3]
#slide.close()            
    
    

