# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.ticker import MultipleLocator,FormatStrFormatter
data = pd.read_csv('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/code/Resnet50_gray_jingxi224/2019_08_14_log.csv')
acc = data['acc']
loss = data['loss']
val_loss = data['val_loss']
val_acc = data['val_acc']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(loss,color='r',linestyle = 'dashed',marker = 'o',label = 'loss')
plt.plot(val_loss,color='k',linestyle = 'dashed',marker = 'o',label = 'val_loss')
plt.legend(loc = 'best')
ax.set_title('The changing curve of Loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
plt.savefig('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/tmp/Resnet_fine_loss.jpg')
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.plot(acc,color = 'r',linestyle = 'dashed',marker = 'o',label = 'acc')
plt.plot(val_acc,color = 'k',linestyle = 'dashed',marker = 'o',label='val_acc')
plt.legend(loc = 'best')
ax.set_title('The changing curve of Acc')
ax.set_xlabel('Epoch')
ax.set_ylabel('Acc')
plt.savefig('/cptjack/totem/yanyiting/gray_focus_unfocus/gray/tmp/Resnet_fine_acc.jpg')