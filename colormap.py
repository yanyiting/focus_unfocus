# -*- coding: utf-8 -*-
def create_colormap(svs_im,matrix_0,title,output_dir):
    plt_size = (svs_im.size[0]//100,svs_im.size[1]//100)
    fig,ax = plt.subplots(figsize = plt_size,dpi = 100)
    matrix = matrix_0.copy()
    matrix = cv2.resize(mqtrix,svs_im.size,interpolation = cv2.INIER_AREA)
    cax = ax.imshow(matrix,cmap = plt.cm.jet,alpha = 0.45)
    svs_im_npy = np.array(svs_im.convert('RGBA'))
    svs_im_npy[:,:][matrix[:,:]>0]=0
    ax.imshow(svs_im_npy)
    
    max_matrix_value= matrix.max()
    plt.colorbar(cax,tick = np.linspace(0,max_matrix_value,25,endpoint = True))
    
    ax.imshow(svs_im_npy)
    
    max_matrix_value = matrix.max()
    plt.colorbar(cax,ticks = np.linspace(0,max_matrix_value,25,endpoint = True))
    ax.set_title(title,fontsize = 20)
    plt.axis('off')
    
    if not os.path.isdir(output_dir):os.makedirs(output_dir)
    
    plt.savefig(os.path.join(output_dir,title+'.png'))
    plt.close('all')
    
    
