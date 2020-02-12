import cv2
import torch
import numpy as np
from imageio import imread, imwrite
import Utils

# If you need to use GPU to accelerate (especially for the need of converting many images)
USE_GPU = True # set False to disable

if __name__ == '__main__':
    # First we create two object
    # e2c is the module to convert equirectangular into cubemap
    # c2e is the module to convert cubemap back to equirectangular
    e2c = Utils.Equirec2Cube(512, 1024, 256, CUDA=USE_GPU)
    c2e = Utils.Cube2Equirec(512, 1024, 256, CUDA=USE_GPU)
    
    # Load image and reshape to 512x1024
    img = cv2.resize(imread('fig/image.jpg', pilmode='RGB'), (1024, 512), interpolation=cv2.INTER_AREA)
    batch = torch.FloatTensor(img.astype(np.float32)/255).permute(2, 0, 1)[None, ...]
    if USE_GPU: batch = batch.cuda()
    
    # First we convert the image to cubemap and then convert it back to equirectangular
    cubemap_tensor = e2c(batch)
    equirec_tensor = c2e(cubemap_tensor)
    print (cubemap_tensor.shape, equirec_tensor.shape)
    
    cubemap = cubemap_tensor.permute(0, 2, 3, 1).cpu().numpy()
    equirec = equirec_tensor.permute(0, 2, 3, 1).cpu().numpy()
    
    # Now we save the cubemap to disk
    order = ['back', 'down', 'front', 'left', 'right', 'up']
    for i, term in enumerate(order):
        face = (cubemap[i] * 255).astype(np.uint8)
        imwrite('fig/gen_face_%s.jpg'%term, face)

    # Now we save the equirectangular to disk
    equi = (equirec[0, ...] * 255).astype(np.uint8)
    imwrite('fig/gen_equirectangular.jpg', equi)
