import os
import numpy as np
from skimage import io
import random

CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

directory = './data/dataset/'
output_directory = './data/test_img/'

for class_id in CLASSES:
    set_name = 'test'
    
    images = (np.load(
        os.path.join(directory, '%s_%s_images.npz' % (class_id, set_name))).items()[0][1])
    
    l = random.sample(range(images.shape[0]), 3)
    for i in range(3):
        img = images[l[i],3,:,:,:]
        img = img.transpose((1,2,0))
        io.imsave(os.path.join(output_directory,  '%s_%d_in.png' % (class_id, i) ), img)

        print '%s_%d img saved' % (class_id, i)
