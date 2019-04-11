import os
import numpy as np
from skimage import io
from skimage import transform

NUM_VIEWS = 12
DEGREE = 360 // NUM_VIEWS
ELEVATION = ['upside','downside']

IDS_ROOT = 'ids/'
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

TYPE = ['val','test','train']

DATASET_ROOT = '../data/dataset/'
SHAPENET_ROOT = '/home4/data/xieyunwei/ShapeNetCore.v2/'

for c in CLASSES:
    for t in TYPE:
        f_ids = open(os.path.join(IDS_ROOT,'%s_%s_ids.txt' % (c,t)))
        f_ids_all = f_ids.readlines()

        dataset_images = []

        for line in f_ids_all:
            if line[-1] == '\n':
                line = line[:-1]
            
            if len(line) <= 5:
                break

            class_id = c
            model_id = line.split('/')[1]
            
            model_root = os.path.join(SHAPENET_ROOT,class_id,model_id)
            render_root = os.path.join(model_root,'render','render_upside_000.png')

            if not os.path.exists(render_root):
                continue
            
            images = []
            for side in ELEVATION:
                for i in range(NUM_VIEWS):
                    azimuth = i * DEGREE
                    png_root = os.path.join(model_root,'render','render_{0}_{1:03d}.png'.format(side, int(azimuth)))

                    img = io.imread(png_root)
                    assert img.shape[2] == 4
                    #img = transform.resize(img,(64,64))
                    img = img.astype('uint8')
                    images.append(img.transpose(2,0,1))
            
            images = np.stack(images, axis = 0)
            assert images.shape[0] == 24

            dataset_images.append(images)
        
        dataset_images = np.stack(dataset_images , axis = 0)
        print(dataset_images.shape)
        print(dataset_images.dtype)
        #dataset_images = ('%s_%s' % (c,t), dataset_images)
        #dataset_images = np.stack([dataset_images], axis = 0)

        np.savez_compressed(os.path.join(DATASET_ROOT,'%s_%s_render.npz' % (c,t)), dataset_images)
        print('%s,%s' % (c,t),'saved')
    


                       
