import os
import numpy as np
import sys
sys.path.append('..')

from mesh_reconstruction import voxelization
import chainer
import neural_renderer
import time

IDS_ROOT = 'ids/'
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

TYPE = ['val','test','train']

DATASET_ROOT = '../data/dataset/'
SHAPENET_ROOT = '/home4/data/xieyunwei/ShapeNetCore.v2/'

start_time = time.time()
print 'start calc voxel!'
counter = 0
for c in CLASSES:
    for t in TYPE:
        f_ids = open(os.path.join(IDS_ROOT,'%s_%s_ids.txt' % (c,t)))
        f_ids_all = f_ids.readlines()

        voxels = []

        for line in f_ids_all:
            if line[-1] == '\n':
                line = line[:-1]
            
            if len(line) <= 5:
                break

            class_id = c
            model_id = line.split('/')[1]
            
            model_root = os.path.join(SHAPENET_ROOT,class_id,model_id)
            obj_root = os.path.join(model_root,'models','model_normalized.obj')
            render_root = os.path.join(model_root,'render','render_upside_000.png')

            if not os.path.exists(render_root):
                continue

            vertices, faces = neural_renderer.load_obj(obj_root)
            #vertices = vertices.reshape((1,vertices.shape[0],vertices.shape[1]))
            #faces = faces.reshape((1,faces.shape[0],faces.shape[1]))
            vertices = chainer.cuda.to_gpu(vertices.astype('float32'), 0)
            faces = chainer.cuda.to_gpu(faces, 0)
            
            vertices = vertices[None,:,:]
            faces = faces[None,:,:]
            faces_coordinates = neural_renderer.vertices_to_faces(vertices, faces)
            #print(faces.shape, type(faces))
            voxel = voxelization.voxelize(faces_coordinates, 32, True)

            #print(voxel.shape)
            voxels.append(voxel.astype(bool).get()[0,:,:,:])
            counter += 1 
            if counter % 10 == 0:
                print 'finish ', counter, 'models'
        
        voxels = np.stack(voxels , axis = 0)
        print(voxels.shape)
        print(voxels.dtype)

        np.savez_compressed(os.path.join(DATASET_ROOT,'%s_%s_voxel.npz' % (c,t)), voxels)
        print('%s,%s' % (c,t),'saved')
        
        break
    break

end_time = time.time()
print 'finished output voxel in', end_time - start_time , ' sec'
print 'in all ', counter ,' models'
