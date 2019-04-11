import os
import numpy as np
import sys
sys.path.append('..')

from mesh_reconstruction import voxelization

import neural_renderer

BATCH_SIZE = 8

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
            render_root = os.path.join(model_id,'render','render_upside_000.png')

            if not os.path.exists(render_root):
                continue

            vertices, faces = neural_renderer.load_obj(obj_root)
            vertices = vertices[None, :,: ]
            faces = faces[None, :, :]
            faces_coordinates = neural_renderer.vertices_to_faces(vertices, faces)
            voxel = voxelization.voxelize(faces_coordinates, 32, False)

            print(type(voxel))
            voxels.append(voxel)
            
        
        voxels = np.stack(voxels , axis = 0)
        print(voxels.shape)
        print(voxels.dtype)

        np.savez_compressed(os.path.join(DATASET_ROOT,'%s_%s_voxel.npz' % (c,t)), voxels)
        print('%s,%s' % (c,t),'saved')
        
        break
    break
