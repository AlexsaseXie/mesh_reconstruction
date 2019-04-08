import os
import subprocess

IDS_ROOT = 'ids/'
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

TYPE = ['train','test','val']

DATASET_ROOT = '../data/dataset/'
SHAPENET_ROOT = '/home4/data/xieyunwei/ShapeNetCore.v2/'

for c in CLASSES:
	for t in TYPE:
		f_ids = open(os.path.join(IDS_ROOT,'%s_%s_ids.txt' % (c,t)))
		f_ids_all = f_ids.readlines()
		for line in f_ids_all:
			if line[-1] == '\n':
				line = line[:-1]
			class_id = c
			model_id = line.split('/')[1]
			
			model_root = os.path.join(SHAPENET_ROOT,class_id,model_id)
			obj_root = os.path.join(model_root,'models','model_normalized.obj')
			subprocess.call(['blender','--python','render_blender.py','--','--output_folder',model_root,obj_root])

			break
		break
	break
			

			