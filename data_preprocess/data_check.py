import os
import subprocess
import multiprocessing
import time


def render_obj(model_root,obj_root):
	start_time = time.time()
	print(model_root,' rendering start')
	p = subprocess.Popen(['blender','--background','--python','render_blender.py','--','--output_folder',model_root,obj_root],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
	standard_out = p.stdout.readlines()

	end_time = time.time()
	print(model_root,' rendering ends, cost:', end_time - start_time)

counter = 0

N_VIEWS = 12
SIDE = ['upside','downside']

IDS_ROOT = 'ids/'
CLASS_IDS_ALL = (
    '02691156,02828884,02933112,02958343,03001627,03211117,03636649,' +
    '03691459,04090263,04256520,04379243,04401088,04530566')
CLASSES = CLASS_IDS_ALL.split(',')

TYPE = ['val','test','train']

DATASET_ROOT = '../data/dataset/'
SHAPENET_ROOT = '/home4/data/xieyunwei/ShapeNetCore.v2/'

f = open(os.path.join(IDS_ROOT,'missing.txt'),'w')

for c in CLASSES:
    for t in TYPE:
        f_ids = open(os.path.join(IDS_ROOT,'%s_%s_ids.txt' % (c,t)))
        f_ids_all = f_ids.readlines()
        for line in f_ids_all:
            if line[-1] == '\n':
                line = line[:-1]
            

            if len(line) <= 5:
                continue

            class_id = c
            model_id = line.split('/')[1]
            
            model_root = os.path.join(SHAPENET_ROOT,class_id,model_id)
            
            flag = True
            
            render_root = os.path.join(model_root,'render','render_{0}_{1:03d}.png'.format('downside', int(11 * 30)))
                    	
            if not os.path.exists(render_root):
                flag = False
                        
            if flag == False:
                print('%d: %s is missing' % (counter, line))
                f.write('%s\n' % line)
                counter += 1

            
print('missing %d!' % counter)
f.close()


pool = multiprocessing.Pool(processes = 8)

print('start rendering!')

counter = 0

f_ids = open(os.path.join(IDS_ROOT,'missing.txt'), 'r')
f_ids_all = f_ids.readlines()
for line in f_ids_all:
    if line[-1] == '\n':
        line = line[:-1]
    class_id = line.split('/')[0]
    model_id = line.split('/')[1]
    
    model_root = os.path.join(SHAPENET_ROOT,class_id,model_id)
    obj_root = os.path.join(model_root,'models','model_normalized.obj')
    if not os.path.exists(obj_root):
        print('%s not exist' % obj_root)
    else:
        pool.apply_async(render_obj,(model_root,obj_root))
    counter += 1

pool.close()
pool.join()
print('Quit!')
e_time = time.time()
print('In all ',counter,' models')
