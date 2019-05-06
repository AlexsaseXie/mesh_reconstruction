import os

import neural_renderer
import numpy as np
import tqdm

import random

class ShapeNet_NView(object):
    def __init__(self, directory=None, class_ids=None, set_name=None, n_views = 3, total_views=24, only_top=True):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.964
        self.distance = 1.1662
        self.n_views = n_views
        self.total_views = total_views

        self.only_top = only_top

        images = []
        voxels = []
        self.num_data = {}
        self.pos = {}
        count = 0
        loop = tqdm.tqdm(self.class_ids)
        loop.set_description('Loading dataset')
        for class_id in loop:
            images.append(np.load(
                os.path.join(directory, '%s_%s_render.npz' % (class_id, set_name))).items()[0][1])
            voxels.append(np.load(
                os.path.join(directory, '%s_%s_voxel.npz' % (class_id, set_name))).items()[0][1])
            self.num_data[class_id] = images[-1].shape[0]
            self.pos[class_id] = count
            count += self.num_data[class_id]
        images = np.concatenate(images, axis=0).reshape((-1, 4, 128, 128))
        images = np.ascontiguousarray(images)
        self.images = images
        self.voxels = np.ascontiguousarray(np.concatenate(voxels, axis=0))
        self.model_count = images.shape[0] // 24

        self.init_viewpoints()
        del images
        del voxels

    def init_viewpoints(self):
        if not self.only_top:
            distance = np.ones(self.total_views, 'float32') * self.distance
            elevation = np.concatenate((np.ones(self.total_views / 2, 'float32') * self.elevation, - np.ones(self.total_views / 2, 'float') * self.elevation) , axis = 0)
            azimuth = np.arange(0, self.total_views, dtype='int32') * (360 / (self.total_views / 2))
            self.viewpoints_pool = neural_renderer.get_points_from_angles(distance, elevation, azimuth)
        else:
            distance = np.ones(self.total_views, 'float32') * self.distance
            elevation = np.ones(self.total_views, 'float32') * self.elevation
            azimuth = np.arange(0, self.total_views, dtype='int32') * (360 / self.total_views)
            self.viewpoints_pool = neural_renderer.get_points_from_angles(distance, elevation, azimuth)

        
    def get_random_batch(self, batch_size):
        data_ids = np.zeros((batch_size, self.n_views), 'int32')
        viewpoint_ids = np.zeros((batch_size, self.n_views), 'int32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])

            r_id = random.sample(range(0, self.total_views), self.n_views)
            for j in range(self.n_views):
                viewpoint_id = r_id[j]
                data_id = (object_id + self.pos[class_id]) * 24 + viewpoint_id
                data_ids[i,j] = data_id
                viewpoint_ids[i,j] = viewpoint_id

        images = self.images[data_ids].astype('float32') / 255.

        #viewpoints = []
        #distances = np.ones(self.n_views, 'float32') * self.distance
        #for i in range(batch_size):
        #    elevations_tmp = np.ones(self.n_views, 'float32') * self.elevation * ((viewpoint_ids[i,:] // 12) * 2 - 1)
        #    viewpoints_tmp = neural_renderer.get_points_from_angles(distances, elevations_tmp, -viewpoint_ids[i,:] * 30)
        #    viewpoints.append(viewpoints_tmp)

        #viewpoints = np.stack(viewpoints, axis=0)
        viewpoints = self.viewpoints_pool[viewpoint_ids]
        return images, viewpoints

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(self.total_views), data_ids.size)
        data_ids = np.repeat(data_ids, self.total_views) * self.total_views + viewpoint_ids
        for i in range((data_ids.size - 1) / batch_size + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] / self.total_views]
            yield images, voxels

    def __getitem__(self, index):
        #print index
        if isinstance(index, list) or isinstance(index, np.ndarray):
            batch_size = len(index)
            data_ids = np.zeros((batch_size, self.n_views), 'int32')
            viewpoint_ids = np.zeros((batch_size, self.n_views), 'int32')
            for i in range(batch_size):
                r_id = random.sample(range(0,self.total_views), self.n_views)
                for j in range(self.n_views):
                    viewpoint_id = r_id[j]
                    data_id = index[i] * self.total_views + viewpoint_id
                    data_ids[i,j] = data_id
                    viewpoint_ids[i,j] = viewpoint_id

            images = self.images[data_ids].astype('float32') / 255.

            #distances = np.ones(self.n_views, 'float32') * self.distance
            #for i in range(batch_size):
            #    elevations_tmp = np.ones(self.n_views, 'float32') * self.elevation * ((viewpoint_ids[i,:] // 12) * 2 - 1)
            #    viewpoints_tmp = neural_renderer.get_points_from_angles(distances, elevations_tmp, -viewpoint_ids[i,:] * 30)
            #    viewpoints.append(viewpoints_tmp)
        
            #viewpoints = np.stack(viewpoints, axis=0)
            viewpoints = self.viewpoints_pool[viewpoint_ids]
            return images, viewpoints

        elif isinstance(index, slice):
            current, stop, step = index.indices(len(self))
            return self[range(current,stop,step)]
        else:
            raise NotImplementedError

    def __len__(self):
        return self.model_count


if __name__ == '__main__':
    CLASS_IDS_ALL = (
    '03211117')
    DATASET_DIRECTORY = './data/dataset'
    ds = ShapeNet_NView(DATASET_DIRECTORY, CLASS_IDS_ALL.split(','), 'train', n_views=3)
    #ds = ShapeNet(DATASET_DIRECTORY, CLASS_IDS_ALL.split(','), 'train')
    print ds.viewpoints_pool
    ds[[1,2,3,4,5]]
    ds[1:5]
