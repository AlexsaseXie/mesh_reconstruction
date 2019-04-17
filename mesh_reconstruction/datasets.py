import os

import neural_renderer
import numpy as np
import tqdm


class ShapeNet(object):
    def __init__(self, directory=None, class_ids=None, set_name=None):
        self.class_ids = class_ids
        self.set_name = set_name
        self.elevation = 30.
        self.distance = 2.732

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
        del images
        del voxels

    def get_random_batch(self, batch_size):
        data_ids_a = np.zeros(batch_size, 'int32')
        data_ids_b = np.zeros(batch_size, 'int32')
        viewpoint_ids_a = np.zeros(batch_size, 'int32')
        viewpoint_ids_b = np.zeros(batch_size, 'int32')
        for i in range(batch_size):
            class_id = np.random.choice(self.class_ids)
            object_id = np.random.randint(0, self.num_data[class_id])
            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = (object_id + self.pos[class_id]) * 24 + viewpoint_id_a
            data_id_b = (object_id + self.pos[class_id]) * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

        images_a = self.images[data_ids_a].astype('float32') / 255.
        images_b = self.images[data_ids_b].astype('float32') / 255.

        distances = np.ones(batch_size, 'float32') * self.distance
        elevations_a = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_ids_a // 12) * 2 - 1)
        elevations_b = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_ids_b // 12) * 2 - 1)
        viewpoints_a = neural_renderer.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 30)
        viewpoints_b = neural_renderer.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 30)

        return images_a, images_b, viewpoints_a, viewpoints_b

    def get_all_batches_for_evaluation(self, batch_size, class_id):
        data_ids = np.arange(self.num_data[class_id]) + self.pos[class_id]
        viewpoint_ids = np.tile(np.arange(24), data_ids.size)
        data_ids = np.repeat(data_ids, 24) * 24 + viewpoint_ids
        for i in range((data_ids.size - 1) / batch_size + 1):
            images = self.images[data_ids[i * batch_size:(i + 1) * batch_size]].astype('float32') / 255.
            voxels = self.voxels[data_ids[i * batch_size:(i + 1) * batch_size] / 24]
            yield images, voxels

    def __getitem__(self, index):
        if isinstance(index, list) or isinstance(index, np.ndarray):
            batch_size = len(index)
            data_ids_a = np.zeros(batch_size, 'int32')
            data_ids_b = np.zeros(batch_size, 'int32')
            viewpoint_ids_a = np.zeros(batch_size, 'int32')
            viewpoint_ids_b = np.zeros(batch_size, 'int32')
            for i in range(batch_size):
                viewpoint_id_a = np.random.randint(0, 24)
                viewpoint_id_b = np.random.randint(0, 24)
                data_id_a = index[i] * 24 + viewpoint_id_a
                data_id_b = index[i] * 24 + viewpoint_id_b
                data_ids_a[i] = data_id_a
                data_ids_b[i] = data_id_b
                viewpoint_ids_a[i] = viewpoint_id_a
                viewpoint_ids_b[i] = viewpoint_id_b

            images_a = self.images[data_ids_a].astype('float32') / 255.
            images_b = self.images[data_ids_b].astype('float32') / 255.

            distances = np.ones(batch_size, 'float32') * self.distance
            elevations_a = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_ids_a // 12) * 2 - 1)
            elevations_b = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_ids_b // 12) * 2 - 1)

            viewpoints_a = neural_renderer.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 30)
            viewpoints_b = neural_renderer.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 30)

            return images_a, images_b, viewpoints_a, viewpoints_b

        elif isinstance(index, slice):
            raise NotImplementedError
        else:
            batch_size = 1 
            data_ids_a = np.zeros(batch_size, 'int32')
            data_ids_b = np.zeros(batch_size, 'int32')
            viewpoint_ids_a = np.zeros(batch_size, 'int32')
            viewpoint_ids_b = np.zeros(batch_size, 'int32')

            viewpoint_id_a = np.random.randint(0, 24)
            viewpoint_id_b = np.random.randint(0, 24)
            data_id_a = index * 24 + viewpoint_id_a
            data_id_b = index * 24 + viewpoint_id_b
            data_ids_a[i] = data_id_a
            data_ids_b[i] = data_id_b
            viewpoint_ids_a[i] = viewpoint_id_a
            viewpoint_ids_b[i] = viewpoint_id_b

            images_a = self.images[data_ids_a].astype('float32') / 255.
            images_b = self.images[data_ids_b].astype('float32') / 255.

            distances = np.ones(batch_size, 'float32') * self.distance
            elevations_a = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_id_a // 12) * 2 - 1)
            elevations_b = np.ones(batch_size, 'float32') * self.elevation * ((viewpoint_id_a // 12) * 2 - 1)
            viewpoints_a = neural_renderer.get_points_from_angles(distances, elevations_a, -viewpoint_ids_a * 30)
            viewpoints_b = neural_renderer.get_points_from_angles(distances, elevations_b, -viewpoint_ids_b * 30)

            return images_a, images_b, viewpoints_a, viewpoints_b

    def __len__(self):
        return self.model_count

