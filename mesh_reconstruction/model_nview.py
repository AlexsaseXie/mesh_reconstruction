import chainer
import chainer.functions as cf
import chainer.links as cl
import neural_renderer

import loss_functions
import renderer
import voxelization

from model_resnet import *
from model_discriminator import *

'''
class Encoder(chainer.Chain):
    def __init__(self, dim_in=4, dim_out=512, dim1=64, dim2=1024):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        with self.init_scope():
            dim_hidden = [dim1 * 2 ** 0, dim1 * 2 ** 1, dim1 * 2 ** 2, dim2, dim2]
            self.conv1 = cl.Convolution2D(dim_in, dim_hidden[0], 5, stride=2, pad=2)
            self.conv2 = cl.Convolution2D(dim_hidden[0], dim_hidden[1], 5, stride=2, pad=2)
            self.conv3 = cl.Convolution2D(dim_hidden[1], dim_hidden[2], 5, stride=2, pad=2)
            self.linear1 = cl.Linear(dim_hidden[2] * 8 * 8, dim_hidden[3])
            self.linear2 = cl.Linear(dim_hidden[3], dim_hidden[4])
            self.linear3 = cl.Linear(dim_hidden[4], dim_out)

    def __call__(self, x):
        x = cf.relu(self.conv1(x))
        x = cf.relu(self.conv2(x))
        x = cf.relu(self.conv3(x))
        x = x.reshape((x.shape[0], -1))
        x = cf.relu(self.linear1(x))
        x = cf.relu(self.linear2(x))
        x = cf.relu(self.linear3(x))
        return x
'''


class Decoder(chainer.Chain):
    def __init__(
            self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0,
            centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()

        with self.init_scope():
            self.vertices_base, self.faces = neural_renderer.load_obj(filename_obj)
            self.num_vertices = self.vertices_base.shape[0]
            self.num_faces = self.faces.shape[0]
            self.centroid_scale = centroid_scale
            self.bias_scale = bias_scale
            self.obj_scale = 0.5

            dim = 1024
            dim_hidden = [dim, dim * 2]
            self.linear1 = cl.Linear(dim_in, dim_hidden[0])
            self.linear2 = cl.Linear(dim_hidden[0], dim_hidden[1])
            self.linear_centroids = cl.Linear(dim_hidden[1], 3)
            self.linear_bias = cl.Linear(dim_hidden[1], self.num_vertices * 3)
            self.linear_centroids.W.lr = centroid_lr
            self.linear_centroids.b.lr = centroid_lr
            self.linear_bias.W.lr = bias_lr
            self.linear_bias.b.lr = bias_lr

    def to_gpu(self):
        super(Decoder, self).to_gpu()
        self.vertices_base = chainer.cuda.to_gpu(self.vertices_base)
        self.faces = chainer.cuda.to_gpu(self.faces)

    def __call__(self, x):
        h = cf.relu(self.linear1(x))
        h = cf.relu(self.linear2(h))
        centroids = self.linear_centroids(h) * self.centroid_scale
        bias = self.linear_bias(h) * self.bias_scale
        bias = cf.reshape(bias, (-1, self.num_vertices, 3))

        base = self.vertices_base * self.obj_scale
        base = self.xp.broadcast_to(base[None, :, :], bias.shape)

        sign = self.xp.sign(base)
        base = self.xp.absolute(base)
        base = self.xp.log(base / (1 - base))

        centroids = cf.broadcast_to(centroids[:, None, :], bias.shape)
        centroids = cf.tanh(centroids)
        scale_pos = 1 - centroids
        scale_neg = centroids + 1

        vertices = cf.sigmoid(base + bias)
        vertices = vertices * sign
        vertices = cf.relu(vertices) * scale_pos - cf.relu(-vertices) * scale_neg
        vertices += centroids
        vertices *= 0.5
        faces = self.xp.tile(self.faces[None, :, :], (x.shape[0], 1, 1))

        return vertices, faces


class Model(chainer.Chain):
    def __init__(self, filename_obj='./data/obj/sphere_642.obj', lambda_smoothness=0., lambda_std = 0.01, n_views=2):
        super(Model, self).__init__()
        self.lambda_smoothness = lambda_smoothness
        self.lambda_std = lambda_std
        self.n_views = n_views
        self.vertices_predicted_a = None
        self.vertices_predicted_b = None
        with self.init_scope():
            self.encoder = ResNet18(dim_out=512)
            self.decoder = Decoder(filename_obj)
            self.smoothness_loss_parameters = loss_functions.smoothness_loss_parameters(self.decoder.faces)

            self.renderer = renderer.Renderer()
            self.renderer.image_size = 128
            self.renderer.viewing_angle = 15.
            self.renderer.anti_aliasing = True

    def to_gpu(self, device=None):
        super(Model, self).to_gpu()
        self.smoothness_loss_parameters = [chainer.cuda.to_gpu(p) for p in self.smoothness_loss_parameters]

    def predict(self, images, viewpoints):
        # image : [batch_size, n_views, 4, 128, 128]
        # viewpoints : [batch_size, n_views, 3]
        batch_size = images.shape[0]
        
        images = self.xp.reshape(images, (batch_size * images.shape[1], images.shape[2], images.shape[3], images.shape[4]))
        # viewpoints_t : viewpoints shift on the second axis
        if self.n_views > 1:
            viewpoints_t = self.xp.concatenate((viewpoints[:,1:,:], self.xp.expand_dims(viewpoints[:,0,:], axis=1)),axis = 1)
            viewpoints_t = self.xp.reshape(viewpoints_t, (batch_size * viewpoints.shape[1], viewpoints.shape[2]))
            viewpoints = self.xp.reshape(viewpoints, (batch_size * viewpoints.shape[1], viewpoints.shape[2]))
            viewpoints = self.xp.concatenate((viewpoints, viewpoints_t), axis=0)
        else :
            viewpoints = self.xp.reshape(viewpoints, (batch_size * viewpoints.shape[1], viewpoints.shape[2]))
        self.renderer.eye = viewpoints
        
        vertices, faces = self.decoder(self.encoder(images))  # [1_view1, 1_view2, ... 1_viewn , 2_view1 ...]
        if self.n_views > 1:
            vertices_c = cf.concat((vertices, vertices), axis=0)  # [1_view1, 1_view2, ... 1_viewn , 2_view1 ...] * 2
            faces_c = cf.concat((faces, faces), axis=0).data  # [1_view1, 1_view2, ... 1_viewn , 2_view1 ...] * 2
            silhouettes = self.renderer.render_silhouettes(vertices_c, faces_c)  # [1_view1 / 1_view1 ...] + [1_view1/ 1_view2, .. 1_viewn/1view1 ...]
            silhouettes_a_a = silhouettes[0: batch_size * self.n_views]
            silhouettes_a_a = cf.reshape(silhouettes_a_a,(batch_size, self.n_views, silhouettes_a_a.shape[1], silhouettes_a_a.shape[2]))
            
            silhouettes_a_nexta = silhouettes[batch_size * self.n_views:]
            silhouettes_a_nexta = cf.reshape(silhouettes_a_nexta,(batch_size, self.n_views, silhouettes_a_nexta.shape[1], silhouettes_a_nexta.shape[2]))
            
            faces = neural_renderer.vertices_to_faces(vertices, faces)
            # faces : (batch_size * n_views) * n_faces * 3 * 3
            f01 = self.xp.linalg.norm(faces[:,:,0,:] - faces[:,:,1,:], axis=2) # (batch_size * n_views) * n_faces 
            f12 = self.xp.linalg.norm(faces[:,:,1,:] - faces[:,:,2,:], axis=2)
            f20 = self.xp.linalg.norm(faces[:,:,2,:] - faces[:,:,0,:], axis=2)
            
            distances = self.xp.linalg.stack((f01,f12,f20), axis=2)
            distances = cf.reshape(distances, (batch_size, self.n_views, distances.shape[1], vertices.shape[2]))

            vertices = cf.reshape(vertices, (batch_size, self.n_views, vertices.shape[1], vertices.shape[2]))
            return silhouettes_a_a, silhouettes_a_nexta, vertices, distances
        else :
            vertices_c = vertices
            faces_c = faces.data
            silhouettes = self.renderer.render_silhouettes(vertices_c, faces_c)
            silhouettes_a_a = silhouettes[0: batch_size * self.n_views]
            silhouettes_a_a = cf.reshape(silhouettes_a_a,(batch_size, self.n_views, silhouettes_a_a.shape[1], silhouettes_a_a[2]))
            
            faces = neural_renderer.vertices_to_faces(vertices, faces)
            # faces : (batch_size * n_views) * n_faces * 3 * 3
            f01 = self.xp.linalg.norm(faces[:,:,0,:] - faces[:,:,1,:], axis=2) # (batch_size * n_views) * n_faces 
            f12 = self.xp.linalg.norm(faces[:,:,1,:] - faces[:,:,2,:], axis=2)
            f20 = self.xp.linalg.norm(faces[:,:,2,:] - faces[:,:,0,:], axis=2)
            
            distances = self.xp.linalg.stack((f01,f12,f20), axis=2)
            distances = cf.reshape(distances, (batch_size, self.n_views, distances.shape[1], vertices.shape[2]))

            vertices = cf.reshape(vertices, (batch_size, self.n_views, vertices.shape[1], vertices.shape[2]))
            return silhouettes_a_a, None, vertices, distances
        

    def reconstruct(self, images):
        vertices, faces = self.decoder(self.encoder(images))
        return vertices, faces

    def reconstruct_and_render(self, images_in, viewpoints):
        self.renderer.eye = viewpoints
        vertices, faces = self.reconstruct(images_in)
        textures = self.xp.ones((viewpoints.shape[0], faces.shape[1], 2, 2, 2, 3), 'float32')
        images_out = self.renderer.render(vertices, faces, textures)

        return images_out

    def evaluate_iou(self, images, voxels):
        vertices, faces = self.decoder(self.encoder(images))
        faces = neural_renderer.vertices_to_faces(vertices, faces).data
        faces = faces * 1. * (32. - 1) / 32. + 0.5  # normalization
        voxels_predicted = voxelization.voxelize(faces, 32, False)
        voxels_predicted = voxels_predicted.transpose((0, 2, 1, 3))[:, :, :, ::-1]
        iou = (voxels * voxels_predicted).sum((1, 2, 3)) / (0 < (voxels + voxels_predicted)).sum((1, 2, 3))
        return iou

    def __call__(self, images, viewpoints):
        # predict vertices and silhouettes
        silhouettes_a_a, silhouettes_a_nexta, vertices, distances = (
            self.predict(images, viewpoints))

        if self.n_views == 1 :
            loss_silhouettes = (loss_functions.iou_loss(images[:, 0, 3, :, :], silhouettes_a_a[:,0,:,:]))
        else:
            loss_silhouettes = 0
            for i in range(self.n_views):
                loss_silhouettes += (loss_functions.iou_loss(images[:, i, 3, :, :], silhouettes_a_a[:,i,:,:]))
            for i in range(self.n_views):
                loss_silhouettes += (loss_functions.iou_loss(images[:, (i + 1) % self.n_views, 3, :, :], silhouettes_a_nexta[:,i,:,:]))
            loss_silhouettes = loss_silhouettes / (2 * self.n_views)
        # compute loss
        #loss_silhouettes = (
        #                       loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_a_a) +
        #                       loss_functions.iou_loss(images_a[:, 3, :, :], silhouettes_b_a) +
        #                       loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_a_b) +
        #                       loss_functions.iou_loss(images_b[:, 3, :, :], silhouettes_b_b)) / 4
        if self.lambda_smoothness != 0:
            loss_smoothness = 0
            for i in range(self.n_views):
                loss_smoothness += loss_functions.smoothness_loss(vertices[:,i,:,:], self.smoothness_loss_parameters)
            loss_smoothness = loss_smoothness / self.n_views
            #loss_smoothness = (
            #                      loss_functions.smoothness_loss(vertices_a, self.smoothness_loss_parameters) +
            #                      loss_functions.smoothness_loss(vertices_b, self.smoothness_loss_parameters)) / 2
        else:
            loss_smoothness = 0

        if self.lambda_std != 0:
            loss_std = 0
            for i in range(self.n_views):
                loss_std += loss_functions.variance_loss(distances[:,i,:])
            loss_std = loss_std / self.n_views
        else:
            loss_std = 0
        
        loss = loss_silhouettes + self.lambda_smoothness * loss_smoothness + self.lambda_std * loss_std

        # report
        loss_list = {
            'loss_silhouettes': loss_silhouettes,
            'loss_smoothness': loss_smoothness,
            'loss_std': loss_std,
            'loss': loss,
        }
        chainer.reporter.report(loss_list, self)

        return loss
