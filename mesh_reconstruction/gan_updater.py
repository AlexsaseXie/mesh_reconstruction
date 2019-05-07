import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable

def loss_l2(h, t):
    return F.sum((h-t)**2) / np.prod(h.data.shape)

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0

        params = kwargs.pop('params')
        self._img_size = params['img_size']
        self._img_chan = params['img_chan']
        self._latent_len = params['latent_len']
        self._dis_iter = params['dis_iter']
        self._batch_size = params['batch_size']
        self._lambda_gp = params['lambda_gp']
        self._n_views = params['n_views']

        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        xp = self.gen.xp
        self._iter += 1

        opt_d = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        device = self.device
        images, viewpoints, real_images, real_viewpoints = self.converter(batch, device)

        #loss_gen = self.gen.gen_loss(images, viewpoints, silhouettes_a_a, silhouettes_a_nexta, vertices, distances)
        self.gen.n_views = 1
        
        for i in range(self._dis_iter):

            d_fake, _, _, _ = (
                self.gen.predict(images[:,i % self._n_views, :, :, :], real_viewpoints))

            d_real = real_images

            y_fake = self.dis(Variable(d_fake))
            y_real = self.dis(Variable(d_real))

            w1 = F.average(y_fake-y_real)

            loss_dis = w1

            # gp
            eta = np.random.rand()
            c = (d_real * eta + (1.0 - eta) * d_fake).astype('f')
            y = self.dis(Variable(c))

            g = xp.ones_like(y.data)
            grad_c = self.dis.backward(Variable(g))
            grad_c_l2 = F.sqrt(F.sum(grad_c**2, axis=(1, 2, 3)))

            loss_gp = loss_l2(grad_c_l2, 1.0)

            loss_dis += self._lambda_gp * loss_gp

            # update
            opt_d.zero_grads()
            loss_dis.backward()
            opt_d.update()

        chainer.report({'loss': loss_dis,'loss_w1': w1}, self.dis)
        opt_g = self.get_optimizer('gen')

        loss_gen = 0

        # update gan
        for i in range(self._n_views):
            d_fake, _, _, _ = (
                    self.gen.predict(images[:, i, :, :, :], real_viewpoints))

            y_fake = self.dis(d_fake)
            loss_gen -= F.average(y_fake)

        chainer.report({'loss_ad': loss_gen}, self.gen)

        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()


        # update supervised
        self.gen.n_views = self._n_views

        silhouettes_a_a, silhouettes_a_nexta, vertices, distances = (
            self.gen.predict(images, viewpoints))

        loss_gen = self.gen.gen_loss(images, viewpoints, silhouettes_a_a, silhouettes_a_nexta, vertices, distances)

        chainer.report({'loss_supervised': loss_gen}, self.gen)

        opt_g.zero_grads()
        loss_gen.backward()
        opt_g.update()


