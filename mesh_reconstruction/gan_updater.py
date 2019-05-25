import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, optimizers, serializers, Variable

class Updater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.gen, self.dis = kwargs.pop('models')
        self._iter = 0

        params = kwargs.pop('params')
        #self._img_size = params['img_size']
        #self._img_chan = params['img_chan']
        #self._latent_len = params['latent_len']
        self._dis_iter = params['dis_iter']
        self._batch_size = params['batch_size']
        self._lambda_gp = params['lambda_gp']
        self._n_views = params['n_views']

        super(Updater, self).__init__(*args, **kwargs)

    def update_core(self):
        self._iter += 1

        opt_d = self.get_optimizer('dis')

        batch = self.get_iterator('main').next()
        device = self.device
        images, viewpoints, real_images, real_viewpoints = self.converter(batch, device)
        
        f_real_viewpoints = real_viewpoints.reshape((real_viewpoints.shape[0], 1, real_viewpoints.shape[1]))
        #loss_gen = self.gen.gen_loss(images, viewpoints, silhouettes_a_a, silhouettes_a_nexta, vertices, distances)
        self.gen.n_views = 1
        
        for i in range(self._dis_iter):
            d_fake, _, _ = self.gen.predict_and_render_current(images[:,(i % self._n_views):(i % self._n_views + 1), :, :, :], f_real_viewpoints)

            d_real = real_images[:,3:4,:,:]

            y_fake = self.dis(d_fake, real_viewpoints)
            y_real = self.dis(d_real, real_viewpoints)

            #w1 = F.average(y_fake-y_real)
            w1 = F.sum(F.softplus(-y_real)) / (self._batch_size * 64) + F.sum(F.softplus(y_fake)) / ( self._batch_size * 64 )

            loss_dis = w1

            # gp
            if self._lambda_gp != 0:
                xp = chainer.cuda.get_array_module(d_real)
                eta = xp.random.uniform(0., 1., (self._batch_size, 1, 1, 1))
                c = (d_real * eta + (1.0 - eta) * d_fake)
            
                v_real_viewpoints = Variable(real_viewpoints)
                mid_c = self.dis(c, v_real_viewpoints)
                grad_c, _ = chainer.grad([mid_c], [c, v_real_viewpoints],
                                 enable_double_backprop=True, loss_scale=0.2)
                #print(grad_c.shape, grad_c.data[0,0,0,0])
                grad_c = F.sqrt(F.batch_l2_norm_squared(grad_c))

                loss_gp = F.mean_squared_error(grad_c, xp.ones_like(grad_c.data))

                loss_dis += self._lambda_gp * loss_gp

            # update
            self.dis.cleargrads()
            loss_dis.backward()
            opt_d.update()

        chainer.report({'loss': loss_dis,'loss_w1': w1}, self.dis)
        opt_g = self.get_optimizer('gen')

        loss_gen = 0

        # update gan
        for i in range(self._n_views):
            d_fake, _, _ = (
                    self.gen.predict_and_render_current(images[:, i: i+1, :, :, :], f_real_viewpoints))

            y_fake = self.dis(d_fake, real_viewpoints)
            #loss_gen -= F.average(y_fake)
            loss_gen += F.sum(F.softplus(-y_fake)) / (self._batch_size * 64)

        chainer.report({'loss_ad': loss_gen}, self.gen)

        #self.gen.cleargrads()
        #loss_gen.backward()
        #opt_g.update()


        # update supervised
        self.gen.n_views = self._n_views

        loss_supervised = self.gen(images, viewpoints)

        chainer.report({'loss_supervised': loss_supervised}, self.gen)

        loss_gen += loss_supervised

        self.gen.cleargrads()
        loss_gen.backward()
        opt_g.update()


