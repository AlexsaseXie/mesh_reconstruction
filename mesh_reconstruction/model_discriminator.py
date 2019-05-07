import chainer
import chainer.functions as cf
import chainer.links as cl

class Discriminator(chainer.Chain):
    def __init__(self, img_size = 128, img_channel=1, pos_size=3):
        super(Discriminator, self).__init__()
        self.img_size = img_size
        self.img_channel = img_channel
        with self.init_scope():
            self.conv1 = cl.Convolution2D(in_channels=self.img_channel,out_channels=32,ksize=5,pad=2,stride=2)
            self.bn1 = cl.BatchNormalization(32)
            self.conv2 = cl.Convolution2D(in_channels=48, out_channels=64, ksize=5,pad=2,stride=2)
            self.bn2 = cl.BatchNormalization(64)
            self.conv3 = cl.Convolution2D(in_channels=64, out_channels=128, ksize=5, pad=2, stride=2)
            self.bn3 = cl.BatchNormalization(128)
            self.conv4 = cl.Convolution2D(in_channels=128, out_channels=128, ksize=5, pad=2, stride=2)
            self.bn4 = cl.BatchNormalization(128)
            self.conv5 = cl.Convolution2D(in_channels=128, out_channels=1, ksize=5, pad=2, stride=2)
            self.bn5 = cl.BatchNormalization(1)

            self.linear1 = cl.Linear(in_size = pos_size, out_size=16)
            self.bnl = cl.BatchNormalization(16)

    def __call__(self, img, pos):
        h = cf.leaky_relu(self.bnl(self.linear1(pos)))
        img_h = cf.leaky_relu(self.bn1(self.conv1(img)))
        h = cf.broadcast_to(h[:, :, None, None], (img_h.shape[0], h.shape[1] , img_h.shape[2], img_h.shape[3]))

        h = cf.concat((img_h, h), axis=1)
        h = cf.leaky_relu(self.bn2(self.conv2(h)))
        h = cf.leaky_relu(self.bn3(self.conv3(h)))
        h = cf.leaky_relu(self.bn4(self.conv4(h)))
        h = self.bn5(self.conv5(h))
        return h
