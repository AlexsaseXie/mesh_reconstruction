import chainer
import chainer.functions as cf
import chainer.links as cl

class ResBlock(chainer.Chain):
    def __init__(self, dim_in, dim_mid, dim_out, stride = 1):
        super(ResBlock, self).__init__()
        self.dim_in = dim_in
        self.dim_mid = dim_mid
        self.dim_out = dim_out
        self.stride = stride
        with self.init_scope():
            self.left_conv1 = cl.Convolution2D(dim_in, dim_mid, ksize=3, stride=stride, pad=1)
            self.left_bn1 = cl.BatchNormalization(dim_mid)
            # relu
            self.left_conv2 = cl.Convolution2D(dim_mid, dim_out, ksize=3, stride=1, pad=1)
            self.left_bn2 = cl.BatchNormalization(dim_out)
			
            if dim_in != dim_out or stride != 1:
                self.shortcut_conv1 = cl.Convolution2D(dim_in, dim_out, ksize=1, stride=stride)
                self.shortcut_bn = cl.BatchNormalization(dim_out)

    def __call__(self, x):
        out = self.left_bn2(self.left_conv2(cf.relu(self.left_bn1(self.left_conv1(x)))))
        if self.dim_in != self.dim_out or self.stride != 1:
            out += self.shortcut_bn(self.shortcut_conv1(x))
        else:
            out += x
        out = cf.relu(out)
        return out

class ResLayer(chainer.ChainList):
    def __init__(self, dim_in, dim_out, num_blocks, stride):
        super(ResLayer, self).__init__()
        self.add_link(ResBlock(dim_in, dim_out, dim_out, stride))
        for _ in range(num_blocks - 1):
            self.add_link(ResBlock(dim_out, dim_out, dim_out))

    def __call__(self, x):
        for f in self.children():
            x = f(x)
        return x

class ResNet18(chainer.Chain):
    def __init__(self, dim_in = 4, img_size=128, dim_out=2048):
        super(ResNet18, self).__init__()
        self.dim_in = dim_in
        with self.init_scope():
            self.conv1 = cl.Convolution2D(dim_in, 64, ksize=3, stride=1, pad=1)
            self.bn1 = cl.BatchNormalization(64)
            # relu

            self.layer1 = ResLayer(dim_in=64,dim_out=64,num_blocks=2,stride=1)
            self.layer2 = ResLayer(dim_in=64,dim_out=128,num_blocks=2,stride=2) #/2
            self.layer3 = ResLayer(dim_in=128,dim_out=256,num_blocks=2,stride=2) #/2
            self.layer4 = ResLayer(dim_in=256,dim_out=512,num_blocks=2,stride=2) #/2

            self.fc = cl.Linear((img_size // 32) * (img_size // 32) * 512, dim_out)
            

    def __call__(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = cf.average_pooling_2d(out, 4)
        out = out.reshape((x.shape[0], -1))
        out = self.fc(out)
        return out

if __name__ == '__main__':
    model = ResNet18()
    print(model.xp)




