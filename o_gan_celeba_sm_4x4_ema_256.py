#! -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.initializers import RandomNormal
import os,json
import warnings
warnings.filterwarnings("ignore") # 忽略keras带来的满屏警告


if not os.path.exists('samples'):
    os.mkdir('samples')


imgs = glob.glob('../../CelebA-HQ/train/*.png')
np.random.shuffle(imgs)
img_dim = 256
z_dim = 256
num_layers = int(np.log2(img_dim)) - 3
max_num_channels = img_dim * 8
f_size = img_dim // 2**(num_layers + 1)
batch_size = 48


def imread(f, mode='gan'):
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        x = x.astype(np.float32)
        return x / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    """
    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []


class ScaleShift(Layer):
    """平移缩放
    """
    def __init__(self, **kwargs):
        super(ScaleShift, self).__init__(**kwargs)
    def call(self, inputs):
        z, beta, gamma = inputs
        for i in range(K.ndim(z) - 2):
            beta = K.expand_dims(beta, 1)
            gamma = K.expand_dims(gamma, 1)
        return z * (gamma + 1) + beta


def SelfModulatedBatchNormalization(h, c):
    num_hidden = z_dim
    dim = K.int_shape(h)[-1]
    h = BatchNormalization(center=False, scale=False)(h)
    beta = Dense(num_hidden, activation='relu')(c)
    beta = Dense(dim)(beta)
    gamma = Dense(num_hidden, activation='relu')(c)
    gamma = Dense(dim)(gamma)
    return ScaleShift()([h, beta, gamma])


# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (4, 4),
               strides=(2, 2),
               padding='same',
               kernel_initializer=RandomNormal(0, 0.02))(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(z_dim,
          kernel_initializer=RandomNormal(0, 0.02))(x)

e_model = Model(x_in, x)
e_model.summary()


# 生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels,
          kernel_initializer=RandomNormal(0, 0.02))(z)
z = Reshape((f_size, f_size, max_num_channels))(z)
z = SelfModulatedBatchNormalization(z, z_in)
z = Activation('relu')(z)

for i in range(num_layers):
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (4, 4),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=RandomNormal(0, 0.02))(z)
    z = SelfModulatedBatchNormalization(z, z_in)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (4, 4),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=RandomNormal(0, 0.02))(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型
x_in = Input(shape=(img_dim, img_dim, 3))
z_in = Input(shape=(z_dim, ))

x_real = x_in
x_fake = g_model(z_in)
x_fake_ng = Lambda(K.stop_gradient)(x_fake)

z_real = e_model(x_real)
z_fake = e_model(x_fake)
z_fake_ng = e_model(x_fake_ng)

train_model = Model([x_in, z_in],
                    [z_real, z_fake, z_fake_ng])

z_real_mean = K.mean(z_real, 1, keepdims=True)
z_fake_mean = K.mean(z_fake, 1, keepdims=True)
z_fake_ng_mean = K.mean(z_fake_ng, 1, keepdims=True)

def correlation(x, y):
    x = x - K.mean(x, 1, keepdims=True)
    y = y - K.mean(y, 1, keepdims=True)
    x = K.l2_normalize(x, 1)
    y = K.l2_normalize(y, 1)
    return K.sum(x * y, 1, keepdims=True)

t1_loss = z_real_mean - z_fake_ng_mean
t2_loss = z_fake_mean - z_fake_ng_mean
z_corr = correlation(z_in, z_fake)
qp_loss = 0.25 * t1_loss[:, 0]**2 / K.mean((x_real - x_fake_ng)**2, axis=[1, 2, 3])

train_model.add_loss(K.mean(t1_loss + t2_loss - 1. * z_corr) + K.mean(qp_loss))
train_model.compile(optimizer=RMSprop(1e-4, 0.99))
train_model.metrics_names.append('t_loss')
train_model.metrics_tensors.append(K.mean(t1_loss))
train_model.metrics_names.append('z_corr')
train_model.metrics_tensors.append(K.mean(z_corr))

# 检查模型结构
train_model.summary()


class ExponentialMovingAverage:
    """对模型权重进行指数滑动平均。
    用法：在model.compile之后、第一次训练之前使用；
    先初始化对象，然后执行inject方法。
    """
    def __init__(self, model, momentum=0.9999):
        self.momentum = momentum
        self.model = model
        self.ema_weights = [K.zeros(K.shape(w)) for w in model.weights]
    def inject(self):
        """添加更新算子到model.metrics_updates。
        """
        self.initialize()
        for w1, w2 in zip(self.ema_weights, self.model.weights):
            op = K.moving_average_update(w1, w2, self.momentum)
            self.model.metrics_updates.append(op)
    def initialize(self):
        """ema_weights初始化跟原模型初始化一致。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        K.batch_set_value(zip(self.ema_weights, self.old_weights))
    def apply_ema_weights(self):
        """备份原模型权重，然后将平均权重应用到模型上去。
        """
        self.old_weights = K.batch_get_value(self.model.weights)
        ema_weights = K.batch_get_value(self.ema_weights)
        K.batch_set_value(zip(self.model.weights, ema_weights))
    def reset_old_weights(self):
        """恢复模型到旧权重。
        """
        K.batch_set_value(zip(self.model.weights, self.old_weights))


EMAer3 = ExponentialMovingAverage(train_model, 0.999)
EMAer3.inject()
EMAer4 = ExponentialMovingAverage(train_model, 0.9999)
EMAer4.inject()


# 采样函数
def sample(path, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                z_sample -= (z_sample).mean(axis=1, keepdims=True)
                z_sample /= (z_sample).std(axis=1, keepdims=True)
                x_sample = g_model.predict(z_sample * 0.9)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100
        self.Z = np.random.randn(self.n_size**2, z_dim)
    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample('samples/test_%s.png' % self.batch,
                self.n_size, self.Z)
            sample_ae('samples/test_ae_%s.png' % self.batch)
            EMAer3.apply_ema_weights()
            sample('samples/test_ema3_%s.png' % self.batch,
                self.n_size, self.Z)
            sample_ae('samples/test_ema3_ae_%s.png' % self.batch)
            train_model.save_weights('./train_model_ema3.weights')
            EMAer3.reset_old_weights()
            EMAer4.apply_ema_weights()
            sample('samples/test_ema4_%s.png' % self.batch,
                self.n_size, self.Z)
            sample_ae('samples/test_ema4_ae_%s.png' % self.batch)
            train_model.save_weights('./train_model_ema4.weights')
            EMAer4.reset_old_weights()
        self.batch += 1


if __name__ == '__main__':

    trainer = Trainer()
    img_data = img_generator(imgs, 'gan', batch_size)

    train_model.fit_generator(img_data.__iter__(),
                              steps_per_epoch=len(img_data),
                              epochs=1000,
                              callbacks=[trainer])
