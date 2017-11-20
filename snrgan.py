"""Generate fake SNR images.

Intended to be run within Hydrogen.

Based somewhat on
https://mxnet.incubator.apache.org/tutorials/unsupervised_learning/gan.html
"""
import os
from math import floor

import mxnet as mx
import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread

plt.switch_backend('agg')


class RandIter(mx.io.DataIter):
    """Produce random numbers."""

    def __init__(self, batch_size, ndim):
        """Initialize `RandIter`."""
        self.batch_size = batch_size
        self.ndim = ndim
        self.provide_data = [('rand', (batch_size, ndim, 1, 1))]
        self.provide_label = []

    def iter_next(self):  # noqa D102
        return True

    def getdata(self):
        """Return random numbers from a gaussian (normal) distribution.

        With mean=0 and standard deviation = 1
        """
        return [mx.random.normal(0, 1.0, shape=(
            self.batch_size, self.ndim, 1, 1))]


def prime_factors(n):
    """Return prime factorization."""
    i = 2
    factors = []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    if n > 1:
        factors.append(n)
    return factors


def fill_buf(buf, num_images, img, shape):
    """Fill buffer.

    Takes the images in our batch and arranges them in an array so that they
    can be plotted using matplotlib
    """
    width = int(buf.shape[0] / shape[1])
    height = int(buf.shape[1] / shape[0])
    img_width = round((num_images % width) * shape[0])
    img_hight = round(floor(num_images / height) * shape[1])
    buf[img_hight:img_hight + shape[1],
        img_width:img_width + shape[0], :] = img


def visualize(fake, real, cnt=0):
    """Plot two images side by side using matplotlib."""
    # Nx3xNxN to NxNxNx3
    fake = fake.transpose((0, 2, 3, 1))
    # Pixel values from 0-255
    fake = np.clip((fake + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)
    # Repeat for real image
    real = real.transpose((0, 2, 3, 1))
    real = np.clip((real + 1.0) * (255.0 / 2.0), 0, 255).astype(np.uint8)

    # Create buffer array that will hold all the images in our batch Fill the
    # buffer so to arrange all images in the batch onto the buffer array
    n = np.ceil(np.sqrt(fake.shape[0]))
    fbuff = np.zeros(
        (int(n * fake.shape[1]),
         int(n * fake.shape[2]), int(fake.shape[3])), dtype=np.uint8)
    for i, img in enumerate(fake):
        fill_buf(fbuff, i, img, fake.shape[1:3])
    rbuff = np.zeros(
        (int(n * real.shape[1]),
         int(n * real.shape[2]), int(real.shape[3])), dtype=np.uint8)
    for i, img in enumerate(real):
        fill_buf(rbuff, i, img, real.shape[1:3])

    # Create a matplotlib figure with two subplots: one for the real and the
    # other for the fake fill each plot with our buffer array, which creates
    # the image
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(fbuff)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(rbuff)
    plt.show()
    fig.savefig(
        'output/gan-' + str(cnt).zfill(4) + '.png', bbox_inches='tight')


print('Fetching...')

# Use a seed so that we get the same random permutation each time
np.random.seed(2)

# Local files
ims = []
# datadir = "../snr-64x64/"
# datadir = "../snr-96x96/"
datadir = "../snr-128x128/"
# datadir = "../chandra-snrs/"
for filename in os.listdir(datadir):
    if not filename.endswith(".jpg"):
        continue
    im = imread(datadir + filename)
    im = np.swapaxes(im, 1, 2)
    im = np.swapaxes(im, 0, 1)

    ims.append(im)

imgd = ims[0].shape[1]

min_csize = 4
pfac = list(sorted(prime_factors(imgd)))
csize = 1
for i, pf in enumerate(pfac):
    csize *= pf
    if csize >= min_csize:
        pfac = pfac[i + 1:]
        break

pprod = np.prod(pfac)
psize = floor((csize - 1) / 2)
batch_size = 16
nepochs = 100

inputd = np.array(ims)
p = [i for s in [
    np.random.permutation(inputd.shape[0]) for x in range(nepochs)] for i in s]

X = inputd[p]
X = X.astype(np.float32) / (255.0 / 2) - 1.0
image_iter = mx.io.NDArrayIter(X, batch_size=batch_size)

print('Arrays created.')

Z = 100
rand_iter = RandIter(batch_size, Z)

# The generator.
print('Defining MX generator.')
no_bias = True
fix_gamma = True
epsilon = 1e-5 + 1e-12

min_filter = 128

rand = mx.sym.Variable('rand')

gs = []
gbns = []
gacts = []

filt_scale = min_filter * pprod // 2
gs.append(mx.sym.Deconvolution(rand, name='g1', kernel=(
    csize, csize), num_filter=filt_scale, no_bias=no_bias))
gbns.append(mx.sym.BatchNorm(
    gs[-1], name='gbn1', fix_gamma=fix_gamma, eps=epsilon))
gacts.append(mx.sym.Activation(gbns[-1], name='gact1', act_type='relu'))

for pi, pf in enumerate(pfac):
    filt_scale //= 2
    sgi = str(pi + 2)
    if pi < len(pfac) - 1:
        gs.append(mx.sym.Deconvolution(
            gacts[-1], name='g' + sgi, kernel=(csize, csize), stride=(pf, pf),
            pad=(psize, psize), num_filter=filt_scale, no_bias=no_bias))
        gbns.append(
            mx.sym.BatchNorm(
                gs[-1], name='gbn' + sgi, fix_gamma=fix_gamma, eps=epsilon))
        gacts.append(mx.sym.Activation(
            gbns[-1], name='gact' + sgi, act_type='relu'))
    else:
        gs.append(mx.sym.Deconvolution(
            gacts[-1], name='g' + sgi, kernel=(csize, csize), stride=(pf, pf),
            pad=(psize, psize), num_filter=3, no_bias=no_bias))
        generatorSymbol = mx.sym.Activation(
            gs[-1], name='gact' + sgi, act_type='tanh')

# The discriminator.
print('Defining MX discriminator.')
data = mx.sym.Variable('data')

filt_scale = min_filter

ds = []
dbns = []
dacts = []

pf = pfac[-1]
ds.append(mx.sym.Convolution(data, name='d1', kernel=(csize, csize), stride=(
    pf, pf), pad=(psize, psize), num_filter=filt_scale, no_bias=no_bias))
dacts.append(
    mx.sym.LeakyReLU(ds[-1], name='dact1', act_type='leaky', slope=0.2))

for pi, pf in enumerate(pfac):
    filt_scale *= 2
    sgi = str(pi + 2)
    if pi < len(pfac) - 1:
        ds.append(mx.sym.Convolution(
            dacts[-1], name='d' + sgi, kernel=(csize, csize),
            stride=(pf, pf), pad=(psize, psize), num_filter=filt_scale,
            no_bias=no_bias))
        dbns.append(
            mx.sym.BatchNorm(
                ds[-1], name='dbn' + sgi, fix_gamma=fix_gamma, eps=epsilon))
        dacts.append(mx.sym.LeakyReLU(
            dbns[-1], name='dact' + sgi, act_type='leaky', slope=0.2))
    else:
        ds.append(mx.sym.Convolution(
            dacts[-1], name='d' + sgi, kernel=(csize, csize),
            num_filter=1, no_bias=no_bias))
        ds[-1] = mx.sym.Flatten(ds[-1])

label = mx.sym.Variable('label')
discriminatorSymbol = mx.sym.LogisticRegressionOutput(
    data=ds[-1], label=label, name='dloss')

# Hyperperameters
sigma = 0.02
lr = 1.0 / float(len(ims)) / float(batch_size)
beta1 = 0.5
ctx = mx.gpu(0)

print('Creating modules.')
# Generator module
generator = mx.mod.Module(symbol=generatorSymbol, data_names=(
    'rand',), label_names=None, context=ctx)
generator.bind(data_shapes=rand_iter.provide_data)
generator.init_params(initializer=mx.init.Normal(sigma))
generator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods = [generator]

# Discriminator module
discriminator = mx.mod.Module(symbol=discriminatorSymbol, data_names=(
    'data',), label_names=('label',), context=ctx)
discriminator.bind(data_shapes=image_iter.provide_data,
                   label_shapes=[('label', (batch_size,))],
                   inputs_need_grad=True)
discriminator.init_params(initializer=mx.init.Normal(sigma))
discriminator.init_optimizer(
    optimizer='adam',
    optimizer_params={
        'learning_rate': lr,
        'beta1': beta1,
    })
mods.append(discriminator)

# Train
print('Training...')
cnt = 0
for epoch in range(nepochs):
    image_iter.reset()
    for i, batch in enumerate(image_iter):
        # Get a batch of random numbers to generate an image from the generator
        rbatch = rand_iter.next()
        # Forward pass on training batch
        generator.forward(rbatch, is_train=True)
        # Output of training batch is the 64x64x3 image
        outG = generator.get_outputs()

        # Pass the generated (fake) image through the discriminator, and save
        # the gradient
        # Label (for logistic regression) is an array of 0's since this image
        # is fake
        label = mx.nd.zeros((batch_size,), ctx=ctx)
        # Forward pass on the output of the discriminator network
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        # Do the backwards pass and save the gradient
        discriminator.backward()
        gradD = [[grad.copyto(grad.context) for grad in grads]
                 for grads in discriminator._exec_group.grad_arrays]

        # Pass a batch of real images through the discriminator
        # Set the label to be an array of 1's because these are the real images
        label[:] = 1
        batch.label = [label]
        # Forward pass on a batch of images
        discriminator.forward(batch, is_train=True)
        # Do the backwards pass and add the saved gradient from the fake images
        # to the gradient generated by this backwards pass on the real images
        discriminator.backward()
        for gradsr, gradsf in zip(
                discriminator._exec_group.grad_arrays, gradD):
            for gradr, gradf in zip(gradsr, gradsf):
                gradr += gradf
        # Update gradient on the discriminator
        discriminator.update()

        # Now that we've updated the discriminator, let's update the generator
        # First do a forward pass and backwards pass on the newly updated
        # discriminator with the current batch
        discriminator.forward(mx.io.DataBatch(outG, [label]), is_train=True)
        discriminator.backward()
        # Get the input gradient from the backwards pass on the discriminator,
        # and use it to do the backwards pass on the generator
        diffD = discriminator.get_input_grads()
        generator.backward(diffD)
        # Update the gradients on the generator
        generator.update()

        # Increment to the next batch, printing every X batches
        i += 1
        if i % 1 == 0:
            cnt += 1
            print('epoch:', epoch, 'iter:', i)
            print
            print("   From generator:        From data:")

            visualize(outG[0].asnumpy(), batch.data[0].asnumpy(), cnt)
