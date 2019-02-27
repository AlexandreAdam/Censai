import sys
sys.path = ['', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python27.zip', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/plat-linux2', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-tk', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-old', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-dynload', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/site-packages', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/site-packages/IPython/extensions', '/mnt/home/yhezaveh/.ipython'] + sys.path

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io
#import h5py
from skimage.transform import resize

tf.enable_eager_execution()
execfile("definitions.py")

batch_size = 50

npix_side = 256
#input_filename = '/mnt/home/yhezaveh/Censai/_matlab/data/KAP_001_512X512_X2.mat'
input_filename = '/mnt/home/yhezaveh/Censai/_matlab/data/old/KAP_001.mat'
#input_filename = '/mnt/home/yhezaveh/Censai/data/KAP_001_normalized.mat'

mat_contents = scipy.io.loadmat(input_filename)
kappa = 2 * mat_contents['KAP']
#f = h5py.File(input_filename)
#for k, v in f.items():
#    kappa = 2. * np.array(v)

kappa = resize(kappa, (npix_side,npix_side), order=1, preserve_range=True)
kappa = np.moveaxis( kappa , 2 , 0)


RESTORE = False

def plot(samples):
    fig = plt.figure(figsize=(8*2, 6*2))
    gs = gridspec.GridSpec(6, 8)
    gs.update(wspace=None, hspace=None)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample, cmap='Greys_r')
        plt.title(str(np.max(sample)))
    return fig

tf.reset_default_graph()



VAE_obj = VAE(npix_side = npix_side)

optimizer = tf.train.AdamOptimizer(0.0005)
checkpoint_path = "checkpoints/model_VAE"
im_i = 0

if (RESTORE==True):
	VAE_obj.load_weights(checkpoint_path)

with tf.device('/gpu:0'):
    for train_iter in range(10000000):

        ind = np.random.randint(0, high=1500, size=batch_size)
        batch_kappa = kappa[ind,:].reshape((-1,npix_side,npix_side))
        batch_kappa = tf.cast( batch_kappa , datatype)

        with tf.GradientTape() as tape:
            tape.watch(VAE_obj.variables)
            cost_value , decoded_im = VAE_obj.cost(batch_kappa)
        weight_grads = tape.gradient(cost_value, [VAE_obj.variables] )

        clipped_grads = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
        optimizer.apply_gradients(zip(clipped_grads, VAE_obj.variables), global_step=tf.train.get_or_create_global_step())

        if (((train_iter+1)%10)==0):
            print( train_iter , cost_value.numpy() )
        if (((train_iter+1)%500)==0):
            im_i = im_i + 1
            print( "saving weights." )
            VAE_obj.save_weights(checkpoint_path)
            print( "done." )
            ims = VAE_obj.draw_image(16)
            ims = [np.reshape(ims[i].numpy(), [npix_side, npix_side]) for i in range(16)] + [np.reshape(decoded_im[i,:,:], [npix_side, npix_side]) for i in range(16)] + [np.reshape(batch_kappa[i,:,:], [npix_side, npix_side]) for i in range(16)]
            fig = plot(ims)
            plt.savefig('out_conv/{}.png'.format(str(im_i).zfill(3)), bbox_inches='tight')
            plt.close(fig)

#
# # saver = tf.train.Saver(max_to_keep=None)
#
# # sess = tf.Session()
# # sess.run(tf.global_variables_initializer())
# saver.restore(sess,'../data/weights/VAE')
#
# im_i = 0
# for i in range(300000):
#     if not i % 20:
#         print (i)
#     ind = np.random.randint(0, high=1500, size=batch_size)
#     batch = kappa[ind,:].reshape((-1,npix_side,npix_side))
#
#     #batch = [np.reshape(b, [28, 28]) for b in mnist.train.next_batch(batch_size=batch_size)[0]]
#     sess.run(optimizer, feed_dict = {X_in: batch, Y: batch, keep_prob: 0.8})
#
#     if not i % 200:
#     print ("Writing files...")
#         randoms = [np.random.normal(0, 1, n_latent) for _ in range(16)]
#         imgs = sess.run(dec, feed_dict = {sampled: randoms, keep_prob: 1.0})
#         imgs = [np.reshape(imgs[i], [npix_side, npix_side]) for i in range(len(imgs))] + [np.reshape(batch[i,:,:], [npix_side, npix_side]) for i in range(16)]
#         fig = plot(imgs)
#         plt.savefig('../data/out/{}.png'.format(str(im_i).zfill(3)), bbox_inches='tight')
#         plt.close(fig)
#         #fig = plot(batch[0:16,:,:])
#         #plt.savefig('../data/out/T{}.png'.format(str(im_i).zfill(3)), bbox_inches='tight')
#         #plt.close(fig)
#         im_i += 1
#         saver.save(sess,'../data/weights/VAE')
#
#
#         # ls, d, i_ls, d_ls, mu, sigm = sess.run([loss, dec, img_loss, latent_loss, mn, sd], feed_dict = {X_in: batch, Y: batch, keep_prob: 1.0})
#         # #plt.imshow(np.reshape(batch[0], [28, 28]), cmap='gray')
#         # #plt.show()
#         # #plt.imshow(d[0], cmap='gray')
#         # #plt.show()
#         # print(i, ls, np.mean(i_ls), np.mean(d_ls))
#         #
#
#
#
