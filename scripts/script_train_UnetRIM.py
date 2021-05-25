import sys
sys.path = ['',
 '/cm/shared/sw/pkg-old/devel/python2/2.7.13/bin',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python27.zip',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/plat-linux2',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/lib-tk',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/lib-old',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/lib-dynload',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/site-packages',
 '/mnt/xfs1/flatiron-sw/pkg/devel/python2/2.7.13/lib/python2.7/site-packages/IPython/extensions',
 '/mnt/home/llevasseur/.ipython']
 

import tensorflow as tf
import numpy as np

tf.enable_eager_execution()



train_batch_size = 1
num_steps = 10
num_features = 512
state_size = 128

load_checkpoint_path_1 = "checkpoints/model1_512"
load_checkpoint_path_2 = "checkpoints/model2_512"
RESTORE=False
save_checkpoint_path_1 = "checkpoints/model1_512_meanstart_N10"
save_checkpoint_path_2 = "checkpoints/model2_512_meanstart_N10"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from IPython import display
import pylab as pl


execfile("definitions.py")

src_side = 20.48/2
im_side = 20.48

with tf.device('/gpu:0'):
    sk_gen = SRC_KAPPA_Generator(train_batch_size=train_batch_size,test_batch_size=train_batch_size,kap_side_length=im_side, num_src_side=num_features,num_kappa_side=num_features,src_side=src_side)
    src_in , kap_in = sk_gen.draw_average_k_s()
    #src_in = src_in * 0.
    kap_in = np.log10(kap_in)
    RIM = RIM_UNET_CELL(train_batch_size , num_steps , num_features , state_size , cond1 = src_in , cond2 = kap_in)
    lens_util_obj = lens_util(im_side= im_side, src_side=src_side, numpix_side = num_features ,kap_side=im_side,  method = "Unet")

    physical_model = lens_util_obj.physical_model

    noise_rms = 0.01
    sk_gen.draw_k_s("test")

    
    if (RESTORE):
        RIM.model_1.load_weights(load_checkpoint_path_1)
        RIM.model_2.load_weights(load_checkpoint_path_2)

    optimizer = tf.train.AdamOptimizer(1e-4)



    for train_iter in range(4000000):
        #if ((train_iter%1)==0):
        print train_iter
        sk_gen.draw_k_s("train")
        noisy_data = lens_util_obj.simulate_noisy_lensed_image(  sk_gen.Source_tr[:,:,:,:],np.log10(sk_gen.Kappa_tr[:,:,:,:]),noise_rms)
        tf_source =  tf.identity(sk_gen.Source_tr[:,:,:,:])
        tf_logkappa  = log10(tf.identity(sk_gen.Kappa_tr[:,:,:,:]) )

        with tf.GradientTape() as tape:
            tape.watch(RIM.model_1.variables)
            tape.watch(RIM.model_2.variables)
            cost_value, _ , _ , OS_src , OS_kap = RIM.cost_function(noisy_data, tf_source , tf_logkappa)
        weight_grads = tape.gradient(cost_value, [RIM.model_1.variables , RIM.model_2.variables] )

        clipped_grads_1 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
        optimizer.apply_gradients(zip(clipped_grads_1, RIM.model_1.variables), global_step=tf.train.get_or_create_global_step())
        clipped_grads_2 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[1]]
        optimizer.apply_gradients(zip(clipped_grads_2, RIM.model_2.variables), global_step=tf.train.get_or_create_global_step())
        print( train_iter , cost_value.numpy() )
        if (((train_iter+1)%100)==0):
            model_im = lens_util_obj.physical_model(OS_src , OS_kap)
            ims = [OS_src[0,:,:,0] , OS_kap[0,:,:,0] , tf_source[0,:,:,0] , tf_logkappa[0,:,:,0] , noisy_data[0,:,:,0] , model_im.numpy()[0,:,:,0] ]
            fig = plot(ims)
            plt.savefig('output_images_5/{}.png'.format(str(train_iter).zfill(3)), bbox_inches='tight')
            plt.close(fig)            
            # pl.clf()
            # fig, ax = pl.subplots(3, 2, figsize = (10, 15))
            # pl.subplots_adjust(wspace = 0, hspace = 0)
            # imsrcP = ax[0, 0].imshow(OS_src[0,:,:,0])
            # immodP = ax[0, 1].imshow(np.log10(OS_kap[0,:,:,0]))
            # imsrcT = ax[1, 0].imshow(tf_source[0,:,:,0])
            # imsrcT = ax[1, 1].imshow(np.log10(tf_kappa[0,:,:,0]))
            # immodT = ax[2, 0].imshow(noisy_data[0,:,:,0])
            # immodP = ax[2, 1].imshow(model_im.numpy()[0,:,:,0])
            # for j in range(2):
            #     for i in range(3):
            #         ax[i,j].axis('off')
            # display.clear_output(wait=True)
            # display.display(pl.gcf())
            print( train_iter , cost_value.numpy() )
        if (((train_iter+1)%100)==0):
            RIM.model_1.save_weights(save_checkpoint_path_1)
            RIM.model_2.save_weights(save_checkpoint_path_2)
            print('saved weights.')

            test
