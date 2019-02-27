import sys
sys.path = ['', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python27.zip', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/plat-linux2', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-tk', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-old', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/lib-dynload', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/site-packages', '/cm/shared/sw/pkg-old/devel/python2/2.7.13/lib/python2.7/site-packages/IPython/extensions', '/mnt/home/yhezaveh/.ipython'] + sys.path


import tensorflow as tf
import numpy as np
import scipy.io
from skimage.transform import resize

tf.enable_eager_execution()
# if (tf.executing_eagerly()==False):
    # tf.enable_eager_execution()

execfile("definitions.py")
execfile("genSIEdef_angles.py")


RESTORE = True
train_batch_size = 40
num_features = 512


RT = RayTracer()
kap_gen = SRC_KAPPA_Generator(train_batch_size=train_batch_size,kap_side_length=20.48,num_src_side=num_features,num_kappa_side=num_features)

#VAE_obj = VAE()
#VAE_checkpoint_path = "checkpoints/model_VAE"
#VAE_obj.load_weights(VAE_checkpoint_path)





optimizer = tf.train.AdamOptimizer(1.0e-7)
checkpoint_path = "checkpoints/model_Unet2"

if (RESTORE==True):
	RT.load_weights(checkpoint_path)

# Load from illustris galaxies and analytical deflection angles
#input_filename = '/mnt/home/yhezaveh/Censai/_matlab/data/old/KAP_001.mat'
#mat_contents = scipy.io.loadmat(input_filename)
#kappa = 2 * mat_contents['KAP']
#
#kappa = resize(kappa, (num_features,num_features), order=1, preserve_range=True)
#kappa = np.moveaxis( kappa , 2 , 0)
#
#x_alpha = np.zeros((train_batch_size,num_features, num_features,1))
#y_alpha = np.zeros((train_batch_size,num_features, num_features,1))

# To train on analytical SIEs
x_alpha = np.zeros((train_batch_size,num_features, num_features,1))
y_alpha = np.zeros((train_batch_size,num_features, num_features,1))
kappa = np.zeros((train_batch_size,num_features, num_features,1))

xim, yim = np.meshgrid( np.linspace(-10.24,10.24,512)*  np.pi / 180 / 3600, np.linspace(-10.24,10.24,512) * np.pi / 180 / 3600)

with tf.device('/gpu:0'):
    for epoch in range(1):
        for train_iter in range(1):
            
            # To train on analytical SIEs
            for i in range(train_batch_size):
                
                xlens = np.random.uniform(low=-1.0, high=1.)
                ylens = np.random.uniform(low=-1.0, high=1.)
                elp = np.random.uniform(low=0.01, high=0.6)
                phi = np.random.uniform(low=0.0, high=2.*np.pi)
                Rein = np.random.uniform(low=0.5, high = 7)
                x_alpha[i,:,:,0] , y_alpha[i,:,:,0]  = raytrace(xim,yim,[Rein, elp , phi*180/np.pi  , xlens , ylens])
                
                kappa[i,:,:,0]  = kap_gen.Kappa_fun(xlens, ylens, elp, phi, Rein)
            x_alpha = x_alpha*180/(np.pi)*3600
            y_alpha = y_alpha*180/(np.pi)*3600
            x_a_labels = tf.cast( x_alpha, datatype)
            y_a_labels = tf.cast( y_alpha, datatype)
            
            tf_kappa = tf.cast( kappa , datatype)
            
            # For the illustris data batches
#            ind = np.random.random_integers(low=0, high = 600)
#            batch_kappa = kappa[ind:ind+train_batch_size,:].reshape(-1,num_features,num_features,1)
#            tf_kappa = tf.cast( batch_kappa , datatype)
#            for i in range(train_batch_size):
#                x_alpha[i, :]=np.load('defangles/x_a_labels' + str(ind+i) + '.npy')
#                y_alpha[i, :]=np.load('defangles/y_a_labels' + str(ind+i) + '.npy')
#            x_a_labels = tf.cast( x_alpha , datatype)
#            y_a_labels = tf.cast( y_alpha , datatype)
                
        
        #sk_gen.draw_k_s("train")
        #tf_kappa  = tf.identity(sk_gen.Kappa_tr[:,:,:,:] )
        #tf_kappa  = tf.reshape(VAE_obj.draw_image(train_batch_size),(-1,256,256,1))
        #tf_kappa = tf.image.resize_images(tf_kappa,(512,512),preserve_aspect_ratio=True,align_corners=True)
        #_ , _ , x_a_labels , y_a_labels = lens_util_obj.get_deflection_angles(tf_kappa)


            with tf.GradientTape() as tape:
                tape.watch(RT.variables)
                cost_value , alpha_net = RT.cost_function(tf_kappa , x_a_labels , y_a_labels)
            weight_grads = tape.gradient(cost_value, [RT.variables] )

            clipped_grads = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
            optimizer.apply_gradients(zip(clipped_grads, RT.variables), global_step=tf.train.get_or_create_global_step())

            print( epoch, train_iter , cost_value.numpy() )
            
            if (((train_iter+1)%10)==0):
                print( "saving weights." )
                RT.save_weights("checkpoints/model_Unet2")
            
            
            
            np.save('kappa.npy', tf_kappa)
            np.save('x_a_analytic.npy', x_a_labels)
            np.save('y_a_analytic.npy', y_a_labels)
##            np.save('x_a_raytraced.npy', x_a_raytraced)
##            np.save('y_a_raytraced.npy', y_a_raytraced)
            np.save('a_net.npy', alpha_net)
            
            

