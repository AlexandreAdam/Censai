import tensorflow as tf
import numpy as np

tf.enable_eager_execution()

execfile("definitions.py")

train_batch_size = 1
num_steps = 4
num_features = 512
state_size = 128

checkpoint_path_1 = "checkpoints/model1_512"
checkpoint_path_2 = "checkpoints/model2_512"



RESTORE=True


src_side = 20.48/2
im_side = 20.48
RIM = RIM_UNET_CELL(train_batch_size , num_steps , num_features , state_size)
sk_gen = SRC_KAPPA_Generator(train_batch_size=train_batch_size,test_batch_size=train_batch_size,kap_side_length=im_side, num_src_side=num_features,num_kappa_side=num_features,src_side=src_side)
lens_util_obj = lens_util(im_side= im_side, src_side=src_side, numpix_side = num_features ,kap_side=im_side,  method = "Unet")
physical_model = lens_util_obj.physical_model

noise_rms = 0.01
sk_gen.draw_k_s("test")

optimizer = tf.train.AdamOptimizer(1e-6)

if (RESTORE):
    RIM.model_1.load_weights(checkpoint_path_1)
    RIM.model_2.load_weights(checkpoint_path_2)

for train_iter in range(400000):
    #if ((train_iter%1)==0):
    print train_iter
    sk_gen.draw_k_s("train")
    noisy_data = lens_util_obj.simulate_noisy_lensed_image(  sk_gen.Source_tr[:,:,:,:],sk_gen.Kappa_tr[:,:,:,:],noise_rms)
    tf_source =  tf.identity(sk_gen.Source_tr[:,:,:,:])
    tf_kappa  = tf.identity(sk_gen.Kappa_tr[:,:,:,:] )

    with tf.GradientTape() as tape:
        tape.watch(RIM.model_1.variables)
        tape.watch(RIM.model_2.variables)
        cost_value, _ , _ , OS_src , OS_kap = RIM.cost_function(noisy_data, tf_source , tf_kappa)
    weight_grads = tape.gradient(cost_value, [RIM.model_1.variables , RIM.model_2.variables] )

    clipped_grads_1 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
    optimizer.apply_gradients(zip(clipped_grads_1, RIM.model_1.variables), global_step=tf.train.get_or_create_global_step())
    clipped_grads_2 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[1]]
    optimizer.apply_gradients(zip(clipped_grads_2, RIM.model_2.variables), global_step=tf.train.get_or_create_global_step())
    print( train_iter , cost_value.numpy() )
    if (((train_iter+1)%100)==0):
        RIM.model_1.save_weights(checkpoint_path_1)
        RIM.model_2.save_weights(checkpoint_path_2)
        print('saved weights.')
        
        
        
