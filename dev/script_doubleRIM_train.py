import tensorflow as tf
import numpy as np

tf.enable_eager_execution()


execfile("definitions.py")

train_batch_size = 4
num_steps = 10
num_features = 512
state_size = 128

checkpoint_path_1 = "checkpoints/model1_512"
checkpoint_path_2 = "checkpoints/model2_512"



RESTORE=True



with tf.device('/gpu:0'):

	RIM = RIM_CELL(train_batch_size , num_steps , num_features , state_size)



	sk_gen = SRC_KAPPA_Generator(train_batch_size=train_batch_size,test_batch_size=train_batch_size,kap_side_length=20.48, num_src_side=num_features,num_kappa_side=num_features)
	lens_util_obj = lens_util(im_side= 20.48, numpix_side = num_features ,kap_side=20.48,  method = "Unet")

	physical_model = lens_util_obj.physical_model



	optimizer = tf.train.AdamOptimizer(1e-3)
    
    
	if (RESTORE==True):
		RIM.source_model.load_weights(checkpoint_path_1)
		RIM.kappa_model.load_weights(checkpoint_path_2)


	noise_rms = 0.01
	sk_gen.draw_k_s("test")

	for train_iter in range(1):
	    #if ((train_iter%1)==0):
	    #	print train_iter
	    sk_gen.draw_k_s("train")
	    noisy_data = lens_util_obj.simulate_noisy_lensed_image(  sk_gen.Source_tr[:,:,:,:],sk_gen.Kappa_tr[:,:,:,:],noise_rms)
	    tf_source =  tf.identity(sk_gen.Source_tr[:,:,:,:])
	    tf_kappa  = tf.identity(sk_gen.Kappa_tr[:,:,:,:] )

	    with tf.GradientTape() as tape:
	        tape.watch(RIM.source_model.variables)
	        tape.watch(RIM.kappa_model.variables)
	        cost_value, os1, os2 = RIM.cost_function(noisy_data, tf_source , tf_kappa)
	    weight_grads = tape.gradient(cost_value, [RIM.source_model.variables , RIM.kappa_model.variables])

	    clipped_grads_1 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[0]]
	    optimizer.apply_gradients(zip(clipped_grads_1, RIM.source_model.variables), global_step=tf.train.get_or_create_global_step())
	    clipped_grads_2 = [tf.clip_by_value(grads_i,-10,10) for grads_i in weight_grads[1]]
	    optimizer.apply_gradients(zip(clipped_grads_2, RIM.kappa_model.variables), global_step=tf.train.get_or_create_global_step())
	    print( train_iter , cost_value.numpy() )
        np.save('noisy_data.npy',noisy_data.numpy() )
        np.save('source.npy',tf_source.numpy() )
        np.save('kappa.npy',tf_kappa.numpy() )
        np.save('os1.npy', os1[-1].numpy())
        np.save('os2.npy', os2[-1].numpy())
        if (((train_iter+1)%100)==0):
                RIM.source_model.save_weights(checkpoint_path_1)
                RIM.kappa_model.save_weights(checkpoint_path_2)
                print('saved weights.')

#filename_1 = "checkpoints/model_1_"
#filename_2 = "checkpoints/model_2_"
#saver1 = tf.train.Checkpoint(RIM.model_1=RIM.model_1)
#saver1.save(filename_1)
#saver2 = tf.train.Checkpoint(model=RIM.model_2)
#saver2.save(filename_2)

#saver1.restore(tf.train.latest_checkpoint(filename_1))
#saver2.restore(tf.train.latest_checkpoint(filename_2))

'''



with tf.device('/gpu:0'):
	start = time.time();
	xx,yy = lens_util_obj.get_deflection_angles(tf.reshape(tf_kappa[0,:,:,:],(1,192,192,1)) ); 
	end = time.time(); 
	print end - start
'''
