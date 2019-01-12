'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''


from time import gmtime, strftime
from os import path
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import rnn as rnn_cell

import iterative_inference_learning.layers.iterative_estimation as iterative_estimation
import iterative_inference_learning.layers.loopfun as loopfun
#import iel_experiments.data.bsds300 as bsds
#from iel_experiments.data.data_helpers import get_image_patches, make_batches
from iel_experiments.models import decorate_rnn,conv_rnn
#from iel_experiments.utils.corruption_operators import MatlabBicubicDownsampling

import Censai as Celi

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '~/Documents/tensorflow_logs/iel/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_boolean('use_rnn', True,
                            """Flag whether to use an RNN or whether to use a ReLu network.""")
tf.app.flags.DEFINE_boolean('use_gray', True,
                            """Flag whether we work on gray images or RGB images.""")
tf.app.flags.DEFINE_boolean('use_grad', True,
                            """Flag whether to use gradient information in the inference model""")
tf.app.flags.DEFINE_boolean('use_prior', True,
                            """Flag whether to input the current estimate again.""")
tf.app.flags.DEFINE_boolean('accumulate_output', True,
                            """Flag whether some teh network outputs over time.""")
tf.app.flags.DEFINE_float('lr_kap', 1.0e-6,
                            """Global learning rate to use""")
tf.app.flags.DEFINE_float('lr_src', 3.0e-7,
                            """Global learning rate to use""")
tf.app.flags.DEFINE_integer('n_pseudo', 1,
                            """How many pseudo samples should be used""")
tf.app.flags.DEFINE_integer('n_epochs', 1,
                            """Number of epochs.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('k_size', 11,
                            """Size of convolutional kernels.""")
tf.app.flags.DEFINE_integer('features', 32,
                            """Number of features per layer.""")
tf.app.flags.DEFINE_integer('depth', 1,
                            """Depth of the network""")
tf.app.flags.DEFINE_integer('batch_size', 5,
                            """Number of samples per batch.""")
tf.app.flags.DEFINE_integer('t_max', 10,
                            """The number of time steps to train on. """
                            """If -1 it will be drawn randomly from a geometrix distribution.""")
tf.app.flags.DEFINE_integer('j_min', 6,
                            """The minimum id of corruption models to sample from""")
tf.app.flags.DEFINE_integer('j_max', 7,
                            """The maximum id of corruption models to sample from""")
tf.app.flags.DEFINE_integer('k_max', 5,
                            """The maximum size x of corruption kernels for BlurImage, and pooling size """
                            """ for resize operations """)
tf.app.flags.DEFINE_integer('k_in', 1,
                            """The number of input channels of corruption kernels for BlurImage""")
tf.app.flags.DEFINE_integer('k_out', 1,
                            """The number of output channels corruption kernels for BlurImage, and pooling size""")
tf.app.flags.DEFINE_integer('stride', 1,
                            """The stride for the ImageBlur operation""")
tf.app.flags.DEFINE_float('noise_lambda', 100.,
                            """The scale parameter for the exponential distribution where """
                            """noise_std is drawn from.""")
tf.app.flags.DEFINE_float('reduction_min', 0.,
                            """The minimum allowable fraction of sample size fro RP""")
tf.app.flags.DEFINE_float('reduction_max', 1.,
                            """The maximum allowable fraction of sample size fro RP""")
tf.app.flags.DEFINE_string('desc', 'default_superres_model_',
                           """A short description of the experiment""")


def my_tf_log10(x):
        return tf.log(x)/tf.log(tf.constant(10, dtype=tf.float32))

def get_psnr(x_est, x_true):
    rmse = tf.sqrt(tf.reduce_mean(tf.square(x_true - x_est),[-3,-2,-1]))
    psnr = 20. * (- my_tf_log10(rmse))

    return psnr

def train():

    # This is the file that we will save the model to.
    model_name = os.environ['CENSAI_PATH']+ '/trained_weights/RIM_kappa-source/Censai_lowres_trueFIXEDsrc_Reinkap_nolog.ckpt'

    
    # DEFINE LAURENCE's stuff
    numpix_side = 48
    numpix_src  = 48
    numkappa_side = 48
    
    batch_size = 20
    test_batch_size = 10
    n_channel = 1
    
    Raytracer = Celi.Likelihood(numpix_side = numpix_side)
    Datagen = Celi.DataGenerator(numpix_side=Raytracer.numpix_side, numkappa_side=numkappa_side, src_side=Raytracer.src_side, im_side = Raytracer.im_side,max_noise_rms=0.0,use_psf=False,lens_model_error=[0.01,0.01,0.01,0.01,0.01,0.01,0.01],binpix=1,mask=False,min_unmasked_flux=1.0)
    
    
    # Numpy arrays to read data
    Datagen.X = np.zeros((batch_size, Datagen.numpix_side , Datagen.numpix_side,1 ))
    Datagen.source = np.zeros((batch_size, Datagen.numpix_side , Datagen.numpix_side,1 ))
    Datagen.kappa = np.zeros((batch_size, Datagen.numkappa_side , Datagen.numkappa_side,1 ))
    max_file_num=None
    train_or_test = 'train'
    read_or_gen = 'gen'

    # Placeholders
    
    Kappatest = tf.placeholder( tf.float32, [None, Datagen.numkappa_side, Datagen.numkappa_side,1] )
    Srctest = tf.placeholder( tf.float32, [None, Raytracer.numpix_side, Raytracer.numpix_side,1] )



    y_image1 = tf.identity(Kappatest)
    y_image2 = tf.identity(Srctest)
    #y_1 = tf.reshape(y_image1, [-1,Datagen.numkappa_side**2])
    #y_2 = tf.reshape(y_image2, [-1,Datagen.numpix_side**2])
    x_init1 = tf.zeros_like(y_image1)
    x_init2 = tf.zeros_like(y_image2)

    Raytracer.trueimage = Raytracer.get_lensed_image(Kappatest,[0.,0.], 7.68, Srctest)
    x_image = Raytracer.trueimage
    
        

    # Needed for Optimization purposes
    global_step = tf.Variable(0, trainable=False, dtype=tf.int32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # Number of steps to perform for inference
    T = tf.constant(FLAGS.t_max, dtype=tf.int32, name='T')

    ## Define some helper functions
    def param2image(x_param):
        tens = tf.constant(10.0)
        x_temp =  x_param#tf.pow(tens, x_param) #tf.nn.sigmoid(x_param)
        #x_temp = tf.nn.leaky_relu(x_param) #tf.nn.sigmoid(x_param)
        return x_temp

    def param2image_src(x_param):
        #x_temp = tf.nn.sigmoid( 5.0 * (x_param-0.5))
        x_temp = tf.nn.leaky_relu(x_param)
        #x_temp = x_param
        return x_temp


    def image2param_src(x):
        x_temp = x#tf.log(x) - tf.log(1 - x)
        return x_temp

#    def param2grad(x_param):
#        x_temp = ones_like(x_param) #tf.nn.sigmoid(x_param) * (1. - tf.nn.sigmoid(x_param))
#        return x_temp
#    
    def error_grad1(x_test , the_other):
        return tf.gradients(Raytracer.Loglikelihood( param2image_src(the_other) , param2image(x_test), [0.,0.], 7.68), x_test)[0]

    def error_grad1_true_src(x_test , the_other):
        return tf.gradients(Raytracer.Loglikelihood( Srctest, param2image(x_test), [0.,0.], 7.68), x_test)[0]

    def error_grad2(x_test , the_other):
        return tf.gradients(Raytracer.Loglikelihood( param2image_src(x_test) , param2image(the_other), [0.,0.], 7.68), x_test)[0]

    def error_grad2_true_kappa(x_test , the_other):
        return tf.gradients(Raytracer.Loglikelihood( param2image_src(x_test) , Kappatest , [0.,0.], 7.68), x_test)[0]

    def redundant_identity(x_test , the_other):
        return tf.identity(x_test)

    def lossfun_1(x_est1,expand_dim=False):
        temp_data1 = y_image1
	#tens = tf.constant(10.0)
	#temp_data1 = tf.pow(tens, temp_data1)
        if expand_dim:
            print('DIMS ARE ... ')
            print(temp_data1.shape)
            temp_data1 = tf.expand_dims(temp_data1,0)
            print(temp_data1.shape)
        return tf.reduce_mean(0.5 * tf.square(x_est1 - temp_data1) , [-3,-2,-1] ) 

    def lossfun_2(x_est2,expand_dim=False):
        temp_data2 = y_image2
        if expand_dim:
            temp_data2 = tf.expand_dims(temp_data2,0)
        return tf.reduce_mean(0.5 * tf.square(x_est2 - temp_data2) , [-3,-2,-1] ) 

    def lossfun(x_est1,x_est2,expand_dim=False):
        temp_data1 = y_image1
        temp_data2 = y_image2
        if expand_dim:
            temp_data1 = tf.expand_dims(temp_data1,0)
            temp_data2 = tf.expand_dims(temp_data2,0)
        return tf.reduce_mean(0.5 * tf.square(x_est1 - temp_data1) , [-3,-2,-1] ) + tf.reduce_sum(0.5 * tf.square(x_est2 - temp_data2) , [-3,-2,-1] )
    ## End helper functions


    ## Setup RNN
    if FLAGS.use_rnn:
        print "Using RNN"
        cell1, cell2 , output_func1 , output_func2 = conv_rnn.gru(FLAGS.k_size, [FLAGS.features]*FLAGS.depth, is_training=is_training)

    ## Defines how the output dimensions are handled
    output_shape_dict = {'mu':n_channel}
    if FLAGS.n_pseudo > 0:
        output_shape_dict.update({'pseudo': n_channel*FLAGS.n_pseudo})

    output_transform_dict_1 = {}
    output_transform_dict_2 = {}
    if FLAGS.use_prior:
        output_transform_dict_1.update({'all':[redundant_identity]})
        output_transform_dict_2.update({'all':[redundant_identity]})

    if FLAGS.use_grad:
        output_transform_dict_1.update({'mu': [error_grad1_true_src]})
        output_transform_dict_2.update({'mu': [error_grad2]})
        if FLAGS.n_pseudo > 0:
            output_transform_dict_1.update({'pseudo':[loopfun.ApplySplitFunction(error_grad1_true_src, 4 - 1, FLAGS.n_pseudo)]})
            output_transform_dict_2.update({'pseudo':[loopfun.ApplySplitFunction(error_grad2, 4 - 1, FLAGS.n_pseudo)]})

    
    input_func1, output_func1, init_func1, output_wrapper1 = decorate_rnn.init(rank=4, output_shape_dict=output_shape_dict,
                                                                  output_transform_dict=output_transform_dict_1,
                                                                  init_name='mu', ofunc = output_func1,
                                                                  accumulate_output=FLAGS.accumulate_output)
    
    input_func2, output_func2, init_func2, output_wrapper2 = decorate_rnn.init(rank=4, output_shape_dict=output_shape_dict,
                                                                  output_transform_dict=output_transform_dict_2,
                                                                  init_name='mu', ofunc = output_func2,
                                                                  accumulate_output=FLAGS.accumulate_output)

    ## This runs the inference
    x_init_feed1 = tf.maximum(tf.minimum( x_init1 , 1.-1e-4), 1e-4) 
    x_init_feed2 =   tf.maximum(tf.minimum( x_init2 , 1.-1e-4), 1e-4) 
    
    
    #print x_init_feed , cell , input_func , output_func , init_func , T

    alltime_output1, alltime_output2, final_output1, final_output2, final_state1, final_state2, p_t, T_ = \
        iterative_estimation.function(x_init_feed1,x_init_feed2, cell1, cell2, input_func1, input_func2, output_func1, output_func2, init_func1, init_func2, T=T)

    
    final_state1 = tf.identity(final_state1, name='final_state1')
    final_state2 = tf.identity(final_state2, name='final_state2')
    p_t = tf.identity(p_t, 'p_t')
    T_ = tf.identity(T_, 'T_')

    #alltime_output2 = param2image_src(alltime_output2)
    #final_output2 = param2image_src(final_output2)

    alltime_output1 = output_wrapper1(alltime_output1, 'mu', 4)
    alltime_output2 = param2image_src( output_wrapper1(alltime_output2, 'mu', 4) )
    #alltime_output2 = tf.layers.conv2d(alltime_output2 , 1 , 1 , activation=tf.nn.relu)
    #alltime_output2_sig = tf.nn.sigmoid( output_wrapper1(alltime_output2, 'mu', 4) )
    final_output1 = output_wrapper2(final_output1, 'mu')
    final_output2 = output_wrapper2(final_output2, 'mu') 

    alltime_output1 = tf.identity(alltime_output1,name='alltime_output1')
    alltime_output2 = tf.identity(alltime_output2,name='alltime_output2')
    final_output1 = tf.identity(final_output1, name='final_output1')
    final_output2 = tf.identity(final_output2, name='final_output2')

    #final_gradient_1 = error_grad1(final_output1 , final_output2 )
    #final_gradient_2 = error_grad2(final_output2 , final_output1 )
    
    #tf.add_to_collection('output1', alltime_output1)
    #tf.add_to_collection('output1', final_output1)
    #tf.add_to_collection('output2', alltime_output2)
    #tf.add_to_collection('output2', final_output2)

    ## Define loss functions
    loss_full_1 = tf.reduce_sum(tf.reduce_mean(p_t * lossfun_1(alltime_output1, True), reduction_indices=[1]))
    loss_full_2 = tf.reduce_sum(tf.reduce_mean(p_t * lossfun_2(alltime_output2, True), reduction_indices=[1]))
    loss_full = tf.reduce_sum(tf.reduce_mean(p_t * lossfun(alltime_output1,alltime_output2, True), reduction_indices=[1]))
    #loss_1 = tf.reduce_mean(lossfun_1(final_output1))
    #loss_2 = tf.reduce_mean(lossfun_2(final_output2))
    #loss = tf.reduce_mean(lossfun(final_output1,final_output2))
    #tf.add_to_collection('losses', loss_full)
    #tf.add_to_collection('losses', loss)

    #psnr = tf.reduce_mean(get_psnr(final_output1, y_image1))
    #psnr_x_init = tf.reduce_mean(get_psnr(x_init1, y_image1))
    #tf.add_to_collection('psnr', psnr)
    #tf.add_to_collection('psnr', psnr_x_init)

    ## Minimizer
    minimize_1 = tf.contrib.layers.optimize_loss(loss_full_1, global_step, FLAGS.lr_kap, "Adam", clip_gradients=1.0,
                                               learning_rate_decay_fn=lambda lr,s: tf.train.exponential_decay(lr, s,
                                               decay_steps=5000, decay_rate=0.96, staircase=True))

    minimize_2 = tf.contrib.layers.optimize_loss(loss_full_2, global_step, FLAGS.lr_src, "Adam",  clip_gradients=1.0,
                                               learning_rate_decay_fn=lambda lr,s: tf.train.exponential_decay(lr, s,
                                               decay_steps=5000, decay_rate=0.96, staircase=True))

    #minimize = tf.contrib.layers.optimize_loss(loss_full, global_step, FLAGS.lr, "Adam", clip_gradients=1.0,
    #                                           learning_rate_decay_fn=lambda lr,s: tf.train.exponential_decay(lr, s,
    #                                           decay_steps=5000, decay_rate=0.96, staircase=True))


    # Initializing the variables
    init_op = tf.global_variables_initializer()

    # Create a summary to monitor cost function
    #loss_summary = tf.summary.scalar("Loss", loss)
    #mse_summary = tf.summary.scalar("PSNR", psnr)

    # Merge all summaries to a single operator
    merged_summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=None)

    # Launch the graph
    with tf.Session() as sess:
#        fisrttime = 1;
        #writer = tf.summary.FileWriter("log/", sess.graph)
        sess.run(init_op)
        # Keep training until reach max iterations

        # Restore session
#        all_vars = tf.global_variables()
#        Adam_pars = []
#        for tvar in all_vars:
#            if "Adam" in tvar.op.name:
#                Adam_pars = Adam_pars + [tvar]
#        vars_to_save = set(tf.global_variables()) - set(Adam_pars)
#        restorer = tf.train.Saver(vars_to_save,  max_to_keep=None)
        
        
#        restorer.restore(sess,model_name)
        #saver.restore(sess,model_name)
        min_test_cost = 0.153253
        # Set logs writer into folder /tmp/tensorflow_logs

	    # Generate test set
        Datagen.Xtest = np.zeros((test_batch_size, Datagen.numpix_side , Datagen.numpix_side,n_channel  ))
        Datagen.sourcetest = np.zeros((test_batch_size, Datagen.numpix_side , Datagen.numpix_side,n_channel  ))
        Datagen.kappatest = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side,n_channel  )) 
        
        Datagen.sourcetest, Datagen.kappatest = Datagen.read_data_batch_fixedsrc(Datagen.Xtest, Datagen.sourcetest, Datagen.kappatest, 'test', 'gen')
        imgs = np.zeros((11,test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))

        imgs_1 = np.zeros((11,test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        imgs_2 = np.zeros((11,test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        true_data = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        #pred_lens_image_1 = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        #pred_lens_image_2 = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        last_grad_1 = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))
        last_grad_2 = np.zeros((test_batch_size, Datagen.numkappa_side , Datagen.numkappa_side, n_channel ))

	print('dims of alltime_output1 are')
	print(alltime_output1.shape)

        for epoch in range(10):
            train_cost = 0.
            train_psnr = 0.

            print "Sampling data"
            #train_batches = make_batches(train_x.shape[0], FLAGS.batch_size)

            print "Iterating..."
            # Loop over all batches
            #Datagen.read_data_batch(Datagen.X ,Datagen.source, Datagen.kappa , train_or_test, read_or_gen)
            for i in range(10000):
                Datagen.read_data_batch_fixedsrc(Datagen.X ,Datagen.source, Datagen.kappa , train_or_test, read_or_gen)
                #print 'generated data batch', i
                #dataprocessor.load_data_batch(100000,'train')
                
                # Fit training using batch data
                #print Datagen.source.shape
                #print Datagen.kappa.shape
                temp_cost_1 = 0
                temp_cost_2 = 0
                if (np.random.uniform() < 1.0):
                    temp_cost_1,_  = sess.run( [ loss_full_1 , minimize_1 ] ,   {Srctest: Datagen.source, Kappatest: Datagen.kappa,is_training:True})
                    #temp_cost_2,_  = sess.run( [ loss_full_2 , minimize_2 ] ,   {Srctest: Datagen.source, Kappatest: #Datagen.kappa,is_training:True})
                else:
                    #temp_cost_2,_ , AL1 , AL2= sess.run( [ loss_full_2 , minimize_2 , alltime_output1 , alltime_output2] ,   {Srctest: Datagen.source, Kappatest: Datagen.kappa,is_training:True})
                    temp_cost_1,_ , AL1 , AL2= sess.run( [ loss_full_1 , minimize_1 , alltime_output1 , alltime_output2] ,   {Srctest: Datagen.source, Kappatest: Datagen.kappa,is_training:True})

                temp_cost = temp_cost_1 + temp_cost_2
                #temp_cost, summary_str,_ = sess.run([loss,merged_summary_op,minimize],   {Srctest: Datagen.source, Kappatest: Datagen.kappa,is_training:True})
                print i, temp_cost_1, temp_cost_2 , temp_cost
                # Compute average loss
                train_cost += temp_cost
                #train_psnr += temp_psnr

                #np.save('kappa_true_fangle' + str(i) + '.npy', Datagen.kappa )
                #np.save('source_true_fangle_train' + str(i) + '.npy', Datagen.source )
                #np.save('kappa_rec_train' + str(i) + '.npy', AL1)
                #np.save('source_rec_train' + str(i) + '.npy', AL2)
                #np.save('TMP_S' + str(i) + '.npy' , TMP_S)

                if (i+1) % 100 == 0:
                    valid_cost = 0.
                    Ttemp_cost_1 = 0.
                    #valid_psnr = 0.
# #
                    for j in range(10):
                        dpm = 1
                        #temp_cost, temp_psnr= sess.run([loss,psnr], {Srctest: Datagen.sourcetest[dpm*j:dpm*(j+1),:], Kappatest: Datagen.kappatest[dpm*j:dpm*(j+1),:],is_training:False})
                        temp_cost_1 , temp_cost_2 , imgs_1[1:,dpm*j:dpm*(j+1),:], imgs_2[1:,dpm*j:dpm*(j+1),:] , true_data[dpm*j:dpm*(j+1),:]  = sess.run([ loss_full_1 , loss_full_2 , alltime_output1,alltime_output2, Raytracer.trueimage ], {Srctest: Datagen.sourcetest[dpm*j:dpm*(j+1),:], Kappatest: Datagen.kappatest[dpm*j:dpm*(j+1),:],is_training:False})
                        # Compute average loss
                        valid_cost += (temp_cost_1 + temp_cost_2)
                        Ttemp_cost_1 += temp_cost_1
                        #valid_psnr += temp_psnr
                        print 'testcost', i, (temp_cost_1 + temp_cost_2)

                    valid_cost /= 10.
                    Ttemp_cost_1 /= 10.
                     #valid_psnr /= 10.
# #
# #                    # Display logs per epoch step
                    print "Epoch:", '%04d' % (epoch+1), "batch:", '%04d' % (i+1)
                    print "cost=", "{:.9f}".format(train_cost/(i+1))
                    #print "psnr=", "{:.9f}".format(train_psnr/(i+1))
                    print "test cost=", "{:.9f}".format(valid_cost)
                    #print "test psnr=", "{:.9f}".format(valid_psnr)
                    #print 'saving numpy arrays ...'
                    #np.save('last_grad_1_fangle.npy', last_grad_1)
                    #np.save('last_grad_2_fangle.npy', last_grad_2)
                    #np.save('pred_lens_image_fangle.npy', pred_lens_image)
                    #np.save('true_data_fangle.npy', true_data)
                    if (1==0):
                        np.save('source_image_fangle.npy', Datagen.sourcetest )
                        np.save('source_image_fangle_train.npy', Datagen.source )
                        np.save('kappa_rec_fangle_1.npy', imgs_1)
                     	np.save('kappa_rec_fangle_2.npy', imgs_2)
                     	np.save('kappa_map_fangle.npy', Datagen.kappatest)

#                    
#
                    # Saving Checkpoint
                    print 'are we saving?', Ttemp_cost_1, min_test_cost
                    if Ttemp_cost_1 < min_test_cost:
                        print "Saving Checkpoint"
#                        if (fisrttime==1):
#                            all_vars = tf.global_variables()
#                            Adam_pars = []
#                            for tvar in all_vars:
#                                if "Adam" in tvar.op.name:
#                                    Adam_pars = Adam_pars + [tvar]
#                                    #print(tvar.op.name)
#                            vars_to_save = set(tf.global_variables()) - set(Adam_pars)
                            #print(tf.global_variables())
#
#                            saver = tf.train.Saver(vars_to_save,  max_to_keep=None)
#                            fisrttime=0
                        
                        saver.save(sess,os.environ['CENSAI_PATH']+ '/trained_weights/RIM_kappa-source/Censai_lowres_trueFIXEDsrc_Reinkap_nolog.ckpt')
                        min_test_cost = Ttemp_cost_1 * 1.

        print "Optimization Finished!"

    sess.close()

# def main(argv=None):  # pylint: disable=unused-argument
FLAGS.train_dir = path.expanduser(FLAGS.train_dir)
tf.gfile.MakeDirs(FLAGS.train_dir)
train()


# if __name__ == '__main__':
#     tf.app.run()
