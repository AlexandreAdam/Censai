from PIL import Image
import tensorflow as tf
import scipy.ndimage
from scipy import misc
from scipy.interpolate import RectBivariateSpline
import numpy as np
import numpy.matlib as ml
import random
import time
import os
#from spatial_transformer import transformer
import gc
import scipy.io





class DataGenerator(object):
    '''
    A class to handle processing of data.
    '''
    
    def __init__(self,datadir=None,numpix_side=192, numkappa_side=193, src_side=2., im_side = 2.,max_noise_rms=0.0,use_psf=False,lens_model_error=[0.01,0.01,0.01,0.01,0.01,0.01,0.01],binpix=1,mask=False,min_unmasked_flux=1.0):
        '''
        Initialize an instance of the class.  Give it the directory
        of the directories containing training/test data.
        '''
        self.datadir = datadir
        if datadir is not None:
            self.num_datadir = len(datadir)
        self.src_side = src_side
        self.numpix_side = numpix_side
        self.im_side = im_side
        self.numkappa_side = numkappa_side
        
        self.num_out = 7
        
        
        
    def gen_source(self,Xsrc, Ysrc, x_src = 0, y_src = 0, sigma_src = 1, numpix_side = 192):
    
        
    
        Im = np.sqrt(((Xsrc-x_src)**2+(Ysrc-y_src)**2) / (2.*sigma_src**2) )
    
        return Im



    def Kappa_fun(self, xlens, ylens, elp, phi, sigma_v, numkappa_side = 193, kap_side_length = 2, rc=0, Ds = 1753486987.8422, Dds = 1125770220.58881, c = 299800000):
    
        x = np.linspace(-1, 1, numkappa_side) * kap_side_length/2
        y = np.linspace(-1, 1, numkappa_side) * kap_side_length/2
        xv, yv = np.meshgrid(x, y)
        
        A = (y[1]-y[0])/2. *(2*np.pi/ (360*3600) )
        
        rcord, thetacord = np.sqrt(xv**2 + yv**2) , np.arctan2(xv, yv)
        thetacord = thetacord - phi
        Xkap, Ykap = rcord*np.cos(thetacord), rcord*np.sin(thetacord)
        
        rlens, thetalens = np.sqrt(xlens**2 + ylens**2) , np.arctan2(xlens, ylens)
        thetalens = thetalens - phi
        xlens, ylens = rlens*np.cos(thetalens), rlens*np.sin(thetalens)
    
        r = np.sqrt((Xkap-xlens)**2 + ((Ykap-ylens) * (1-elp) )**2) *(2*np.pi/ (360*3600) )
    
        Rein = (4*np.pi*sigma_v**2/c**2) * Dds /Ds 
    
        kappa = np.divide( np.sqrt(1-elp)* Rein ,  (2* np.sqrt( r**2 + rc**2)))
    
        mass_inside_00_pix = 2.*A*(np.log(2.**(1./2.) + 1.) - np.log(2.**(1./2.)*A - A) + np.log(3.*A + 2.*2.**(1./2.)*A))
    
        
    
        density_00_pix = np.sqrt(1.-elp) * Rein/(2.) * mass_inside_00_pix/((2.*A)**2.)
    
    
    
        ind = np.argmin(r)
    
        kappa.flat[ind] = density_00_pix
    
        return kappa
    
    
    def read_data_batch( X , Y , train_or_test, read_or_gen,  max_file_num=None):
    
        batch_size = len(X)
        #mag = np.zeros((batch_size,1))

        if read_or_gen == 'read':
    #        if train_or_test=='test':
    #            #inds = range(batch_size)
    #            np.random.seed(seed=136)# 136 ->arc_1, 137 -> arc_2
    #            d_path = [[],[]]
    #            d_path[0] = test_data_path_1
    #            d_path[1] = test_data_path_2
    #
    #            #d_lens_path = [[],[],[]]  
    #            #d_lens_path[0] = testlens_data_path_1
    #            #d_lens_path[1] = testlens_data_path_2
    #            #d_lens_path[2] = testlens_data_path_3
    #    #        inds = np.random.randint(0, high = max_file_num , size= batch_size)
    #            inds = range(max_file_num)
    #        else:
            np.random.seed(seed=None)
    #            inds = np.random.randint(0, high = max_file_num , size= batch_size)
    #
    #            d_path = [[],[]]
    #            d_path[0] = arcs_data_path_1
    #            d_path[1] = arcs_data_path_2
    #
    #            #d_lens_path = [[],[],[]]
    #            #d_lens_path[0] = lens_data_path_1
    #            #d_lens_path[1] = lens_data_path_2
    #            #d_lens_path[2] = lens_data_path_3

            #inds = np.zeros((batch_size,),dtype='int')
        else:
            np.random.seed(seed=136)
            x = np.linspace(-1, 1, self.numpix_side) * self.src_side/2
            y = np.linspace(-1, 1, self.numpix_side) * self.src_side/2
            Xsrc, Ysrc = np.meshgrid(x, y)

            for i in range(batch_size):

                #parameters for kappa
                xlens = 0
                ylens = 0
                elp = np.random.uniform()
                phi = np.random.uniform(low=0.0, high=2.*np.pi)
                sigma_v = 200000

                #parameters for source
                sigma_src = np.random.uniform(low=0, high=0.5)
                #np.random.normal(loc=0.0, scale = 0.01)
                x_src = np.random.uniform(low=-0.16, high=0.16)
                y_src = np.random.uniform(low=-0.16, high=0.16)

                self.Y[i,:] = self.gen_source(Xsrc, Ysrc, x_src = x_src, y_src = y_src, sigma_src = sigma_src, numpix_side = self.numpix_side)

                self.kappa[i,:] = self.Kappa_fun(xlens, ylens, elp, phi, sigma_v, numkappa_side = 193, kap_side_length = 2, rc=0, Ds = 1753486987.8422, Dds = 1125770220.58881, c = 299800000)

    #    for i in range(batch_size):
    #        
    #        #ARCS=1
    #        #nt = 0
    #
    #        while True:
    #            ARCS=1
    #            nt = 0
    #            while np.min(ARCS)==1 or np.max(ARCS)<0.4:
    #                nt = nt + 1
    #                if nt>1:
    #                    print 'Start adding effects again... ', i
    #                    
    #                    np.random.uniform()
    #                    #inds[i] = np.random.randint(0, high = max_file_num)
    ##
    ##
    ##
    ##                pick_folder = np.random.randint(0, high = num_data_dirs)
    #                pick_folder = 0
    #                arc_filename = d_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'
    #                #lens_filename = d_lens_path[pick_folder] +  train_or_test + '_' + "%07d" % (inds[i]+1) + '.png'
    #
    #                if train_or_test=='test':
    #                    Y[i,:] = Y_all_test[pick_folder][inds[i],0:num_out]
    #                    #mag[i] = Y_all_test[pick_folder][inds[i],7]
    #                else:
    #                    Y[i,:] = Y_all_train[pick_folder][inds[i],0:num_out]
    #                    #mag[i] = Y_all_train[pick_folder][inds[i],7]
    #
    #
    #                ARCS = np.array(Image.open(arc_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0;
    #                #LENS = np.array(Image.open(lens_filename),dtype='float32').reshape(numpix_side*numpix_side,)/65535.0
    #
    #                ARCS_SHIFTED, lensXY , m_shift, n_shift = pick_new_lens_center(ARCS,Y[i,:], xy_range = max_xy_range);
    ##                return;
    #                #LENS_SHIFTED = im_shift(LENS.reshape((numpix_side,numpix_side)), m_shift , n_shift ).reshape((numpix_side*numpix_side,))
    #
    #                ARCS = np.copy(ARCS_SHIFTED) 
    #                Y[i,3] = lensXY[0]
    #                Y[i,4] = lensXY[1]
    #
    #
    #            if (np.all(np.isnan(ARCS)==False)) and ((np.all(ARCS>=0)) and (np.all(np.isnan(Y[i,3:5])==False))) and ~np.all(ARCS==0):
    #                break
    #
    #        np.random.uniform()
    #        rand_state = np.random.get_state()
    #
    #        im_telescope = np.copy(ARCS) 
    #        apply_psf(im_telescope , max_psf_rms , apply_prob = 0.5)
    #        add_poisson_noise(im_telescope , apply_prob = 0.5)
    #        add_cosmic_ray(im_telescope,apply_prob = 0.5 )
    #        add_gaussian_noise(im_telescope)
    #        mask = gen_masks( 30 , ARCS.reshape((numpix_side,numpix_side)) , apply_prob = 0.5 )
    #        #mask = 1.0
    #
    #
    #
    #	
    #        if np.any(ARCS>0.4):
    #        	val_to_normalize = np.max(im_telescope[ARCS>0.4])
    #		if val_to_normalize==0:
    #			val_to_normalize = 1.0
    #		int_mult = np.random.normal(loc=1.0, scale = 0.01)
    #        	im_telescope = (im_telescope / val_to_normalize) * int_mult 
    #
    #
    #        im_telescope =  im_telescope.reshape(numpix_side,numpix_side)
    #        zero_bias = np.random.normal(loc=0.0, scale = 0.05)
    #        im_telescope = (im_telescope+zero_bias) * mask
    #        X[i,:] = im_telescope.reshape((1,-1))
    #        
    #        if np.any(np.isnan(X[i,:])) or np.any(np.isnan(Y[i,:])):
    #            X[i,:] = np.zeros((1,numpix_side*numpix_side))
    #            Y[i,:] = np.zeros((1,num_out))
    #
    #        np.random.set_state(rand_state)
    #	#return 0
