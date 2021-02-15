import numpy as np
import matplotlib.pyplot as plt
import random
import time
#%matplotlib inline

import tensorflow as tf
from astropy.cosmology import Planck15 as cosmo

def cart2pol(x,y):
    r = np.sqrt(x**2. + y**2.)
    theta = np.arctan2(y,x)
    return r, theta

def pol2cart(r,theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x,y

def raytrace(xim,yim,pars):
    Msun= 1.98892e30
    c = 2.998E8
    G = 6.67259E-11
    pc = 3.0857E16
    Mpc = pc * 1.0e6

    Zlens = 0.5
    Zsource = 2.0
    Dd = cosmo.angular_diameter_distance(Zlens).value
    Dds= cosmo.angular_diameter_distance_z1z2(Zlens,Zsource).value
    Ds = cosmo.angular_diameter_distance(Zsource).value

    REIN = pars[0]
    sigma = np.sqrt(299800000.0**2/(4.0*np.pi) * REIN *np.pi/180./3600. * Ds/Dds)
    M = (np.pi*(sigma**2)*(REIN*np.pi/180/3600)*Dd*Mpc)/G/Msun

    #print REIN , Ds , Dds
    #print sigma_cent , M
    
    elp = pars[1]
    angle = pars[2]
    xlens = pars[3] * np.pi /180.0/3600.0
    ylens = pars[4] * np.pi /180.0/3600.0
    
    ximage, yimage = xim.copy(), yim.copy()

    f = 1. - elp
    fprime = np.sqrt(1. - f**2.)

    Xi0 = 4*np.pi * (sigma/c)**2. * (Dd*Dds/Ds)

    ximage -= xlens
    yimage -= ylens
    
    r,theta = cart2pol(ximage,yimage)
    ximage,yimage = pol2cart(r,theta-(angle * np.pi /180))
    phi = np.arctan2(yimage,ximage)

    dxs = (Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsinh(np.cos(phi)*fprime/f)
    dys = (Xi0/Dd)*(np.sqrt(f)/fprime)*np.arcsin(np.sin(phi)*fprime)

    r,theta = cart2pol(dxs,dys)
    alphax,alphay = pol2cart(r,theta+(angle*np.pi/180))
    
    xsr = xim - alphax
    ysr = yim - alphay

    return alphax , alphay

