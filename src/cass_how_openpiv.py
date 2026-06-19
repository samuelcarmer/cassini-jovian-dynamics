import glob,os,gc

import cass_config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pylab import *
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import sys
import math
#import cv2 ## not sure that this is needed? 

#import openpyxl

from scipy.ndimage import gaussian_filter1d

import pandas as pd

from load_frames import load_frames
from filter_image import kern_sharpen,lap_sharpen, normalize_images

import drift_vel
from drift_vel import rot_correction, find_t_space, track_dark_centroid, scan_lat, shift_image
import ws_tracking 
from ws_tracking import track_ws
import piv_vel
from piv_vel import zonal_vortex_vel

from scipy.stats import linregress


def find_element(array,element):
        import numpy as np
        idx  = (np.abs(array-element)).argmin()
        return idx




# load in frames as well as find data sizes and initialize frame list
images_list,num_images,images_shape,frames_list = load_frames()

# initizialize
FRAME_IDS = []
coords = []

for i in range(num_images):
    FRAME_IDS.append(frames_list[i])

####################################################################################
SHARPEN1 = False
#Kernal sharpening
if SHARPEN1:
    images_list = kern_sharpen(images_list)

####################################################################################
SHARPEN2 = False
#Laplacian sharpening 
if SHARPEN2:
    images_list = lap_sharpen(images_list)
####################################################################################
NORMALIZED = False
#Normalize the images
if NORMALIZED:
    images_list = normalize_images(images_list) 

# input scale of globe
lon_init  = 0
lon_final = 360
lat_init  = -90 # probably wrong-Li etal (03) says the images range from +-80 deg
lat_final = 90

# input ROI
#LON_MIN = 150
#LON_MAX = 162
LON_MIN = 150
#LON_MIN = 149.7
LON_MAX = 200
#LON_MAX = 198.3
LAT_MIN = -43.5
LAT_MAX = -28.5
OFFSET  = 0.0
LAT_CENTER = (LAT_MAX+LAT_MIN)/2
LAT_CENTER = -34.5 # more accurate than the avg
LAT_CENTER = LAT_CENTER+OFFSET
#print(LAT_CENTER)
dlat = 11 # input what lat pixel range to look above and below the centered latitude (-36 deg)




lon    = np.linspace(lon_init,lon_final,images_shape[1]) #longitudes in pixel space
lat    = np.linspace(lat_init,lat_final,images_shape[0]) #latitudes in pixel space

# dx is number of degrees every pixel, which is 0.1 for these frames
dx     = 360./images_shape[1]
#print(f"dx{dx}")

#Find area to focus on. LONMIN and LONMAX are manually found and entered values
j1 = find_element(lon, LON_MIN)
j2 = find_element(lon, LON_MAX)+1
#print(j1, j2)

# make sure lon crop boundaries are correct
if j1 > j2:
    j1, j2 = j2, j1
    print('something weirds going on. check the lon bounds input')


lon_crop = lon[j1:j2] #cropped space that focuses near the ROI
lon_range = j2 - j1 # pixel length of cropped region
print("lon_range: ", lon_range)



# find what pixel row corresponds to the LAT I want to focus on
lat_index = find_element(lat, LAT_CENTER)
i1 = lat_index - dlat
#print("i1: ",i1)
i2 = lat_index + dlat + 1
#print("i2: ",i2)
lat_crop = lat[i1:i2]
lat_range = i2-i1
#print(lat_range)

# initialize contiainer for cropped images which focus on latitude window, width is of cropped region in pixel space
band = [img[i1:i2,j1:j2] for img in images_list]
print(np.shape(band))
#images_list[:][i1:i2,j1:j2]

# corresponds frames to time gone by with periods file
# T is periods gone by every frame, tper is hours gone by every frame, DeltaP is periods between frames, deg_omega is ang vel of Jupiter
T,tper,DeltaP, deg_omega = find_t_space(images_list,images_shape)
def format_coord(x, y):
    col = find_element(lon_crop, x)  # longitude pixel index
    row = y  # frame / row index
    return f"row={row}, col={col}"



if False:
    for i in range(num_images):
        if i >= 80:
            plt.figure()
            plt.imshow(
            band[i],
            cmap='gray',
            vmin=0, vmax=255,
            extent=[lon_crop[0], lon_crop[-1], lat_crop[0], lat_crop[-1]],
            origin='lower',
            aspect='auto'
            )

            plt.xlabel("Longitude")
            plt.ylabel("i")
            plt.title(f"Cropped image: {i}, ID: {frames_list[i]}, Height:{lat_range * 0.1:.1f}, Width:{lon_range * 0.1:.1f} deg ")


        ax = plt.gca()



    #ax.format_coord = format_coord


u , s2n = zonal_vortex_vel(band,DeltaP, FRAME_IDS)

#print("Frame IDS: ", FRAME_IDS[0:10])

#print("u array shape: ",np.shape(u))
#print("u cols", u[0,:,:])
#plt.show()








# initialize containers for howmoller— every row is scan from 1 image, width is of cropped region in pixel space
LON_SCANS = np.empty((num_images,lon_range))
LON_SCANS1 = np.empty((num_images,lon_range))