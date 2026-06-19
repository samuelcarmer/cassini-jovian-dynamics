import samuel_cass_config

import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.stats import linregress


import cv2 ## not sure that this is needed? 


from scipy.ndimage import gaussian_filter1d

import pandas as pd

from load_frames import load_frames
from filter_image import kern_sharpen,lap_sharpen, normalize_images

from drift_vel import find_t_space, track_dark_centroid, shift_image

from ws_tracking import track_ws

def find_element(array,element):
        import numpy as np
        idx  = (np.abs(array-element)).argmin()
        return idx

def format_coord(x, y):
    col = find_element(lon_crop, x)  # longitude pixel index
    row = y  # frame / row index
    return f"row={row}, col={col}"




images_list,num_images,images_shape,frames_list = load_frames()


FRAME_IDS = []
coords = []

for i in range(num_images):
    FRAME_IDS.append(frames_list[i])




####################################################################################
SHARPEN1 = True
#Kernal sharpening
if SHARPEN1:
    images_list = kern_sharpen(images_list, strength=5)

####################################################################################
SHARPEN2 = False
#Laplacian sharpening 
if SHARPEN2:
    images_list = lap_sharpen(images_list)

NORMALIZED = False
#Normalize the images
if NORMALIZED:
    images_list = normalize_images(images_list) #Without dividing it is just demeaned



lon_init  = 0
lon_final = 360
lat_init  = -90 # I think this may be wrong
lat_final = 90

LON_MIN = 150
LON_MAX = 200
LAT_MIN = -43.5
LAT_MAX = -28.5
OFFSET  = 0.0
LAT_CENTER = (LAT_MAX+LAT_MIN)/2
LAT_CENTER = -34.5 # more accurate than the avg
LAT_CENTER = LAT_CENTER+OFFSET
dlat = 2 # input what lat pixel range to look above and below the centered latitude (-36 deg)


print("LAT_CENTER: ",LAT_CENTER)



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
print("initial lon_range: ", lon_range)



# find what pixel row corresponds to the LAT I want to focus on
lat_index = find_element(lat, LAT_CENTER)
i1 = lat_index - dlat
#print("i1: ",i1)
i2 = lat_index + dlat + 1
#print("i2: ",i2)
lat_crop = lat[i1:i2]
lat_range = i2-i1
#print(lat_range)

LON_SCANS = np.empty((num_images,lon_range))
print("LON_SCNAS shape: ",np.shape(LON_SCANS))
# initialize contiainer for cropped images which focus on latitude window, width is of cropped region in pixel space
band = [img[i1:i2,j1:j2] for img in images_list]
for i in range(num_images):
    profile = np.mean(band[i],axis=0)
    #print(np.shape(profile))
    profile = (profile-profile.mean())/profile.std()
    LON_SCANS[i,:] = profile

# corresponds frames to time gone by with periods file
# T is periods gone by every frame, thour is hours gone by every frame, DeltaP is periods between frames, deg_omega is ang vel of Jupiter
T,thour,DeltaP, deg_omega = find_t_space(images_list,images_shape)




# track dark vortex of resulting howmoller in both lon and pixel space
xpix_track, xlon_track = track_dark_centroid(
    LON_SCANS,
    lon_crop,
    seed_lon=155,
    dx=dx,
    LON_MIN=150.0,
    LON_MAX=200.0,
    search_halfwidth=10,
    smooth_sigma=1
)


# fit linear regression to extract drift velocity. Assuming x = x0 + v(T) since howmoller looks linear
#not currently using for shift, just to extract velocity
#mask = np.isfinite(xlon_track)
slope, intercept, r_value, p_value, std_err = linregress(
    T,
    xlon_track
)
slopePix = slope/dx
print("slope in deg/period: ",slope)
#print("slope x DP: ",slopePix*DeltaP)
#print("increments: ", np.diff(xpix_track))

xfit_pix = [intercept/dx + slopePix * i for i in range(num_images)]
xfit_deg = intercept + slope * T


shift_from_start = xpix_track - xpix_track[0]


LON_SCANS_shifted = shift_image(LON_SCANS,shift_from_start)

c_max = np.max(LON_SCANS_shifted)
c_min = np.min(LON_SCANS_shifted)
#print(c_max)
#print(c_min)





plt.figure()
plt.imshow(LON_SCANS,cmap='gray',vmin=c_min,vmax=c_max,extent=[lon_crop[0],lon_crop[-1],0,num_images],interpolation='none',origin='lower', aspect='auto')
plt.title("Unshifted + Dark region Tracking. Latitude = "+str(np.round(LAT_CENTER,decimals=1)))
plt.plot(xlon_track, np.arange(len(xlon_track)), 'r-')
#plt.plot(xfit_deg, np.arange(len(xlon_track)), 'b-')
plt.xlabel("Longitude")
plt.ylabel("i")
#plt.savefig(samuel_cass_config.save_path + "ws_how_unshifted_darktrack.pdf", bbox_inches="tight")


plt.figure()
plt.imshow(LON_SCANS_shifted,cmap='gray',vmin=c_min,vmax=c_max,extent=[lon_crop[0],lon_crop[-1],0,num_images],interpolation='none',origin='lower', aspect='auto')
plt.title("Shifted. Latitude = "+str(np.round(LAT_CENTER,decimals=1)))
#plt.savefig(samuel_cass_config.save_path + "ws_how_shifted.pdf", bbox_inches="tight")



#plt.show()

# only look at rows 56 - 70
rs = 55
re = 70

good = (i < re and i >rs)
print("final crop lon_range: ", lon_range)
lon_range_cr = lon_range - 300
LON_SCANS_g = np.empty((re-rs, lon_range_cr))
print("final LON_SCANS shape: ", np.shape(LON_SCANS_g))
iterations = np.shape(LON_SCANS_shifted)[0]
#print(f" iterations: {iterations}")
new_frames_list = []
for i in range(iterations):
    if i < re and i >rs:
        #print(f"working, iteration {i}")
        #print(np.shape(LON_SCANS_shifted[i,:]))
        LON_SCANS_g[i-rs,:] = LON_SCANS_shifted[i,0:lon_range_cr]
        new_frames_list.append(frames_list[i])

print("new frames: ", new_frames_list)


# final lon crop. end boundary is 170 deg
j2n = 1700
lon_crop_n = lon[j1:j2n]
#print(f"lon_crop: {lon_crop_n}")


# signal is not clear before the 3rd row
start = 3
ws_lon = track_ws(LON_SCANS_g, lon_crop_n, start=start, wtrack=10)

# only look at T values that correspond to the correct rows
T = T[56+start:70+1]
# create linspace for clean graph of sin curve fit
T_graph = np.linspace(T[0],T[-1], 500)
print("T (in number of periods): ",T)

# not useful unless i want to find ang freq in rad/ row, which is not physically meaningful
rows = np.arange(56+start,70+1, dtype=float)
rows_graph = np.linspace(56+start,70+1,500)

print("ws size: ", len(ws_lon))
#print(f"ws_lon: {ws_lon:.0f}")
print(f"ws_lon: {np.array2string(ws_lon, formatter={'float_kind': lambda x: f'{x:.2f}'})}")
#print("row size: ", len(rows))
#print("rows: ", rows)



Period=9.92492 # number of hours per period, according to https://ntrs.nasa.gov/citations/19770042742



# find period with scipy find peaks
troughs, prop = find_peaks(-ws_lon)
#print("troughs: ",troughs)

tp = T[troughs]
wp = ws_lon[troughs]

ws_period = tp[-1] - tp[0]
ws_period = ws_period * Period * 3600
omega_from_peaks = 2 * np.pi / ws_period
print(ws_period, omega_from_peaks)







# fitting curve


initial_guesses = [0.75,2*np.pi/10,0,157.5]

x = T - T[0]
x_graph = T_graph - T[0]

def Sinusoid(X, A, ome, phi, D):
    return A * np.sin(ome * X + phi) + D

popt, pcov = curve_fit(f=Sinusoid,xdata=x, ydata=ws_lon, p0=initial_guesses)

print("popt: ", popt)
y_graph = Sinusoid(x_graph,*popt)
y = Sinusoid(T,*popt)

# mean absolute error
mae = np.abs(ws_lon - y)
print("mae avg: ", np.mean(mae))
print("mae std: ", np.std(mae))

'''
plt.figure()
plt.title("MAE")
'''




# omega is fitted in rad/ period. Then converted to rad/ hours, then rad/ sec
omega = popt[1]/(Period*3600)



plt.figure()
plt.imshow(LON_SCANS_g,cmap='gray',vmin=-2,vmax=0,extent=[lon_crop_n[0],lon_crop_n[-1],56,70],interpolation='none',origin='lower', aspect='auto')
plt.scatter(ws_lon, rows,color='red')
plt.ylim(56, np.max(rows)) # Extra room at the top for labels

plt.title(f"WS Tracking. Frames {new_frames_list[0]}-{new_frames_list[-1]}. Latitude = {str(np.round(LAT_CENTER,decimals=1))}")
plt.savefig(samuel_cass_config.save_path + "06-26/ws_track_0065_0078.png", dpi=1200)


plt.figure()
plt.scatter(T, ws_lon, color="b")
plt.plot(T, ws_lon,"b-", label="Data")
plt.plot(T_graph,y_graph ,"g-", label="Fit")
plt.scatter(tp, wp, color="red", marker="x", s=50, label="Peaks", zorder=3)

#plt.plot(rows, ws_lon - np.mean(ws_lon),"b-")
#plt.plot(rows_graph,y_graph - np.mean(y_graph),"g-")
#plt.scatter(rows, mae)
plt.xlabel("T in # Periods")
plt.ylabel("Longitude")
plt.title(f"WS Longitude vs. Jupiter Periods Frames {new_frames_list[0]}-{new_frames_list[-1]}")
plt.text(112.5, 158.25, fr"Fit: $\omega = {omega:.2e}$ rad/s", size = 6) 
plt.text(111.8, 158.45, fr"Trough distance: $\omega = {omega_from_peaks:.2e}$ rad/s", size = 6) 

plt.legend()


#plt.savefig(samuel_cass_config.save_path + "ws_sincurve_0065_0078.pdf", bbox_inches="tight")

plt.show()




'''
for x, y in zip(ws_lon, rows):
    # Format the label string (shows the y value)
    label = f"{y}" 
    
    # plt.text(x_coordinate, y_coordinate, string)
    # Adding a small offset to 'y' so the text sits right above the dot
    plt.text(x, y + 1, label, ha='center', va='bottom', fontsize=9)'''