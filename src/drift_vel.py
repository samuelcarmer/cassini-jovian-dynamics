def test():
    print("it works")




def find_t_space(images_list,images_shape,Period=9.92492):
    from pandas import read_excel
    import numpy as np

    num_images = len(images_list)

    lon_length = images_shape[1]
    dx     = 360./lon_length #According to Limings 2004 paper for the movie this should be 0.1 
    #degrot0 = -0.2160

    period_file_path = '/Users/samuelcarmer/Documents/Cassini/CassiniHighRes/periods.xlsx'

    df = read_excel(period_file_path)
    deltaP = df.iloc[:num_images,0].to_numpy() 
    thour = Period * np.concatenate(([0], np.cumsum(deltaP[:-1]))) # hours gone by every frame
    deg_per_hr = 360/Period # one rotation per period

    #print(deltaP)
    #print(thour/Period)

    return thour/Period,thour,deltaP, deg_per_hr




def scan_lat(images_list,frames_list, lat, num_images, j1,j2, dlat=3, LAT_CENTER=-36.0):

    import numpy as np
    def find_element(array,element):
        import numpy as np
        idx  = (np.abs(array-element)).argmin()
        return idx
    
    lon_range = j2 - j1
    LON_SCANS = np.empty((num_images,lon_range))
    band_list = []
    

    FRAME_IDS = []

    

    for i in range(num_images):
        FRAME_IDS.append(frames_list[i])
        lat_index = find_element(lat, LAT_CENTER)
        i1 = max(0, lat_index - dlat)
        i2 = min(images_list[i].shape[0], lat_index + dlat + 1)
            
        band = images_list[i][i1:i2, j1:j2]
        #print(f"Band {np.shape(band)}")
        band_list.append(band)
        

        profile_align = np.mean(band, axis=0)
        #profile_align = images_list[i][lat_index, j1:j2]
        profile_align_norm = (profile_align - profile_align.mean()) / profile_align.std()
        #profile_align_norm = profile_align

        LON_SCANS[i,:] = profile_align_norm

    return LON_SCANS, np.array(band_list), FRAME_IDS






def track_dark_centroid(LON_SCANS, lon_crop, seed_lon=157.0, dx=0.1,
                        LON_MIN=150.0, LON_MAX=200.0, search_halfwidth=10,
                        smooth_sigma=1):
    
    import numpy as np
    from scipy.ndimage import gaussian_filter1d
    

    n_rows, n_cols = LON_SCANS.shape

    
    x_prev = (np.abs(lon_crop - seed_lon)).argmin()

    xpix_track = np.zeros(n_rows)
    xlon_track = np.zeros(n_rows)


    xpix_track = np.full(n_rows, np.nan)
    xlon_track = np.full(n_rows, np.nan)

    for i in range(n_rows):
        row = LON_SCANS[i].copy()

        if smooth_sigma is not None and smooth_sigma > 0:
            row = gaussian_filter1d(row, sigma=smooth_sigma)

        left  = max(0, x_prev - search_halfwidth)
        right = min(n_cols-1, x_prev + search_halfwidth)

        row_win = row[left:right+1]
        lon_win = lon_crop[left:right+1]
        x_idx   = np.arange(left, right+1)

        # dark values have higher weight
        weights = np.maximum(0.0, -row_win)

        if weights.sum() == 0:
            # fallback to use darkest pixel
            idx_local = np.argmin(row_win)
            x_curr_idx = x_idx[idx_local]
            x_curr_lon = lon_win[idx_local]
        else:
            x_curr_lon = np.sum(lon_win * weights) / np.sum(weights)
            x_curr_idx = (np.abs(lon_crop -  x_curr_lon)).argmin()

        xpix_track[i] = x_curr_idx
        xlon_track[i] = x_curr_lon

        # Update center for next row
        x_prev = int(x_curr_idx)

    return xpix_track, xlon_track


def shift_image(image_set,shift):
    import numpy as np
    image_set_shifted = np.empty_like(image_set)

    length = np.shape(image_set)[0]
    shape = np.shape(image_set)
    print("len shape: ", len(shape))
    print("length: ", length)
    if len(shape) > 2 and length >1:
        for i in range(length):
            current_shift = int(round(shift[i]))
            image_set_shifted[i] = np.roll(image_set[i], -current_shift, axis=1)
    elif len(shape) == 2 and length>1:
        print(len(shape))
        for i in range(length):
            current_shift = int(round(shift[i]))
            image_set_shifted[i, :] = np.roll(image_set[i,:], -current_shift, axis=0)
    else:
        image_set_shifted[i, :] = np.roll(image_set[i, :], shift)

    return image_set_shifted








def rot_correction(images_list,images_shape):
    from pandas import read_excel
    import numpy as np

    num_images = len(images_list)

    lon_length = images_shape[1]
    dx     = 360./lon_length #According to Limings 2004 paper for the movie this should be 0.1 
    Period = 9.92492 #nasa measurements
    Period = 10.0
    #degrot0 = -0.2160

    period_file_path = '/Users/samuelcarmer/Documents/Cassini/CassiniHighRes/periods.xlsx'

    df = read_excel(period_file_path)
    deltaP = df.iloc[:num_images,0].to_numpy() 
    thour = Period * np.concatenate(([0], np.cumsum(deltaP[:-1]))) # hours gone by every frame
    deg_per_hr = 360/Period # one rotation per period

    print(deltaP)

    
    for i in range(num_images):

        pixel_shift = int(thour[i]* deg_per_hr /dx )
        nshift = int(0.216*i*(Period/24)/dx) #raul method
        #images_list[i]=np.roll(images_list[i],nshift,axis=1) #raul method
        #images_list[i]=np.roll(images_list[i],pixel_shift,axis=1)
        
        print(f' pixelshift: {pixel_shift}')
    
    return images_list, thour