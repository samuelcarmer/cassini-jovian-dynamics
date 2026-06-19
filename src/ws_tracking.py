

def track_ws(LON_SCANS,lon_crop,rough_left=154,rough_right=161,dx=0.1,start=3,wtrack = 6,sg = 1):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks

    print("wtrack: ", wtrack)

    l_lon_list = [] #initialize list of positions



    # find left/right estimates in pixel space
    rL = (np.abs(lon_crop -  rough_left)).argmin() 
    rR = (np.abs(lon_crop -  rough_right)).argmin() 
    print("rL: ", rL, "rR: ", rR)


    # crops all rows to look between estimated boundaries
    # this window is used to search for exact edges of dark region
    search_dark_edges = LON_SCANS[:,rL:rR]
    #print(np.shape(LON_SCANS))
    #print(np.shape(search_dark_edges))


    # 
    prev_search_idx = None

    for i in range(start, LON_SCANS.shape[0]):

        row = search_dark_edges[i, :]
        #print("i: ", i)

        # on first run, find exact boundaries of dark region
        # Use dark region boundaries as edges of WS search window
        if prev_search_idx is None:
            grad = np.gradient(row)
            left = np.argmin(grad)
            right = np.argmax(grad)

            s0 = min(left, right)
            s1 = max(left, right)

        # s's are moving boundaries for WS search window
        # s's are the previous image's WS +- half a search window length
        # max/ min used to prevent s's from indexing points outside the dark region
        else:
            s0 = max(0, prev_search_idx - wtrack)
            s1 = min(row.size, prev_search_idx + wtrack + 1)



        search_ws = row[s0:s1]
        search_ws = gaussian_filter1d(search_ws, sigma=sg)


        # take lightest/ brightest spot in search window
        light_idx_local = np.argmax(search_ws)

        # search index is relative to dark region boundary
        # light index is relative to left side of all frames
        search_idx = s0 + light_idx_local
        light_idx = rL + search_idx

        prev_search_idx = search_idx

        # Finds actual lon of WS relative to map of planet
        light_lon = lon_crop[light_idx]
        l_lon_list.append(light_lon)


    return np.array(l_lon_list)
        

