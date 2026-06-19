

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
        





'''plt.figure()
        plt.imshow(LON_SCANS,cmap='gray',vmin=-2,vmax=2, extent=[lon_crop[0],lon_crop[-1],56,70],interpolation='none',origin='lower', aspect='auto')
        plt.axvline(x=lon_crop[0] + (rL_pix+left)/10, color='red')
        plt.axvline(x = lon_crop[0]+(rL_pix+ right)/10, color='blue')
        plt.axvline(x = lon_crop[0]+(rL_pix)/10, color='green')
        plt.axvline(x = lon_crop[0]+(rR_pix)/10, color='darkorange')


        plt.xlim([lon_crop[0],lon_crop[-1]])
        plt.ylim([56,70])'''


''' for i in range(start,LON_SCANS.shape[0]):
        print(i)
        grad = np.gradient(search_dark_edges[i, :])
        #print("grad : ",grad)
        #if np.argmin(grad #### could correct this so that left/right have to be within a certain distance from rL/rR
        left = np.argmin(grad)   
        right = np.argmax(grad)
        print("left: ", left, "right: ", right)

        
        
        if i >= start and left <= right:
            search_ws = search_dark_edges[i,left:right]
            print("search width: ",len(search_ws))
            print("search 1: ", search_ws)

            if i == start:
                s0, s1 = left, right
                print("s's: ", s0,s1)
                baseline = rL_pix + left 
            else:
                s0 = prev_local - wtrack
                s1 = prev_local + wtrack
                print("s's: ", s0,s1)

                if left > s0:
                    baseline = rL_pix + left                    

                else: 
                    baseline = prev_baseline + s0
                    search_ws = search_ws[s0:s1]

            print("search 2: ", search_ws)
            print("search width 2nd: ",len(search_ws))
            search_ws = gaussian_filter1d(search_ws, sigma=sg)

            #spot_row = LON_SCANS1[i,s0:s1]

            light_idx_local = np.argmax(search_ws)
            print("local", light_idx_local)

            light_idx = baseline + int(light_idx_local)
            prev_local = light_idx - rL_pix

            prev_idx = light_idx
            prev_baseline = baseline
            print("baseline: ", prev_baseline)
            print("light_idx: ", prev_idx)
            light_lon = lon_crop[light_idx]
            l_lon_list.append(light_lon)
        else:
            print(i)
            raise KeyError("left < right")'''