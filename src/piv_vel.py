



def zonal_vortex_vel(band, DeltaP, FRAME_IDS,
        dx=0.1,
        window_size = 8,
        search_area_size = 10,
        overlap = 4,
        sig2noise_method='peak2peak'):
    


    import numpy as np
    from openpiv import tools, pyprocess, validation, filters, scaling
    import matplotlib.pyplot as plt

    u_list = []
    shape = np.shape(band)
    print("band shape in function: ",shape)
    n_frames = shape[0]
    print("SA size: ",search_area_size)
    print("W size: ",window_size)
    print("OV size: ",overlap)
    print(n_frames)
    for i in range(n_frames):

        if i>=1:

            dt = DeltaP[i]
            
            frame_a = band[i-1]
            frame_b = band[i]
            #print("a shape: ",frame_a.shape)
            #print("b shape: ",frame_b.shape)

            print("dt: ",dt)
            u, v, sig2noise = pyprocess.extended_search_area_piv(
                frame_a,
                frame_b,
                window_size=window_size,
                overlap=overlap,
                dt=dt,  
                search_area_size=search_area_size,
                sig2noise_method=sig2noise_method
            )
            
            #u_deg = u * dx
            u_list.append(u)
            #print("u array shape 1: ",np.shape(u))

            #print("i/5: ",i/5)
            #print("type : ",type(i))
            #print("u0 shape", np.shape(u0))

            # build grid (match u shape)
            if i % 10 ==0:
                ny, nx = u.shape
                x = np.linspace(0, frame_a.shape[1], nx)
                y = np.linspace(0, frame_a.shape[0], ny)
                X, Y = np.meshgrid(x, y)
                ua = np.empty_like(u)
                for i in range(ua.shape[0]):
                    ua[i,:] = np.nanmean(u[i],axis=0)
                print("ua shape: ",np.shape(ua))
                print("first line: ", ua[0,0])
                print("second line: ", ua[1,0])
                print("third line: ", ua[2,0])

                plt.figure()
                plt.imshow(frame_a, cmap='gray', origin='lower')
                plt.quiver(X, Y, ua, 0, color='red', scale=30)

                plt.title(f"PIV velocity field, frame {FRAME_IDS[i-1]}")
                plt.figure()
                plt.imshow(frame_b, cmap='gray', origin='lower')
                #plt.quiver(X, Y, u, 0, color='blue', scale=30)

                plt.title(f"PIV velocity field, frame {FRAME_IDS[i]}")


            plt.show()



    

    return (np.array(u_list)), sig2noise