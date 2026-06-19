




def load_frames(whole=True):
    import samuel_cass_config

    import glob,os,gc
    import cv2


    import numpy as np



    whole = True
    if whole:
        images_path = samuel_cass_config.images_path
    else: 
        images_path = samuel_cass_config.subset



    images_list = []
    frames_list = []
    


    pattern = os.path.join(images_path, "*.png")
    #print(pattern)

    for file in sorted(glob.glob(pattern)):
    # 	print("Reading file ...",os.path.basename(file))
        image = cv2.imread(file,0) ##0,1,2 three different colors?
        image = np.flipud(image)[:,:]
    #	image = np.squeeze(image)
        frame = os.path.basename(file)[5:9]
        #print(frame)
        if int(frame) < 98:
            images_list.append(image)
            frames_list.append(frame)
        

    num_images   = len(images_list)
    print("Total number of images = " , num_images)
    images_shape = np.shape(images_list[0])


    print("Images shape = ", images_shape)
    #print('returns: images_list,num_images,images_shape,frames_list')


    return images_list,num_images,images_shape,frames_list