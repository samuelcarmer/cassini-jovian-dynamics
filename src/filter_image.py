


def kern_sharpen(images_list,strength):
    import numpy as np
    import cv2
    # Create the sharpening kernel 
    kernel = np.array([[0, -1, 0], [-1, strength, -1], [0, -1, 0]]) 
    print("Sharpening images ... \n")
    for image in images_list:			
        image  = cv2.filter2D(image,-1,kernel)

    return images_list




def lap_sharpen(images_list):
    import cv2
    print("Sharpening images ... \n")
    # Sharpen the image using the Laplacian operator 
    for image in images_list:			
        image = cv2.Laplacian(image, cv2.CV_64F)

    return images_list


def normalize_images(images_list):
    print("Normalizing images ... \n")
    for image in images_list:			
        for j in range(0,len(image[:,0])-1):
            image[j,:]  = (image[j,:]-image[j,:].mean())/image[j,:].std()
    
    return images_list


def clahe_sharpen(images_list, cL = 4.0, n = 8):
    from cv2 import createCLAHE
    print("Clahe'ing images ... \n")
    clahe = createCLAHE(clipLimit=cL, tileGridSize=(n, n))
    for image in images_list:
        image = clahe.apply(image)
    return images_list