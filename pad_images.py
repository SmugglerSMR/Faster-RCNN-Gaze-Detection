import cv2
import glob

# im_pth = "VOCdevkit/VOC2012/EYE_test/"
# im_pth = "VOCdevkit/VOC2012/Original/"
im_pth = "EYE_val/"
desired_size = 450
incrementer = 1
for filename in glob.glob(im_pth + '*15.jpg'): #assuming gif
    im = cv2.imread(filename)    
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format
    im = cv2.resize(im, (new_size[1], new_size[0])) 

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
    value=color)
    cv2.imwrite("EYE_valout/EYE_vaaa{:02d}.jpg".format(incrementer), new_im)    
    incrementer = incrementer+1
    #hor_flip = cv2.flip(new_im,1)
    #cv2.imwrite("temp/EYE_{}.jpg".format(incrementer), hor_flip)
    #incrementer = incrementer+1
