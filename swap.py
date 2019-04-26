import numpy as np
import cv2
    
im = cv2.imread('EYE_test/EYE_te01.jpg')
cv2.imshow("image", im)

im[np.where((im == [103,range(100,127),range(1,255)]))] = [255,0,0]
# cv2.imwrite('output.png', im)
cv2.imshow("image2", im)
cv2.waitKey(0)
cv2.destroyAllWindows()