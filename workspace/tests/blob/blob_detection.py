# Standard imports
import cv2
import numpy as np
 
# Read image
img_path='/Users/agustingenoud/Desktop/MeLi/[MELI][CVC][20220719]prueba_técnica/workspace/data/imgs/D_601200-MLM49340822797_032022-F.jpg'
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
 
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create()
 
# Detect blobs.
keypoints = detector.detect(im)

 
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]),
    (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)