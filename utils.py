import numpy as np
import cv2
SEAM_COLOR = np.array([100, 100, 250]) 
def showPicture(im, show_seam_mask=None, rotate=False,picture_name = "picture",stop = False):
    vis = im.astype(np.uint8)
    if show_seam_mask is not None:
        vis[np.where(show_seam_mask == False)] = SEAM_COLOR
    if rotate:
        vis = rotateImg(vis, False)
    cv2.imshow(picture_name, vis)
    if stop:
        cv2.waitKey(0)
    cv2.waitKey(1)
    return vis

def resize(image, width):
    """
    return img after resize 
    """
    dim = None
    h, w = image.shape[:2]
    dim = (width, int(h * width / float(w)))
    return cv2.resize(image, dim)

def rotateImg(image, clockwise):
    k = 1 if clockwise else 3
    return np.rot90(image, k)   