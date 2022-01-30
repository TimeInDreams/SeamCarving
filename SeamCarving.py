import sys
from tqdm import trange
import numpy as np
import cv2
from numba import jit
import argparse
import utils

ENERGY_MASK_CONST = 1000000.0
MASK_THRESHOLD = 10.0
USE_FORWARD_ENERGY = True

def backward_energy(img):

    """
        Simple gradient magnitude energy map.
    """    
    img = img.astype(np.float64)
    kernel_x = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    kernel_y = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    dst_x = cv2.filter2D(img, -1, kernel=kernel_x)
    dst_y = cv2.filter2D(img, -1, kernel=kernel_y)

    energy_map =np.absolute(cv2.addWeighted(np.absolute(dst_x), 0.5, np.absolute(dst_y), 0.5, 0))
    energy_map = energy_map.sum(axis=2)

    #norm to make value in (0,255) to show energy distribution
    dst_norm=np.empty(energy_map.shape,dtype=np.float64)
    
    cv2.normalize(energy_map,dst_norm,0,255,norm_type=cv2.NORM_MINMAX)
    dst_norm = dst_norm.astype(np.uint8)
    cv2.imshow("E_map",dst_norm)
    
    return energy_map

@jit
def forward_energy(img):

    """
       minimum inserted energy to make energy map.
    """
    h,w,_ = img.shape
    im = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    energy_map = np.zeros((h,w))
    mat_M = np.zeros((h,w))

    U = np.roll(im, 1, axis=0)
    L = np.roll(im, 1, axis=1)# ->
    R = np.roll(im, -1, axis=1)# <-

    mat_cU = np.abs(R - L)
    mat_cL = np.abs(U - L) + mat_cU
    mat_cR = np.abs(U - R) + mat_cU

    for i in range (0,h-1):
        row_U = mat_M[i]
        row_L = np.roll(row_U,1)
        row_R = np.roll(row_U,-1)

        array_mULR = np.array([row_U,row_L,row_R])#3 x weight
        array_cULR = np.array([mat_cU[i+1],mat_cL[i+1],mat_cR[i+1]])
        array_mULR += array_cULR

        row_min_idx = np.argmin(array_mULR,axis=0)#[0 1 1 1 2],return index of min from U/L/R
        mat_M[i+1] = np.choose(row_min_idx,array_mULR)# min value in a row
        energy_map[i+1] = np.choose(row_min_idx,array_cULR)


    return energy_map


@jit
def addOneSeam(img,seam_idx):

    """
       insert a seam in img along the idx of seam_idx
       seam_idx record every col num in [0,h]
    """
    h,w = img.shape[:2]
    dst = np.zeros((h,w+1,3))

    #insert pixel into dst[i,seam_idx[i]+1]
    for i in trange(h):
        col = seam_idx[i]#the idx of col seam
        
        for channel in range(3):
            if col == 0:
                pixel = np.average(img[i,col:col+2,channel])
                dst[i,col,channel] = img[i,col,channel]
                dst[i, col + 1, channel] = pixel
                dst[i,col+1:,channel] = img[i,col:,channel]
            else:
                pixel = np.average(img[i,col-1:col+1,channel])
                dst[i,: col,channel] = img[i,: col,channel]
                dst[i, col, channel] = pixel
                dst[i,col+1:,channel] = img[i,col:,channel]

    return dst

def addOneSeamIntoMask(src_mask,seam_idx):
    h, w = src_mask.shape[:2]
    output = np.zeros((h, w + 1))
    for row in range(h):
        col = seam_idx[row]
        if col == 0:
            p = np.average(src_mask[row, col: col + 2])
            output[row, col] = src_mask[row, col]
            output[row, col + 1] = p
            output[row, col + 1:] = src_mask[row, col:]
        else:
            p = np.average(src_mask[row, col - 1: col + 1])
            output[row, : col] = src_mask[row, : col]
            output[row, col] = p
            output[row, col + 1:] = src_mask[row, col:]

    return output

def getDarwMask(img):
    h,w = img.shape[:2]
    img_temp = img.copy()
    mask =np.zeros((h,w))
    points = []
    start_point = [-1,-1]
    cv2.namedWindow("draw_win")
    cv2.setMouseCallback("draw_win",OnMouse,[img_temp,points,start_point])
   
    
    c = cv2.waitKey(0)
    
    contours = np.array(points)
    cv2.drawContours(mask,[contours],-1,255,thickness= -1)
    
    utils.showPicture(mask,picture_name="mask",stop=True)
    cv2.destroyAllWindows()
    return mask

def OnMouse(event,x,y,flags,param):
    start_point = param[2]
    cv2.imshow("draw_win",param[0])
    if  event == cv2.EVENT_LBUTTONDOWN:
        param[2] = [x,y] #获取起始点，并保存
        param[1].append(param[2])
        cv2.circle(param[0],tuple(start_point),5,(0,0,255),0)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        Cur_point = [x,y]
        # print(points)
        if param[1][-1]!=[-1,-1] and Cur_point!=[-1,-1]:
            cv2.line(param[0],tuple(param[1][-1]),tuple(Cur_point),(0,0,255)) #以上一个点和当前点为起始和结束点，绘制轨迹
            param[1].append(Cur_point)
    elif event == cv2.EVENT_LBUTTONUP:
        Cur_point=start_point
        cv2.line(param[0],tuple(param[1][-1]),tuple(Cur_point),(255,255,255))
        cv2.circle(param[0],tuple(Cur_point),1,(255,255,255))
    else:
        # print("mouse event is undefined")
        pass

@jit 
def rmOneSeam(img,mask):
    h,w = img.shape[:2]
    mask_BGR = np.stack([mask]*3,axis=2)
    return img[mask_BGR].reshape((h,w-1,3))
@jit
def rmSeamFromMask(src_mask, seam_mask):
    h, w = src_mask.shape[:2]
    return src_mask[seam_mask].reshape((h, w - 1))

@jit
def getMinSumEnergySeam(img,protective_mask = None,rm_mask = None,use_forward_energy = True):
    """
    get min energy map "energy_map" and min sum energy map "M"
    :param mask : protective mask
    :param use_forward_energy: calculate energy map method
    """
    h,w = img.shape[:2]
    if use_forward_energy:
        energy_fun = forward_energy 
    else:
        energy_fun = backward_energy

    M = energy_fun(img)

    if protective_mask is not None:
        M[np.where(protective_mask > MASK_THRESHOLD)] = ENERGY_MASK_CONST
    if rm_mask is not None:
        M[np.where(rm_mask > MASK_THRESHOLD)] = -ENERGY_MASK_CONST

    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, h):
        for j in range(0, w):
            if j == 0:
                idx = np.argmin(M[i - 1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy 

    seam_idx = []
    mask = np.ones((h,w),dtype=np.bool)   
    j = np.argmin(M[-1])
    for i in range(h-1, -1, -1):
        mask[i, j] = False
        seam_idx.append(j)
        j = backtrack[i, j]

    seam_idx.reverse()#col idx of seam (0,h)

    return np.array(seam_idx), mask

def rmAllSeams(img,num_rm,mask = None ,is_visable = False, rot = False):
    for i in trange(num_rm):
        _,seam_mask = getMinSumEnergySeam(img,protective_mask=mask,use_forward_energy=USE_FORWARD_ENERGY)
        if is_visable:
            utils.showPicture(img,seam_mask,rotate=rot)
        img = rmOneSeam(img,seam_mask)
        if mask is not None:
            mask = rmSeamFromMask(mask, seam_mask)
    return img,mask

def insertAllSeams(img,num_insert,mask = None , is_visable = False, rot = False):
    img_temp = img.copy()
    seam_idx_record =[]
    if mask is not None:
        mask_temp = mask.copy()
    else:
        mask_temp = None
    for i in range(num_insert):
        seam_idx , seam_mask=getMinSumEnergySeam(img_temp,mask_temp,use_forward_energy=USE_FORWARD_ENERGY)
        if is_visable:
            utils.showPicture(img_temp,seam_mask,rotate=rot)

        seam_idx_record.append(seam_idx)
        img_temp = rmOneSeam(img_temp,seam_mask)

        if mask_temp is not None:
            mask_temp = rmSeamFromMask(mask_temp,seam_mask)

    seam_idx_record.reverse()
    while seam_idx_record:
        seam_idx = seam_idx_record.pop()
        # print("****************seam idx record num:",len(seam_idx_record),"***************")
        img = addOneSeam(img,seam_idx)

        if is_visable:
            utils.showPicture(img,rotate=rot,picture_name="insert seam")
        if mask is not None:
            mask = addOneSeamIntoMask(mask,seam_idx)
        for remaining_seam in seam_idx_record:
            remaining_seam[np.where(remaining_seam >= seam_idx)] += 2

    return img , mask

def seamCarving(img,dy,dx,mask = None,is_visable = False):
    '''
    SeamCarving interface 
    '''
    img = img.astype(np.float64)
    h,w = img.shape[:2]

    assert h + dy > 0 and w + dx > 0 and dy <= h and dx <= w , 'img resize idx out of range'

    if mask is not None:
        mask = mask.astype(np.float64)

    dst = img

    if dx < 0:
        dst, mask = rmAllSeams(dst, -dx, mask, is_visable)

    elif dx > 0:
        dst, mask = insertAllSeams(dst, dx, mask, is_visable)

    if dy < 0:
        dst = utils.rotateImg(dst, True)
        if mask is not None:
            mask = utils.rotateImg(mask, True)
        dst, mask = rmAllSeams(dst, -dy, mask, is_visable, rot=True)
        dst = utils.rotateImg(dst, False)

    elif dy > 0:
        dst = utils.rotateImg(dst, True)
        if mask is not None:
            mask = utils.rotateImg(mask, True)
        dst, mask = insertAllSeams(dst, dy, mask, is_visable, rot=True)
        dst = utils.rotateImg(dst, False)

    return dst



if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("-resize", action='store_true')
    group.add_argument("-scale",action="store_true")

    ap.add_argument("-in", help="Path to image", required=True)
    ap.add_argument("-out", help="Output file name", required=True)

    add_mask_method = ap.add_mutually_exclusive_group(required=True)
    add_mask_method.add_argument("-drawmask",help="draw a (protective) mask",action='store_true')
    add_mask_method.add_argument("-mask", help="Path to (protective) mask")
    add_mask_method.add_argument("-nomask",help="No (protective) mask",action='store_true')
    ap.add_argument("-rmask", help="Path to removal mask")
    ap.add_argument("-dy", help="Number of vertical seams to add/subtract", type=float, default=0)
    ap.add_argument("-dx", help="Number of horizontal seams to add/subtract", type=float, default=0)
    ap.add_argument("-vis", help="Visualize the seam removal process", action='store_true')
    ap.add_argument("-backward_energy", help="Use backward energy map (default is forward)", action='store_true')
    args = vars(ap.parse_args())

    IM_PATH, MASK_PATH, OUTPUT_NAME, R_MASK_PATH = args["in"], args["mask"], args["out"], args["rmask"]

    im = cv2.imread(IM_PATH)
    assert im is not None
    mask = cv2.imread(MASK_PATH, 0) if MASK_PATH else None
    rmask = cv2.imread(R_MASK_PATH, 0) if R_MASK_PATH else None

    mask = getDarwMask(im) if args["drawmask"] else None
    USE_FORWARD_ENERGY = not args["backward_energy"]

    # image resize mode
    dy =dx =None
    h,w =im.shape[:2]
    if args["resize"]:
        dy, dx = int(args["dy"]), int(args["dx"])
    # image scale resize mode
    if args["scale"]:
        assert args["dy"] >-1.0 and args["dy"] <1.0 and args["dx"]>-1.0 and args["dx"]<1.0
        dy, dx = int(args["dy"]*h) ,int(args["dx"]*w)
        

    assert dy is not None and dx is not None
    
    output = seamCarving(im, dy, dx, mask, args["vis"])
    cv2.imwrite(OUTPUT_NAME, output)