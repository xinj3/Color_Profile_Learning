import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.interpolate import CubicSpline

CS_bc_type = "natural"
# CS_bc_type = ("natural","clamped" )


class Image:
    '''
    Image class that takes in an image file name and calculates the RGB historgams
    '''
    def __init__(self, fn):
        def getHistograms(fn):
            BGRimg = cv2.imread(fn)
            R = np.histogram(BGRimg[:,:,2], bins=256)[0]
            G = np.histogram(BGRimg[:,:,1], bins=256)[0]
            B = np.histogram(BGRimg[:,:,0], bins=256)[0]
            return R, G, B
        self.histogramR = []
        self.histogramG = []
        self.histogramB = []
        self.histogramR, self.histogramG, self.histogramB, = getHistograms(fn)
    


class Data:
    '''
    data class for the initial approach, given a folder name, load all the images in the folder
    and calculates each image's RGB histogram
    '''
    def __init__(self, dirName):
        self.dirName = dirName
        self.filenames = [self.dirName+"/"+x for x in os.listdir(self.dirName)]
        self.numImg = len(self.filenames)

        self.histBinNum = 256
        self.histogram = []
        self.histogramR = []
        self.histogramG = []
        self.histogramB = []

    def getHistograms(self):
        for fn in self.filenames:
            if fn[-3:] != "jpg":
                continue
            BGRimg = cv2.imread(fn)

            self.histogramR.append(np.histogram(BGRimg[:,:,2], bins=self.histBinNum)[0])
            self.histogramG.append(np.histogram(BGRimg[:,:,1], bins=self.histBinNum)[0])
            self.histogramB.append(np.histogram(BGRimg[:,:,0], bins=self.histBinNum)[0])

            gray = cv2.cvtColor(BGRimg, cv2.COLOR_BGR2GRAY)
            self.histogram.append(np.histogram(gray, bins=self.histBinNum)[0])


class LearnFilter():
    '''
    initial approach's transformation learning class.
    takes in two data variables, and learn the linear transformation from data1 to data2
    '''
    def __init__(self, data1, data2):
        self.data1 = data1
        self.data2 = data2
        self.A = []
        self.A_R = []
        self.A_G = []
        self.A_B = []
        self.A_cdf = []
        self.A_cdf_R = []
        self.A_cdf_G = []
        self.A_cdf_B = []
        self.R_dis = np.arange(0, 256)
        self.G_dis = np.arange(0, 256)
        self.B_dis = np.arange(0, 256)

    
        

    def learnFormHist(self):
        X = np.reshape(self.data1.histogram, np.size(self.data1.histogram))
        X_R = np.reshape(self.data1.histogramR, np.size(self.data1.histogramR))
        X_G = np.reshape(self.data1.histogramG, np.size(self.data1.histogramG))
        X_B = np.reshape(self.data1.histogramB, np.size(self.data1.histogramB))

        X = np.dstack([X, np.ones(np.size(X))])[0]
        X_R = np.dstack([X_R, np.ones(np.size(X_R))])[0]
        X_G = np.dstack([X_G, np.ones(np.size(X_G))])[0]
        X_B = np.dstack([X_B, np.ones(np.size(X_B))])[0]

        y = np.reshape(self.data2.histogram, np.size(self.data2.histogram))
        y_R = np.reshape(self.data2.histogramR, np.size(self.data2.histogramR))
        y_G = np.reshape(self.data2.histogramG, np.size(self.data2.histogramG))
        y_B = np.reshape(self.data2.histogramB, np.size(self.data2.histogramB))

        self.A = np.linalg.lstsq(X, y, rcond=None)[0]
        self.A_R = np.linalg.lstsq(X_R, y_R, rcond=None)[0]
        self.A_G = np.linalg.lstsq(X_G, y_G, rcond=None)[0]
        self.A_B = np.linalg.lstsq(X_B, y_B, rcond=None)[0]

    def learnFormCDF(self):

        X = np.reshape(np.cumsum(self.data1.histogram), np.size(np.cumsum(self.data1.histogram)))
        X_R = np.reshape(np.cumsum(self.data1.histogramR), np.size(np.cumsum(self.data1.histogramR)))
        X_G = np.reshape(np.cumsum(self.data1.histogramG), np.size(np.cumsum(self.data1.histogramG)))
        X_B = np.reshape(np.cumsum(self.data1.histogramB), np.size(np.cumsum(self.data1.histogramB)))

        X = np.dstack([X, np.ones(np.size(X))])[0]
        X_R = np.dstack([X_R, np.ones(np.size(X_R))])[0]
        X_G = np.dstack([X_G, np.ones(np.size(X_G))])[0]
        X_B = np.dstack([X_B, np.ones(np.size(X_B))])[0]

        y = np.reshape(np.cumsum(self.data2.histogram), np.size(np.cumsum(self.data2.histogram)))
        y_R = np.reshape(np.cumsum(self.data2.histogramR), np.size(np.cumsum(self.data2.histogramR)))
        y_G = np.reshape(np.cumsum(self.data2.histogramG), np.size(np.cumsum(self.data2.histogramG)))
        y_B = np.reshape(np.cumsum(self.data2.histogramB), np.size(np.cumsum(self.data2.histogramB)))

        self.A_cdf = np.linalg.lstsq(X, y, rcond=None)[0]
        self.A_cdf_R = np.linalg.lstsq(X_R, y_R, rcond=None)[0]
        self.A_cdf_G = np.linalg.lstsq(X_G, y_G, rcond=None)[0]
        self.A_cdf_B = np.linalg.lstsq(X_B, y_B, rcond=None)[0]

def filterImage(A, A_R, A_G, A_B, filename, type="Hist"):
    '''
    takes initial aproach's linear transform and apply on the given filename image
    '''
    if type not in ["Hist", "CDF"]:
        print("Wrong type")
        pass

    img = cv2.imread(filename)
    
    histogramR = np.histogram(img[:,:,2], bins=256)[0]
    histogramG = np.histogram(img[:,:,1], bins=256)[0]
    histogramB = np.histogram(img[:,:,0], bins=256)[0]
    if type == "Hist":
        # print(np.shape(histogramR))
        histogramR = np.dstack([histogramR, np.ones(256)])[0]
        histogramG = np.dstack([histogramG, np.ones(256)])[0]
        histogramB = np.dstack([histogramB, np.ones(256)])[0]
        histogramR = histogramR @ A_R
        histogramG = histogramG @ A_G
        histogramB = histogramB @ A_B

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    histogram = np.histogram(gray, bins=256)[0]
    if type == "Hist":
        histogram = np.dstack([histogram, np.ones(256)])[0]
        histogram = histogram @ A

    CDF = np.cumsum(histogram)
    CDF_R = np.cumsum(histogramR)
    CDF_G = np.cumsum(histogramG)
    CDF_B = np.cumsum(histogramB)

    if type == "CDF":
        CDF = np.dstack([CDF, np.ones(256)])[0]
        CDF_R = np.dstack([CDF_R, np.ones(256)])[0]
        CDF_G = np.dstack([CDF_G, np.ones(256)])[0]
        CDF_B = np.dstack([CDF_B, np.ones(256)])[0]
        CDF = CDF @ A
        CDF_R = CDF_R @ A_R
        CDF_G = CDF_G @ A_G
        CDF_B = CDF_B @ A_B

    filtered = img.copy()
    R_channel = np.resize(img[:,:,2], np.size(gray))
    filtered[:,:,2] = np.resize((((CDF_R[R_channel]-1)/np.size(gray)*255)+0.5).astype("uint8"), np.shape(gray))  # R
    G_channel = np.resize(img[:,:,1], np.size(gray))
    filtered[:,:,1] = np.resize((((CDF_G[G_channel]-1)/np.size(gray)*255)+0.5).astype("uint8"), np.shape(gray))  # G
    B_channel = np.resize(img[:,:,0], np.size(gray))
    filtered[:,:,0] = np.resize((((CDF_B[B_channel]-1)/np.size(gray)*255)+0.5).astype("uint8"), np.shape(gray))  # B

    filtered = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    V_channel = np.resize(filtered[:,:,2], np.size(gray))
    filtered[:,:,2] = np.resize((((CDF[V_channel]-1)/np.size(gray)*255)+0.5).astype("uint8"), np.shape(gray))  # V

    filtered = cv2.cvtColor(filtered, cv2.COLOR_HSV2RGB)

    plt.imshow(filtered)
    plt.show()

    return filtered


def func(hist, curve_points, mount = [0,63,125,195,255]):
    '''
    apply the curve_points on given histogram
    input:
        hist: histogram with 256 bins
        curve_points: curve points of the curve function
        mount: mount points of the curve function
    output:
        histogram with curve_points applied on the hist
    '''
    cs = CubicSpline(mount, curve_points, bc_type=CS_bc_type)

    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256).astype("uint8")
    out = np.zeros(256)
    for i,x in enumerate(hist):
        out[curve[i]] += x
    return signal.savgol_filter(out, 21, 3)
    # return out

def cal_loss(input, output):
    '''
    calculate the MSE loss between input and output histograms
    input:
        input: histogram
        output: histogram
    output:
        the MSE difference
    '''
    def normalize(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    norm_input = normalize(input)
    norm_output = normalize(output)
    return np.mean((norm_input - norm_output)**2)

def optimize_curve(inputHist, outputHist, estCurve=np.array([0,63,125,195,255]), mount=[0,63,125,195,255], fineTune=True, fix_BC=True):
    '''
    return the optimized curve points that transform inputHist to outputHist
    '''
    estHist = np.array(inputHist)

    for i in range(5):
        old_loss = np.inf
        old_w = estCurve[i]
        for w in range(estCurve[i]-100,estCurve[i]+100, 1):
            estCurve[i] = w
            estCurve = np.clip(estCurve, 0, 256)
            estHist = func(inputHist, estCurve, mount=mount)
            loss = cal_loss(estHist,outputHist)
            if old_loss < loss:
                estCurve[i] = old_w
                estCurve = np.clip(estCurve, 0, 256)
            else:
                old_loss = loss
                old_w = w

    if fineTune:
        if fix_BC:
            estCurve[0] = 0
            estCurve[-1] = 255
        for i in [1,2,3]:
            old_loss = np.inf
            old_w = estCurve[i]
            for w in range(estCurve[i]-50,estCurve[i]+50, 1):
                estCurve[i] = w
                estCurve = np.clip(estCurve, 0, 256)
                estHist = func(inputHist, estCurve, mount=mount)
                loss = cal_loss(estHist,outputHist)
                if old_loss < loss:
                    estCurve[i] = old_w
                    estCurve = np.clip(estCurve, 0, 256)
                else:
                    old_loss = loss
                    old_w = w
    

    return estCurve


def apply_curve_actually(img, curve_points):
    '''
    return the result of curve_points filtered img
    '''
    mount = [0,63,125,195,255]
    cs = CubicSpline(mount, curve_points[0], bc_type=CS_bc_type)
    curveR = cs(np.arange(0,256))
    curveR = np.clip(curveR, 0, 256)
    cs = CubicSpline(mount, curve_points[1], bc_type=CS_bc_type)
    curveG = cs(np.arange(0,256))
    curveG = np.clip(curveG, 0, 256)
    cs = CubicSpline(mount, curve_points[2], bc_type=CS_bc_type)
    curveB = cs(np.arange(0,256))
    curveB = np.clip(curveB, 0, 256)

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            img[i,j,0] = curveR[img[i,j,0]]
            img[i,j,1] = curveG[img[i,j,1]]
            img[i,j,2] = curveB[img[i,j,2]]
    return img
   
def filterImage_curve(curveR,curveG,curveB, filename):

    img = cv2.imread(filename)
    filtered = apply_curve_actually(img[:,:,[2,1,0]],[curveR,curveG,curveB])
    
    plt.imshow(filtered)
    plt.show()

    return filtered


def get_diff(img1, img2):
    r_diff = cal_loss(np.histogram(img1[:,:,0], bins=256)[0], np.histogram(img2[:,:,0], bins=256)[0])
    g_diff = cal_loss(np.histogram(img1[:,:,1], bins=256)[0], np.histogram(img2[:,:,1], bins=256)[0])
    b_diff = cal_loss(np.histogram(img1[:,:,2], bins=256)[0], np.histogram(img2[:,:,2], bins=256)[0])

    return r_diff + g_diff + b_diff
    
################################################################

def func_advance(hist, curve_points, mount=[0,63,125,195,255]):
    '''
    Input:
        hist: original histogram
        curve_points: curve points that will be applied to the hist
        mount: the mount points(in original hist) for curve_points (for output hist)
    output:
        out: histogram with curve applied on input hist
    '''
    cs = CubicSpline(mount, curve_points, bc_type=CS_bc_type)
    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256).astype("uint8")
    if mount[0] != 0:
        curve[:mount[0]] = curve_points[0]
    if mount[-1] != 255:
        curve[mount[-1]:] = curve_points[-1]
    out = np.zeros(256)
    for i,x in enumerate(hist):
        out[curve[i]] += x
    # return signal.savgol_filter(out, 21, 3)
    return out

def cal_loss_adv(input, output):
    def normalize(x):
        return (x-np.min(x))/(np.max(x)-np.min(x))

    norm_input = normalize(input)
    norm_output = normalize(output)
    loss = 0
    inv = 1
    for i in range(0,256,inv):
        loss += (np.sum(norm_input[i:i+inv]) - np.sum(norm_output[i:i+inv]))**2
    return loss/256

def optimize_curve_point(inputHist, outputHist, curve, curve_indx, mount=[0,63,125,195,255]):
    '''
    Input:
        inputHist: input original histogram
        outputHist: the expected histogram inputHist needs to transform to
        curve: the curve that need to be optimized
        mount_indx: index of the curve points need to be optimized
        mount: the mount points for the curve
    output:
        estCurve: the optimized curve (only for curve_indx)
    '''

    old_loss = np.inf
    old_mount = curve[curve_indx]
    estCurve = list(curve)
    lower = curve[curve_indx]-100
    upper = curve[curve_indx]+100
    if lower < 0:
        lower = 0
    if upper > 255:
        upper = 255
    for w in range(lower, upper, 1):
        estCurve[curve_indx] = w
        estCurve = np.clip(estCurve, 0, 256)
        estHist = func_advance(inputHist, estCurve, mount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estCurve[curve_indx] = old_mount
            estCurve = np.clip(estCurve, 0, 256)
        else:
            old_loss = loss
            old_mount = w

    return estCurve

def optimize_advance_curve(inputHist, outputHist):
    '''
    input:
        inputHist: input original histogram
        outputHist: the expected histogram inputHist needs to transform to
    output:
        estMount: the mount points for estCurve points 
        estCurve: the curve points that transform inputHist to outputHist
    '''

    shadow = 0 + (255 - 0)//4
    highlight = 0 + (255 - 0)//4*3
    midtone = 0 + (255 - 0)//4*2
    estCurve = [0, shadow, midtone, highlight, 255]
    estMount = [0, shadow, midtone, highlight, 255]
    
    estHist = np.array(inputHist)

    
    # optimize black
    old_loss = np.inf
    old_mount = list(estMount)
    idx = 0
    for w in range(0, 50, 1):
        estMount[idx] = w
        cur_Curve = optimize_curve_point(inputHist, outputHist, estCurve, idx, estMount)
        estHist = func_advance(inputHist, cur_Curve, estMount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estMount = list(old_mount)
        else:
            old_loss = loss
            old_mount = list(estMount)
            estCurve = list(cur_Curve)
    
    
    # optimize white
    old_loss = np.inf
    old_mount = list(estMount)
    idx = -1
    for w in range(200, 256, 1):
        estMount[idx] = w
        cur_Curve = optimize_curve_point(inputHist, outputHist, estCurve, idx, estMount)
        estHist = func_advance(inputHist, cur_Curve, estMount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estMount = list(old_mount)
        else:
            old_loss = loss
            old_mount = list(estMount)
            estCurve = list(cur_Curve)


        
    # optimize shadow
    old_loss = np.inf
    old_mount = list(estMount)
    idx = 1
    for w in range(50, 100, 1):
        estMount[idx] = w
        cur_Curve = optimize_curve_point(inputHist, outputHist, estCurve, idx, estMount)
        estHist = func_advance(inputHist, cur_Curve, estMount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estMount = list(old_mount)
        else:
            old_loss = loss
            old_mount = list(estMount)
            estCurve = list(cur_Curve)
    

    
    # optimize highlight
    old_loss = np.inf
    old_mount = list(estMount)
    idx = 3
    for w in range(150, 200, 1):
        estMount[idx] = w
        cur_Curve = optimize_curve_point(inputHist, outputHist, estCurve, idx, estMount)
        estHist = func_advance(inputHist, cur_Curve, estMount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estMount = list(old_mount)
        else:
            old_loss = loss
            old_mount = list(estMount)
            estCurve = list(cur_Curve)

    # optimize midtone
    old_loss = np.inf
    old_mount = list(estMount)
    idx = 2
    for w in range(100, 150, 1):
        estMount[idx] = w
        cur_Curve = optimize_curve_point(inputHist, outputHist, estCurve, idx, estMount)
        estHist = func_advance(inputHist, cur_Curve, estMount)
        loss = cal_loss_adv(estHist, outputHist)
        if old_loss < loss:
            estMount = list(old_mount)
        else:
            old_loss = loss
            old_mount = list(estMount)
            estCurve = list(cur_Curve)

    return estMount, estCurve

def fit_curve(curve, mount=[0,63,125,195,255]):
    curve = np.clip(curve, 0, 256)
    if mount[0] != 0:
        curve[:mount[0]] = curve[0]
    if mount[-1] != 255:
        curve[mount[-1]:] = curve[-1]

    return curve

def apply_adcvane_curve(img, curve_points, mounts):
    cs = CubicSpline(mounts[0], curve_points[0], bc_type=CS_bc_type)
    curveR = cs(np.arange(0,256))
    curveR = np.clip(curveR, 0, 256)
    curveR = fit_curve(curveR, mounts[0])
    cs = CubicSpline(mounts[1], curve_points[1], bc_type=CS_bc_type)
    curveG = cs(np.arange(0,256))
    curveG = np.clip(curveG, 0, 256)
    curveG = fit_curve(curveG, mounts[1])
    cs = CubicSpline(mounts[2], curve_points[2], bc_type=CS_bc_type)
    curveB = cs(np.arange(0,256))
    curveB = np.clip(curveB, 0, 256)
    curveB = fit_curve(curveB, mounts[2])

    for i in range(np.shape(img)[0]):
        for j in range(np.shape(img)[1]):
            img[i,j,0] = curveR[img[i,j,0]]
            img[i,j,1] = curveG[img[i,j,1]]
            img[i,j,2] = curveB[img[i,j,2]]
    return img
   
def filterImage_curve_advanced(curves, mounts, filename):

    img = cv2.imread(filename)
    filtered = apply_adcvane_curve(img[:,:,[2,1,0]], curves, mounts)
    
    plt.imshow(filtered)
    plt.show()

    return filtered

