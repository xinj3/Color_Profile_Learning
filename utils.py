import cv2
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

CS_bc_type = "natural"

def print_histogram(fns, names=["target","estimation"], isfn=True, over=False, cdf=False):
    '''
    input:
        fns: list of file names(if isfn=True) or loaded images(isfn=False)
        names: list of names for the labels
        over: plot graph on the same graph is true, plot seperately otherwise
        cdf: print cdf if true, print histogram otherwise
    '''

    num = len(fns)
    fig, ax = plt.subplots(5,num,figsize=(15,15))

    for i, fn in enumerate(fns):
        if isfn:
            img1 = cv2.imread(fn)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        else:
            img1 = fn


        ax[0,i].imshow(img1)
        ax[0,i].set_title(names[i])
        if over:
            i = 0

        R_channel = np.histogram(img1[:,:,0], bins=256)[0]
        G_channel = np.histogram(img1[:,:,1], bins=256)[0]
        B_channel = np.histogram(img1[:,:,2], bins=256)[0]

        R_channel = signal.savgol_filter(R_channel, 21, 3)
        G_channel = signal.savgol_filter(G_channel, 21, 3)
        B_channel = signal.savgol_filter(B_channel, 21, 3)

        if cdf:
            R_channel = np.cumsum(R_channel)
            G_channel = np.cumsum(G_channel)
            B_channel = np.cumsum(B_channel)
        ax[1,i].plot(R_channel)
        ax[1,i].set_title("R Channel")
        ax[2,i].plot(G_channel)
        ax[2,i].set_title("G Channel")
        ax[3,i].plot(B_channel)
        ax[3,i].set_title("B Channel")
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray_channel = np.histogram(gray1, bins=256)[0]
        if cdf:
            gray_channel = np.cumsum(gray_channel)
        ax[4,i].plot(gray_channel)
        ax[4,i].set_title("Grayscale")
        fig.tight_layout()

    plt.show()


def plot_curve(curve_point, mount=[0,63,125,195,255]):
    cs = CubicSpline(mount, curve_point, bc_type=CS_bc_type)
    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256)
    if mount[0] != 0:
        curve[:mount[0]] = curve_point[0]
    if mount[-1] != 255:
        curve[mount[-1]:] = curve_point[-1]
    plt.plot(curve)
    plt.plot(np.arange(0,256))
    plt.scatter(mount, curve_point)
    plt.show()

def plot_curves(curve_points):
    mount = [0,63,125,195,255]
    cs = CubicSpline(mount, curve_points[0], bc_type=CS_bc_type)
    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256)
    plt.plot(curve)
    plt.plot(np.arange(0,256))
    plt.title("R Channel")
    plt.scatter(mount, curve_points[0])
    plt.show()

    cs = CubicSpline(mount, curve_points[1], bc_type=CS_bc_type)
    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256)
    plt.plot(curve)
    plt.plot(np.arange(0,256))
    plt.title("G Channel")
    plt.scatter(mount, curve_points[1])
    plt.show()

    cs = CubicSpline(mount, curve_points[2], bc_type=CS_bc_type)
    curve = cs(np.arange(0,256))
    curve = np.clip(curve, 0, 256)
    plt.plot(curve)
    plt.title("B Channel")
    plt.plot(np.arange(0,256))
    plt.scatter(mount, curve_points[2])
    plt.show()

