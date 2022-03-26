from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def load_img(file='dataset/cat.jpg'):
    cat = Image.open(file)
    return np.asarray(cat)

def display_img(arr, title=""):
    plt.title(title)
    plt.imshow(arr, cmap='gray')
    plt.show()
    
def resize_img(arr, scale_x=0.1, scale_y=0.1):
    return cv.resize(arr,None,fx=scale_x, fy=scale_y, interpolation = cv.INTER_CUBIC)

def grayscale_img(arr):
    return cv.cvtColor(arr, cv.COLOR_BGR2GRAY)

def flip_img(arr):
    return 1-arr

def threshold_img(arr, lower, upper):
    if len(arr.shape) == 3:
        arr = grayscale_img(arr)
    _, arr = cv.threshold(arr, lower, upper, cv.THRESH_TOZERO_INV)
    return arr

def blur_img(arr, kernel=5):
    return cv.blur(arr, (kernel, kernel))
    # return cv.bilateralFilter(arr, kernel, 75, 75)

def canny_img(arr, sigma=0.33):
    m = np.median(arr)
    lower = int(max(0, (1.0 - sigma) * m))
    upper = int(min(255, (1.0 + sigma) * m))
    return cv.Canny(arr, lower, upper)

def denoise_img(arr):
    if len(arr.shape) == 2:
        arr = cv.fastNlMeansDenoising(arr, None, 10, 7, 21)
    elif len(arr.shape) == 3:
        arr = cv.fastNlMeansDenoisingColored(arr, None, 10, 10, 7, 21)
    return arr

def erode_img(arr, kernel, itr=1):
    """Thin boundaries"""
    kernel = np.ones((kernel, kernel), np.uint8)
    return cv.erode(arr, kernel, iterations=itr)

def dilate_img(arr, kernel=5, itr=1):
    """Thicken boundaries"""
    kernel = np.ones((kernel, kernel), np.uint8)
    return cv.dilate(arr, kernel, iterations=itr)

def open_img(arr, kernel=5, itr=1):
    """Close white holes"""
    arr = erode_img(arr, kernel, itr=itr)
    arr = dilate_img(arr, kernel, itr=itr)
    return arr

def close_img(arr, kernel=5, itr=1):
    """Close black holes"""
    arr = dilate_img(arr, itr=itr)
    arr = erode_img(arr, itr=itr)
    return arr

def downres_img(arr, itr=1):
    for _ in range(itr):
        arr = cv.pyrDown(arr)
    return arr

def upres_img(arr, itr=1):
    for _ in range(itr):
        arr = cv.pyrUp(arr)
    return arr

def oilpaint_img(arr):
    return cv.xphoto.oilPainting(arr, 7, 1)

def watercolor_img(arr, size=60, color=0.6):
    return cv.stylization(arr, sigma_s=size, sigma_r=color)

def sharpen_img(arr):
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    return cv.filter2D(arr, -1, kernel)

def discrete_img(arr):
    arr[arr > 0] = 1
    return arr