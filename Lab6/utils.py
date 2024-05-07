import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
import PIL.Image as Image
import os

image_extensions = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif']

def read_images(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.split('.')[-1].lower() in image_extensions:
            img = plt.imread(os.path.join(folder, filename))
            images[filename.split('.')[0]] = img
    return images

def show_distr(val,draw=True):
    if not draw:
        matplotlib.use('Agg')
    plt.hist(val, bins=256, range=(0,255), density=True)
    plt.title('Distribution of Pixel Values')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.legend()
    plt.draw()
    plt.pause(0.25)

def select_roi(image):
    """
    Display an image and let the user draw a rectangle on the image.
    Return the coordinates of the rectangle corners.
    
    Args:
    image (numpy array): The image to display in grayscale.
    
    Returns:
    tuple: Coordinates of the rectangle (x1, y1, x2, y2).
    """
    # Initial rectangle coordinates
    rect = (0, 0, 1, 1)
    drawing = False

    def on_mouse(event, x, y, flags, param):
        nonlocal rect, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            rect = (x, y, x, y)
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            rect = (rect[0], rect[1], x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            rect = (rect[0], rect[1], x, y)
            drawing = False

    cv2.namedWindow('Select ROI')
    cv2.setMouseCallback('Select ROI', on_mouse)
    
    while True:
        img_copy = image.copy()
        if rect[2] - rect[0] > 0 and rect[3] - rect[1] > 0:
            cv2.rectangle(img_copy, (rect[0], rect[1]), (rect[2], rect[3]), (255, 0, 0), 2)
        cv2.imshow('Select ROI', img_copy)
        if cv2.waitKey(1) & 0xFF == 13:  # Press ENTER to exit
            break

    cv2.destroyAllWindows()
    return rect

