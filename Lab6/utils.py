import numpy as np
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