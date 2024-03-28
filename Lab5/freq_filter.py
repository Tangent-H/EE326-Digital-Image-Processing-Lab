# %%
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# %%
def conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 0, padding: int = 0) -> np.ndarray:
    temp = np.pad(img, padding, mode='reflect')
    out_img = np.zeros(img.shape)
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            out_img[i, j] = np.sum(temp[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return out_img
# %%
def show_fft(img: np.ndarray):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min()) * 255
    magnitude_spectrum = magnitude_spectrum.astype(np.uint8)
    plt.figure()
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.show()
    return 
# %%
def sobel_spatial(img: np.ndarray):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = conv2d(img, sobel_y, stride=1, padding=1)
    return mask_y

# %%
def sobel_freq(img: np.ndarray):
    pass
# %%
if __name__ == "__main__":
    # read image
    img_t = []
    for i in range (1, 4):
        img_t.append(plt.imread(f"in/Q5_{i}.tif"))
    
