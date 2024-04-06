# %%
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd

# %%
def pd_print(arr: np.ndarray):
    df = pd.DataFrame(arr)
    print(df)
    return

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

    phase_spectrum = np.angle(fshift)
    phase_spectrum = (phase_spectrum - phase_spectrum.min()) / (phase_spectrum.max() - phase_spectrum.min()) * 255
    phase_spectrum = phase_spectrum.astype(np.uint8)

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.title('Input Image')
    plt.subplot(1,3,2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Magnitude Spectrum')
    plt.subplot(1,3,3)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Phase Spectrum')
    return 
# %%
def sobel_spatial(img: np.ndarray):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = conv2d(img, sobel_y, stride=1, padding=1)
    return mask_y

# %%
def pad_sobel(padding_x: int = 1, padding_y: int = 1):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # add additional zeros along each axes
    sobel_y = np.vstack((np.zeros(sobel_y.shape[1]), sobel_y))
    sobel_y = np.hstack((np.zeros((sobel_y.shape[0],1)), sobel_y))
    sobel_y = np.pad(sobel_y, ((padding_x, padding_x), \
                    (padding_y, padding_y)), mode='constant', constant_values=0)
    return sobel_y

# %%
def sobel_freq(img: np.ndarray):
    sobel = pad_sobel(int(img.shape[0]/2), int(img.shape[1]/2))
    img_pad = np.pad(img, ((0, 4), (0, 4)), mode='constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    H_sobel = np.fft.fft2(sobel)
    Y_img = X_img * H_sobel
    y_img = np.fft.ifft2(Y_img)
    y_img = np.real(y_img)
    y_img = y_img[0:-4, 0:-4]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(sobel, cmap='gray')
    plt.axis('off')
    plt.title('Sobel Filter')
    plt.subplot(1,2,2)
    plt.imshow(img_pad, cmap='gray')
    plt.axis('off')
    plt.title('Padded Image')
    plt.show()

    show_fft(img_pad)
    show_fft(sobel)
    show_fft(y_img)
    return y_img

# %%
if __name__ == "__main__":
    # read image
    img_t = []
    for i in range (1, 4):
        img_t.append(plt.imread(f"in/Q5_{i}.tif"))
        show_fft(img_t[i-1])
    
    # apply sobel filter in spatial domain and frequency domain
    sobel_s = sobel_spatial(img_t[0])
    sobel_f = sobel_freq(img_t[0])
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(sobel_s, cmap='gray')
    plt.axis('off')
    plt.title('Sobel Spatial')
    plt.subplot(1,2,2)
    plt.imshow(sobel_f, cmap='gray')
    plt.axis('off')
    plt.title('Sobel Frequency')
    plt.show()

# %%
