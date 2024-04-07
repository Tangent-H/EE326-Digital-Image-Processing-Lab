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

    # real and imaginary part
    # real = np.real(fshift)
    # real = (real - real.min()) / (real.max() - real.min()) * 255
    # real = np.log(real + 1)
    # real = real.astype(np.uint8)

    # imag = np.imag(fshift)
    # imag = (imag - imag.min()) / (imag.max() - imag.min()) * 255
    # imag = np.log(imag + 1)
    # imag = imag.astype(np.uint8)

    # plt.figure()
    # plt.subplot(1,2,1)
    # plt.imshow(real, cmap='gray')
    # plt.axis('off')
    # plt.title('Real Part')
    # plt.subplot(1,2,2)
    # plt.imshow(imag, cmap='gray')
    # plt.axis('off')
    # plt.title('Imaginary Part')
    # plt.show()

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
    sobel_y = np.pad(sobel_y, ((0, padding_x), \
                    (0, padding_y)), mode='constant', constant_values=0)
    return sobel_y

# %%
def sobel_freq(img: np.ndarray):
    sobel = pad_sobel(int(img.shape[0]), int(img.shape[1]))
    img_pad = np.pad(img, ((0, 4), (0, 4)), mode='constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    # X_img = np.fft.fftshift(X_img)
    H_sobel = np.fft.fft2(sobel)
    # H_sobel = np.fft.fftshift(H_sobel)
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
    # for debugging
    # show_fft(img_pad)
    # show_fft(sobel)
    # show_fft(y_img)
    
    # for i in range(y_img.shape[0]):
    #     for j in range(y_img.shape[1]):
    #         y_img[i, j] = y_img[i, j] * (-1) ** (i + j)
    return y_img

# %%
def ideal_lpf(D: int, img: np.ndarray):
    # create a mask
    H_lpf = np.zeros((img.shape[0]*2, img.shape[1]*2))
    
    img_pad = np.pad(img, ((0,img.shape[0]), (0,img.shape[1])), mode='constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    X_img = np.fft.fftshift(X_img)
    xx, yy = np.ogrid[0:H_lpf.shape[0], 0:H_lpf.shape[1]]
    # compared to mgrid, ogrid is more efficient(only generate the axis value: a vector)
    mid_x = H_lpf.shape[0] // 2
    mid_y = H_lpf.shape[1] // 2
    H_lpf = np.where((xx-mid_x) ** 2 + (yy-mid_y) ** 2 <= D ** 2, 255, 0)
    
    # # debugging
    # plt.figure()
    # plt.imshow(H_lpf, cmap='gray')
    # plt.axis('off')
    # plt.title('Ideal LPF')
    # plt.show()

    Y_img = X_img * H_lpf
    Y_img = np.fft.ifftshift(Y_img)
    y_img = np.fft.ifft2(Y_img)
    y_img = np.real(y_img)
    y_img = y_img[0:img.shape[0], 0:img.shape[1]]
    return y_img

# %%
def gaussian_filter(D: int, img: np.ndarray, lowpass: bool = True):
    
# %% 
#-----------------Testing Section-----------------#
# read image
np.set_printoptions(threshold=np.inf)
# read image
img_t = []
for i in range (1, 4):
    img_t.append(plt.imread(f"in/Q5_{i}.tif"))
    show_fft(img_t[i-1])

# %% 
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
# ideal low pass filter
D = [10, 30, 60, 160, 460]
plt.figure()
for i in range(5):
    lpf = ideal_lpf(D[i], img_t[1])
    plt.subplot(2,3,i+1)
    plt.imshow(lpf, cmap='gray')
    plt.axis('off')
    plt.title(f"Ideal LPF D={D[i]}")
    
plt.show()

# %% 
# Gaussian low pass / high pass


