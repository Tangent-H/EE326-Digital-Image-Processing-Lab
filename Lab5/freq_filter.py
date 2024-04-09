# %%
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
import datetime

# %%
def pd_print(arr: np.ndarray):
    df = pd.DataFrame(arr)
    print(df)
    return

# %%
def conv2d(img: np.ndarray, kernel: np.ndarray, stride: int = 0, padding: int = 0, flipping =True) -> np.ndarray:
    temp = np.pad(img, padding, mode='reflect')
    if flipping:
        kernel = np.flipud(np.fliplr(kernel)) #! flip the kernel up-down and left-right (according to the definition of convolution)
    out_img = np.zeros(img.shape)
    for i in range(0, img.shape[0], stride):
        for j in range(0, img.shape[1], stride):
            out_img[i, j] = np.sum(temp[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    return out_img
# %%
def show_fft(img: np.ndarray, return_specturm: bool = False, savefig=False):
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

    if savefig:
        t = datetime.datetime.now().strftime('%H-%M-%S')
        plt.savefig(f'out/{t}.png',bbox_inches='tight')
    plt.show()

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
    if return_specturm:
        return magnitude_spectrum
    return 
# %%
def sobel_spatial(img: np.ndarray, flipping = True):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    if flipping:
        mask_y = conv2d(img, sobel_y, stride=1, padding=1,flipping=True)
    else:
        mask_y = conv2d(img, sobel_y, stride=1, padding=1,flipping=False)
    return mask_y

# %%
def pad_sobel(padding_x: int = 1, padding_y: int = 1):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # add additional zeros along each axes
    # sobel_y = np.vstack((np.zeros(sobel_y.shape[1]), sobel_y))
    # sobel_y = np.hstack((np.zeros((sobel_y.shape[0],1)), sobel_y))
    sobel_y = np.pad(sobel_y, ((0, padding_x), \
                    (0, padding_y)), mode='constant', constant_values=0)
    '''np.pad(img, ((pad before first axis, pad after first axis),(pad before second axis, pad after second axis)),...)'''
    return sobel_y

# %%
def sobel_freq(img: np.ndarray):
    sobel = pad_sobel(int(img.shape[0]-1), int(img.shape[1])-1)
    img_pad = np.pad(img, ((0, 2), (0, 2)), mode='constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    # X_img = np.fft.fftshift(X_img)
    H_sobel = np.fft.fft2(sobel)
    # H_sobel = np.fft.fftshift(H_sobel)
    Y_img = X_img * H_sobel
    y_img = np.fft.ifft2(Y_img)
    y_img = np.real(y_img)
    y_img = y_img[0:-2, 0:-2]
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(sobel, cmap='gray')
    # plt.axis('off')
    plt.title('Sobel Filter')
    plt.subplot(1,2,2)
    plt.imshow(img_pad, cmap='gray')
    plt.axis('off')
    plt.title('Padded Image')
    plt.show()
    # for debugging
    show_fft(img_pad)
    show_fft(sobel)
    show_fft(y_img)
    
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
def gaussian_filter(D0: int, img: np.ndarray, order: int = 2, lowpass: bool = True):
    img_pad = np.pad(img, ((0,img.shape[0]), (0,img.shape[1])), mode='constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    X_img = np.fft.fftshift(X_img)

    xx, yy = np.ogrid[0:img_pad.shape[0], 0:img_pad.shape[1]]
    mid_x = img_pad.shape[0] // 2
    mid_y = img_pad.shape[1] // 2
    D2 = (xx - mid_x) ** 2 + (yy - mid_y) ** 2
    H_lpf = np.exp(-D2 / (2 * D0 ** 2))
    
    Y_img = np.zeros(X_img.shape)
    if lowpass:
        Y_img = X_img * H_lpf
    else:
        Y_img = X_img * (1 - H_lpf)
    Y_img = np.fft.ifftshift(Y_img)
    y_img = np.fft.ifft2(Y_img)
    y_img = np.real(y_img)
    y_img = y_img[0:img.shape[0], 0:img.shape[1]]
    return y_img

# %% 
def butter_notch(D0k: list, img: np.ndarray, uk: list, vk: list ,order: int = 4, scale = True):
    assert len(D0k) == len(uk) == len(vk), 'Invalid input. # of D0k, uk, vk should be the same.'
    img_pad = np.pad(img, ((0,img.shape[0]), (0,img.shape[1])), mode= 'constant', constant_values=0)
    X_img = np.fft.fft2(img_pad)
    X_img = np.fft.fftshift(X_img)

    uu,vv = np.ogrid[0:X_img.shape[0], 0:X_img.shape[1]]

    Dk = []
    D_k = []
    H = np.ones(X_img.shape)

    if scale:
        uk = [u * X_img.shape[0] for u in uk]
        vk = [v * X_img.shape[1] for v in vk]

    for k in range(len(D0k)):
        dk = np.sqrt((uu-uk[k])**2 + (vv -vk[k])**2)
        d_k = np.sqrt((uu-(X_img.shape[0]-uk[k]))**2 + (vv - (X_img.shape[1]-vk[k]))**2)
        H = H * (1/(1+(D0k[k]/dk)**(2*order))) * (1/(1+(D0k[k]/d_k)**(2*order)))
    
    # debugging
    plt.figure()
    plt.imshow(H, cmap='gray')
    plt.axis('off')
    plt.title('Butterworth Notch Filter')
    plt.savefig('out/butterworth_notch.png',bbox_inches='tight')
    plt.show()
    Y_img = X_img * H
    Y_img = np.fft.ifftshift(Y_img)
    y_img = np.fft.ifft2(Y_img)
    y_img = np.real(y_img)
    y_img = y_img[0:img.shape[0], 0:img.shape[1]]
    return y_img

# %% 
#-----------------Testing Section-----------------#
# read image
np.set_printoptions(threshold=np.inf)
# read image
img_t = []
for i in range (1, 4):
    img_t.append(plt.imread(f"in/Q5_{i}.tif"))
    show_fft(img_t[i-1],savefig=True)

# %% -----------------------------------------------------
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
plt.savefig('out/sobel_spatial_freq.png',bbox_inches='tight')
plt.show()

sobel_s_nonflip = sobel_spatial(img_t[0], flipping=False)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(sobel_s, cmap='gray')
plt.axis('off')
plt.title('Sobel Spatial')
plt.subplot(1,2,2)
plt.imshow(sobel_s_nonflip, cmap='gray')
plt.axis('off')
plt.title('Sobel Spatial Non-flip')
plt.savefig('out/sobel_spatial.png',bbox_inches='tight')
plt.show()



# %%---------------------------------------------------------
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

# %% -----------------------------------------------------
# Gaussian low pass / high pass
D0 = [30, 60, 160]
plt.figure()
for i in range(3):
    lpf = gaussian_filter(D0[i], img_t[1])
    plt.subplot(1,3,i+1)
    plt.imshow(lpf, cmap='gray')
    plt.axis('off')
    plt.title(f"GLPF D0={D0[i]}")
plt.show()

for i in range(3):
    lpf = gaussian_filter(D0[i], img_t[1], lowpass=False)
    plt.subplot(1,3,i+1)
    plt.imshow(lpf, cmap='gray')
    plt.axis('off')
    plt.title(f"GHPF D0={D0[i]}")
plt.show()

# %%----------------------------------------------------------
# Butterworth notch filter

Dk = [30, 30, 30, 30]
uk = [0.17, 0.33, 0.66, 0.84]
vk = [0.35, 0.35, 0.35, 0.35]

# Dk = [10]
# uk = [0.9]
# vk = [0.1]
# vk = [0.5-v for v in vk]
nrf = butter_notch(Dk, img_t[2], uk, vk, order=4)
plt.figure()
plt.imshow(nrf, cmap='gray')
plt.axis('off')
plt.title('NRF Result')
plt.savefig('out/nrf.png',bbox_inches='tight')
plt.show()
show_fft(nrf,savefig=True)

# %%
