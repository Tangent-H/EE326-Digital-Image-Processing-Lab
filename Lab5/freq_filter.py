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

    Re = np.real(fshift)
    Re = (Re - Re.min()) / (Re.max() - Re.min()) * 255
    Im = np.imag(fshift)
    Im = (Im - Im.min()) / (Im.max() - Im.min()) * 255
    phase_spectrum = np.arctan2(Im, Re)
    phase_spectrum = (phase_spectrum - phase_spectrum.min()) / (phase_spectrum.max() - phase_spectrum.min()) * 255

    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Magnitude Spectrum')
    plt.subplot(2,2,2)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Phase Spectrum')
    plt.subplot(2,2,3)
    plt.imshow(Re, cmap='gray')
    plt.axis('off')
    plt.title('Real Spectrum')
    plt.subplot(2,2,4)
    plt.imshow(Im, cmap='gray')
    plt.axis('off')
    plt.title('Imaginary Spectrum')
    plt.show()
    return 
# %%
import numpy as np
import matplotlib.pyplot as plt
import cv2

def show_fft(img: np.ndarray):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    magnitude_spectrum = cv2.equalizeHist(magnitude_spectrum)

    Re = np.real(fshift)
    Re = cv2.normalize(Re, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Re = cv2.equalizeHist(Re)
    Im = np.imag(fshift)
    Im = cv2.normalize(Im, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    Im = cv2.equalizeHist(Im)

    phase_spectrum = np.arctan2(Im, Re)
    # 调整相位谱的值范围并归一化
    phase_spectrum = (phase_spectrum + np.pi) / (2 * np.pi) * 255
    phase_spectrum = phase_spectrum.astype(np.uint8)
    phase_spectrum = cv2.equalizeHist(phase_spectrum)

    plt.figure(figsize=(10, 8))
    plt.subplot(2,2,1)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Magnitude Spectrum')
    plt.subplot(2,2,2)
    plt.imshow(phase_spectrum, cmap='gray')
    plt.axis('off')
    plt.title('Phase Spectrum')
    plt.subplot(2,2,3)
    plt.imshow(Re, cmap='gray')
    plt.axis('off')
    plt.title('Real Spectrum')
    plt.subplot(2,2,4)
    plt.imshow(Im, cmap='gray')
    plt.axis('off')
    plt.title('Imaginary Spectrum')
    plt.show()
    return
# %%
def sobel_spatial(img: np.ndarray):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mask_y = conv2d(img, sobel_y, stride=1, padding=1)
    return mask_y

# %%
def pad_sobel(padding: int = 1):
    sobel_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    # add additional zeros along each axes
    sobel_y = np.vstack((np.zeros(sobel_y.shape[1]), sobel_y))
    sobel_y = np.hstack((np.zeros((sobel_y.shape[0],1)), sobel_y))
    return np.pad(sobel_y, padding, mode='constant', constant_values=0)

# %%
def sobel_freq(img: np.ndarray):
    pass
# %%
if __name__ == "__main__":
    # read image
    img_t = []
    for i in range (1, 4):
        img_t.append(plt.imread(f"in/Q5_{i}.tif"))
    
