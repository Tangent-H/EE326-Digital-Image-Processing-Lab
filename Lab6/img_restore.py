# %%
import numpy as np
from matplotlib import pyplot as plt
import PIL.Image as Image
from utils import *

# %%
def median_filter(input_image, n_size):
    '''
    flipping the image at the edges would produce better results
    '''
    # padding, and ensure n_size is odd
    if n_size % 2 == 0:
        n_size += 1
    temp = np.pad(input_image, n_size//2, mode='reflect')

    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            output_image[i, j] = np.median(temp[i:i+n_size, j:j+n_size])
    return output_image

# %%
def adaptive_median_filter(input_image, size_max):
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    temp = np.pad(input_image, size_max//2, mode='reflect')
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            size = 3
            while size <= size_max:
                # get the window
                offset = size_max//2 - size//2
                window = temp[i+offset:i+offset+size, j+offset:j+offset+size]
                # calculate the params
                zmin = np.min(window)
                zmax = np.max(window)
                zmed = np.median(window)
                zxy = input_image[i, j].astype(np.int32)

                A1 = zmed - zmin
                A2 = zmed - zmax
                # check if the median pixel is impulse noise
                if A1 > 0 and A2 < 0:
                    B1 = zxy - zmin
                    B2 = zxy - zmax
                    if B1 > 0 and B2 < 0:
                        output_image[i, j] = zxy
                        break
                    else:
                        output_image[i, j] = zmed
                        break
                else:
                    size += 2
                    continue
    return output_image

# %%
def alpha_trimmed_mean_filter(input_image, n_size, d):
    '''
    flipping the image at the edges would produce better results
    '''
    temp = np.pad(input_image, n_size//2, mode='reflect')

    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)

    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            window = temp[i:i+n_size, j:j+n_size]
            window = np.sort(window.flatten())
            window = window[d//2:-d//2]
            output_image[i, j] = np.mean(window)
    return output_image

# %%
def adaptive_mean_filter(input_image, noise_var, n_size):
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    temp = np.pad(input_image, n_size//2, mode='reflect')
    for i in range(input_image.shape[0]):
        for j in range(input_image.shape[1]):
            window = temp[i:i+n_size, j:j+n_size]
            window_var = np.var(window)
            output_image[i, j] = input_image[i, j] - (noise_var/window_var) * (input_image[i, j] - np.mean(window))
    return output_image

# %%
def butter_freq_lpf(img_fft, D0, n):
    xx, yy = np.ogrid[0:img_fft.shape[0], 0:img_fft.shape[1]]
    mid_x = img_fft.shape[0] / 2
    mid_y = img_fft.shape[1] / 2
    D = np.sqrt((xx - mid_x) ** 2 + (yy - mid_y) ** 2)
    H_lpf = 1/(1+(D/D0)**(2*n))
    Y_img = img_fft * H_lpf
    return Y_img
# %%
def wiener(img_fft, H, K):
    H_abs_2 = np.abs(H)**2
    Y_img = (H_abs_2/H/(H_abs_2+K)) * img_fft
    return Y_img

# %%
# def get_motion_blur_filter(img_shape, a, b, T):
#     epsilon = 1e-2
#     uu, vv = np.ogrid[0:img_shape[0], 0:img_shape[1]]
#     return T/(np.pi*(uu*a+vv*b+epsilon))*np.sin(np.pi*(uu*a+vv*b))*np.exp(-1j*np.pi*(uu*a+vv*b))

def get_motion_blur_filter(img_shape, a, b, T, epsilon=1e-10):
    uu, vv = np.ogrid[0:img_shape[0], 0:img_shape[1]]
    D = uu * a + vv * b
    print(D)
    H = (T / (np.pi * (D + epsilon))) * np.sin(np.pi * D) * np.exp(-1j * np.pi * D)
    return H
#%% ---------------------------------------------------------
# read all the images from the folder
folder = 'in'
images = read_images(folder)
for name, img in images.items():
    plt.figure()
    print(name)
    plt.imshow(img,cmap='gray')
    plt.show()

#%% ---------------------------------------------------------
# apply median filter
test_image = ['Q6_1_1', 'Q6_1_2', 'Q6_1_3', 'Q6_1_4']
for name in test_image:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[name], cmap='gray',vmin=0,vmax=255)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    img = images[name]
    img_filtered = median_filter(median_filter(img, 3),3)
    plt.imshow(img_filtered, cmap='gray',vmin=0,vmax=255)
    plt.title('Median Filtered')
    plt.axis('off')
    plt.savefig(f'out/{name}_median.png', bbox_inches='tight')
    plt.show()

#%% ---------------------------------------------------------
# apply adaptive median filter
test_image = ['Q6_1_3', 'Q6_1_4']
for name in test_image:
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(images[name], cmap='gray',vmin=0,vmax=255)
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    img = images[name]
    img_filtered = adaptive_median_filter(img, 11)
    plt.imshow(img_filtered, cmap='gray',vmin=0,vmax=255)
    plt.title('Adaptive Median Filtered')
    plt.axis('off')
    plt.savefig(f'out/{name}_adaptive_median.png', bbox_inches='tight')
    plt.show()
# %% ---------------------------------------------------------
# according to the ROI pixel values ditribution, the noise is SAP and uniform
# we first use the adaptive median filter to remove the impulse noise
# then we use the adaptive mean filter to remove the uniform noise
test_image = "Q6_1_4"
img = images[test_image]
img_ad = adaptive_median_filter(img,11)
img_alpha = alpha_trimmed_mean_filter(img_ad, 3, 4)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_alpha, cmap='gray',vmin=0,vmax=255)
plt.title('Alpha Trimmed Mean Filtered')
plt.axis('off')
plt.savefig(f'out/{test_image}_alpha_trimmed_mean.png', bbox_inches='tight')
# %% ---------------------------------------------------------
# get the local intensity distribution to estimate the noise type
roi_index = select_roi(img_ad)    # press ENTER to close the window
print(roi_index)
roi = img_ad[roi_index[1]:roi_index[3], roi_index[0]:roi_index[2]]
plt.figure()
plt.imshow(roi, cmap='gray',vmin=0,vmax=255)
plt.title('ROI')
plt.axis('off')
plt.savefig(f'out/{test_image}_roi.png', bbox_inches='tight')
plt.show()
show_distr(roi.flatten())
# estimate the noise variance
var_noise = np.var(roi)
print(f"Estimated noise variance: {var_noise}")
img_mean = adaptive_mean_filter(img_ad, var_noise, 9)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_mean, cmap='gray',vmin=0,vmax=255)
plt.title('Adaptive Mean Filtered')
plt.axis('off')
plt.savefig(f'out/{test_image}_adaptive_mean.png', bbox_inches='tight')
plt.show()
# %% ---------------------------------------------------------
# Full inverse filtering
# note that the degration filter define its origin at the filter center
test_image = 'Q6_2'
img = images[test_image]
print(f"Max: {np.max(img)}, Min: {np.min(img)}")
img_fft = np.fft.fftshift(np.fft.fft2(img))
H = np.zeros(img_fft.shape, dtype=img_fft.dtype)
u_center = img_fft.shape[0]/2
v_center = img_fft.shape[1]/2
print(f"Center: ({u_center}, {v_center})")
for u in range(img_fft.shape[0]):
    for v in range(img_fft.shape[1]):
        H[u, v] = np.exp(-0.0025*((u-u_center)**2+(v-v_center)**2)**(5/6))
plt.figure()
plt.imshow(np.real(H), cmap='gray')
plt.title('Degradation Filter')
plt.axis('off')
plt.savefig(f'out/{test_image}_degradation.png', bbox_inches='tight')
plt.show()

img_fft_restore = np.fft.ifftshift(img_fft / (H + 1e-2))
img_restore = np.real(np.fft.ifft2(img_fft_restore))
print(f"Max: {np.max(img_restore)}, Min: {np.min(img_restore)}")
# img_restore = img_restore.astype(np.uint8)
img_restore = np.clip(img_restore, 0, 255).astype(np.uint8)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_restore, cmap='gray',vmin=0,vmax=255)
plt.title('Restored')
plt.axis('off')
plt.savefig(f'out/{test_image}_restore.png', bbox_inches='tight')
# %%
# Full inverse filtering + Butterworth low-pass filter
plt.figure()
for i in range(6):
    img_fft_restore_bt = butter_freq_lpf(img_fft / H,30 + 10*i, 10)
    img_restore_bt = np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_restore_bt)))
    print(f"Max: {np.max(img_restore_bt)}, Min: {np.min(img_restore_bt)}")
    img_restore_bt = np.clip(img_restore_bt, 0, 255).astype(np.uint8)
    plt.subplot(2, 3, i+1)
    plt.imshow(img_restore_bt, cmap='gray',vmin=0,vmax=255)
    plt.title(f'D0={30+10*i}')
    plt.axis('off')
plt.savefig(f'out/{test_image}_restore_butterworth.png', bbox_inches='tight')
plt.show()
# %% ---------------------------------------------------------
# Wiener filtering
plt.figure()
img_wieners = []
for i in range(6):
    K = 1e-4*i
    img_fft_wiener = wiener(img_fft, H, K)
    img_wieners.append(np.real(np.fft.ifft2(np.fft.ifftshift(img_fft_wiener))))
    print(f"Max: {np.max(img_wieners[i])}, Min: {np.min(img_wieners[i])}")
    img_wiener = np.clip(img_wieners[i], 0, 255).astype(np.uint8)
    plt.subplot(2, 3, i+1)
    plt.imshow(img_wiener, cmap='gray',vmin=0,vmax=255)
    plt.title(f'K={K:.4f}')
    plt.axis('off')
    print(f"K={K}")
plt.savefig(f'out/{test_image}_restore_wiener.png', bbox_inches='tight')
plt.show()

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_restore, cmap='gray',vmin=0,vmax=255)
plt.title('Full Inverse Filtering plus 0.01')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_wieners[2], cmap='gray',vmin=0,vmax=255)
plt.title('Wiener Filtering 0.0003')
plt.axis('off')
plt.savefig(f'out/{test_image}_compare.png', bbox_inches='tight')
plt.show()



# %% ---------------------------------------------------------
# Restoring motion-blurred image
# Failed
test_image = ['Q6_1_1','Q6_3_1', 'Q6_3_2', 'Q6_3_3']

vx = 1
vy = 1
delta_t = 1e-2
step = 1/delta_t

uu, vv = np.ogrid[0:images[test_image[0]].shape[0], 0:images[test_image[0]].shape[1]]
print(uu.shape, vv.shape)

M = uu * vx + vv * vy
M = M[np.newaxis, :, :]
t = np.arange(0, 1, delta_t)
t = t[:, np.newaxis, np.newaxis]
H_mb = np.sum(np.exp(-1j*2*np.pi*M*t) * delta_t,axis=0)
print(H_mb.shape)

img = images['Q6_1_1']
img_fft = np.fft.fft2(img)
img_blur = np.real(np.fft.ifft2(img_fft * H_mb))
img_blur = np.clip(img_blur, 0, 255).astype(np.uint8)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_blur, cmap='gray',vmin=0,vmax=255)
plt.title('Motion Blurred')
plt.axis('off')
plt.savefig(f'out/{test_image[0]}_motion_blur.png', bbox_inches='tight')
plt.show()

# %% ---------------------------------------------------------
test_image = ['Q6_1_1','Q6_3_1', 'Q6_3_2', 'Q6_3_3']
H_mb = get_motion_blur_filter(images[test_image[1]].shape, 0.1, 0.1, 1)
plt.figure()
plt.imshow(np.real(H_mb), cmap='gray')
plt.title('Motion Blur Filter')
plt.axis('off')
plt.show()

img = test_image[1]
img = images[img]
img_fft = np.fft.fftshift(np.fft.fft2(img))
img_mb_fft = img_fft / H_mb
img_mb = np.real(np.fft.ifft2(np.fft.ifftshift(img_mb_fft)))
img_mb = np.clip(img_mb, 0, 255).astype(np.uint8)
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray',vmin=0,vmax=255)
plt.title('Original')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(img_mb, cmap='gray',vmin=0,vmax=255)
plt.title('Motion Blurred')
plt.axis('off')
plt.savefig(f'out/{test_image[0]}_motion_blur.png', bbox_inches='tight')
plt.show()


