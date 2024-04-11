# %%
import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image

# %%
def gamma(image, gamma):
    # 0-255 -> 0-1
    image_normalized = image.astype(np.float64) / 255.0
    image_gamma = image_normalized ** gamma
    # 0-1 -> 0-255
    image_gamma_out = (image_gamma * 255).astype(np.uint8)
    return image_gamma_out
# %%
def normalize(image):
    image_enhanced = image.copy()
    image_enhanced = image_enhanced - np.min(image_enhanced)
    image_enhanced = image_enhanced / np.max(image_enhanced) * 255
    image_enhanced = image_enhanced.astype(np.uint8)
    return image_enhanced
# %% for comparison
def hist_equ(input_image):
    # number of pixel
    N = input_image.shape[0] * input_image.shape[1]
    # histogram
    input_hist, _ = np.histogram(input_image.flatten(), 256, [0, 256])
    input_hist = input_hist/N
    # cumulative histogram
    cdf = input_hist.cumsum() #implicitly converted to float64
    # histogram equalization
    output_image = np.zeros(input_image.shape, dtype=input_image.dtype)
    output_image = 255 * cdf[input_image]
    output_image = output_image.astype(np.uint8)
    output_hist, _ = np.histogram(output_image.flatten(), 256, [0, 256])
    output_hist = output_hist/N
    return (output_image, output_hist, input_hist)

# %%
def binarize(image, threshold):
    image_binarized = image.copy()
    image_binarized[image_binarized < threshold] = 0
    image_binarized[image_binarized >= threshold] = 255
    return image_binarized

# %%

# %%-----------------------------------------------------------------------
fig1  = plt.imread('in/Figure1.tif')
Image.fromarray(fig1).save('out/Figure1.jpg')
fig1_enhanced_gamma = gamma(fig1, 3)
Image.fromarray(fig1_enhanced_gamma).save('out/Figure1_enhanced_gamma.jpg')
fig1_enhanced_hist = hist_equ(fig1)
Image.fromarray(fig1_enhanced_hist[0]).save('out/Figure1_enhanced_hist.jpg')
fig1_enhanced_norm = normalize(fig1)
Image.fromarray(fig1_enhanced_norm).save('out/Figure1_enhanced_norm.jpg')

# %%-----------------------------------------------------------------------
fig2 = plt.imread('in/Figure2.tif')
Image.fromarray(fig2).save('out/Figure2.jpg')
fig2_binaraized = binarize(fig2, 50)
fig2_gamma = gamma(fig2, 0.1)
Image.fromarray(fig2_binaraized).save('out/Figure2_binaraized.jpg')
Image.fromarray(fig2_gamma).save('out/Figure2_gamma.jpg')
