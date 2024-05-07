# Image Restoration

# Abstract

This paper attempted different kinds of methods to restore images degraded to different extents. Firstly, we restore the images degraded only by noise. Then, we restore the images not only degraded by noise but also by a degradation function. We achieve this by modeling the degradation function mathematically. However, when degradation function and noises are combined, the restoration result is usually not satisfactory.

Key: image restoration, degradation function, noise reduction

# Introduction

Image restoration is a crucial field in image processing focused on reconstructing images corrupted by noise or other distortions. Key techniques include median, adaptive median, adaptive mean, and alpha-trimmed mean filtering, each tailored to reduce specific types of noise while preserving important image details. Additionally, methods like Wiener filtering and constrained least squared filtering address the restoration by modeling and inverting the degradation process, aiming to achieve a balance between noise reduction and detail preservation. These approaches are fundamental in enhancing image quality across various applications such as medical imaging, astronomy, and forensic analysis. In this lab, we utilized the above methods in order to restore several types of degraded images and analyze the effect of each filtering technique in detail. The main experiment results are listed as follows:

1. For images only distorted by salt and pepper noise, a median filter and adaptive median filter can restore them.
2. For images not only distorted by salt and pepper noise but also Gaussian noise, an adaptive median filter followed by an adaptive mean filter can restore them.
3. For images that undergo atmospheric turbulence or motion blur, we may estimate the degradation function first, then use full inverse filtering (add a tiny constant), Wiener filtering or constraint least square filtering to restore them.

# Question Formulation

For different types of degraded images, we should adopt different strategies to restore them, as discussed below.

## Impulse Noise (Salt and Pepper Noise)

Note that this type of noise 