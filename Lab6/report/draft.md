# Image Restoration

# Abstract

This paper attempted different kinds of methods to restore images degraded to different extents. Firstly, we restore the images that are degraded only by noise. Then, we restore the images not only degraded by noise but also by a degradation function. We achieve this by modeling the degradation function mathematically. However, when degradation function and noises are combined, the restoration result is usually not satisfactory.

Key: image restoration, degradation function, noise reduction

# Introduction

Image restoration is a crucial field in image processing focused on reconstructing images corrupted by noise or other distortions. Key techniques include median, adaptive median, adaptive mean, and alpha-trimmed mean filtering, each tailored to reduce specific types of noise while preserving important image details. Additionally, methods like Wiener filtering and constrained least squared filtering address the restoration by modeling and inverting the degradation process, aiming to achieve a balance between noise reduction and detail preservation. These approaches are fundamental in enhancing image quality across various applications such as medical imaging, astronomy, and forensic analysis. 

In this lab, we utilized the above methods in order to restore several types of degraded images and analyze the effect of each filtering technique in detail. The main experiment results are listed as follows:

1. For images only distorted by salt and pepper noise, a median filter and adaptive median filter can restore them.
2. For images not only distorted by salt and pepper noise but also Gaussian noise, an adaptive median filter followed by an adaptive mean filter can restore them.
3. For images that undergo atmospheric turbulence or motion blur, we may estimate the degradation function first, then use full inverse filtering (add a tiny constant), Wiener filtering, or constraint least square filtering to restore them.

# Question Formulation

We should adopt different strategies to restore different types of degraded images, as discussed below.

## Salt and Pepper Noise (Impulse Noise)

Salt and pepper noise, also known as impulse noise, is a type of noise where some of the pixels in an image randomly turn black or white, making it appear as if the image has been sprinkled with salt and pepper. This noise can drastically disrupt the visual quality of images, presenting a challenge in image processing.

To address this, median and adaptive median filters are particularly effective due to their ability to preserve edges while removing noise. Here’s a more detailed mathematical explanation:

1. **Median Filter:**
   - The median filter works by moving a window (kernel) over each pixel of the image, considering the pixel itself and its immediate neighbors.
   - It then replaces the value of the current pixel with the median of the intensity values within that window.
   - Mathematically, if the window has an odd number of pixels (say 3x3, 5x5), the median is the middle value when all the intensities are sorted.
   - This is effective against salt and pepper noise because the extreme values (0 or 255 for black and white, respectively) caused by the noise do not affect the median as much unless a majority of the pixels in the window are corrupted, which is statistically unlikely.
2. **Adaptive Median Filter:**
   - The adaptive median filter extends the idea of the median filter by adjusting the size of the window based on the local variance of pixel values.
   - This filter starts with a small window and increases its size until the condition that the median minus the minimum and median plus the maximum within the window are not equal is met. This helps in distinguishing between noise and fine details.
   - This method is more robust in maintaining details while effectively removing noise, as it adapts to the 'noise level' of different parts of the image.

## Gaussian Noise

Gaussian noise, a common type of noise encountered in image processing, resembles the statistical properties of Gaussian (normal) distributions. This noise is characterized by adding to each pixel in the image a random value selected from a Gaussian distribution. Unlike salt and pepper noise, which presents extreme values at random pixels, Gaussian noise affects all pixels with small variations centered around zero and with some standard deviation.

The distribution density of the Gaussian noise is
$$
G(x,y)=\frac1{2\pi\sigma^2}\exp\left(-\frac{x^2+y^2}{2\sigma^2}\right)
$$
Note that Gaussian noise is single-peak; normally, we can use all sorts of mean filters to remove it.

However, this method usually blurs the image since not only is the noise removed, but the texture in the image is also treated as noise and blurred.

So, another method known as adaptive mean filter is used. The formula for the adaptive mean filter is shown below:
$$
\hat{f}(x,y)=g(x,y)-\frac{\sigma_\eta^2}{\sigma_L^2}[g(x,y)-m_L]
$$
where

- $\hat{f}(x,y)$: This represents the output pixel value at coordinates (𝑥,𝑦)(*x*,*y*) after applying the adaptive mean filter. It's the filtered version of the original pixel value.
- $g(x,y)$: This is the original pixel value at coordinates (𝑥,𝑦)(*x*,*y*) in the image before filtering. It represents the intensity of the pixel that is subject to noise and other image distortions.
- $\sigma_\eta^2$: This symbol denotes the variance of the noise present in the image. It's a measure of how spread out the noise distribution is around its mean (usually zero). This value is crucial for adjusting the strength of the filter based on the noise level.
- $\sigma_L^2$: This represents the local variance around the pixel at (𝑥,𝑦)(*x*,*y*). It measures how much the pixel values in a local neighborhood vary, which helps in identifying areas with significant image details or edges.
- $m_L$: This is the local mean around the pixel at (𝑥,𝑦)(*x*,*y*). It calculates the average pixel value in a neighborhood, providing a baseline from which deviations due to noise can be identified and adjusted.

However, using this method requires that the image has a roughly “flat” region, where the pixels’ value of the image does not vary, and in this area, we can estimate the variance of the noise.

## With Degradation Function

Most of the time, the image is distorted not only by noise but also by some degradation functions, like atmospheric turbulence and motion blur.

In this situation, we need to first model the degradation function, transfer it into the frequency domain, and then apply full inverse filtering, Wiener filtering, or constraint least square filtering.

### Modeling

#### Atmospheric Turbulence

The mathematic expression for atmospheric turbulence in the frequency domain is
$$
H(\mu,\nu)=e^{-k(\mu^2+\nu^2)^{5/6}}
$$
where $k$ is a constant that depends on the nature of the turbulence.

#### Motion Blur

Motion blurring can be viewed as the effect of pixel integration along the motion path. Thus the formulation of the motion blur filter is
$$
\begin{aligned}
g(x, y) & =\int_0^T f\left[x-x_0(t), y-y_0(t)\right] d t \\
G(\mu, v) & =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} g(x, y) e^{-j 2 \pi(\mu x+v y)} d x d y \\
& =\int_{-\infty}^{\infty} \int_{-\infty}^{\infty}\left[\int_0^T f\left[x-x_0(t), y-y_0(t)\right] d t\right] e^{-j 2 \pi(\mu x+v y)} d x d y \\
& =\int_0^T\left[\int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f\left[x-x_0(t), y-y_0(t)\right] e^{-j 2 \pi(\mu x+v y)} d x d y\right] d t \\
& =\int_0^T F(u, v) e^{-j 2 \pi\left[\mu x_0(t)+v y_0(t)\right]} d t \\
& =F(u, v) \int_0^T e^{-j 2 \pi\left[\mu x_0(t)+v y_0(t)\right]} d t
\end{aligned}
$$
Here, the integration $ \int_0^T e^{-j 2 \pi\left[\mu x_0(t)+v y_0(t)\right]} d t$​ is the motion blur filter we desire to derive.

If we model the motion in $x$ and $y$ direction as uniform linear motion as follows:
$$
x_0(t)=at/T\\
y_0(t)=bt/T
$$
Then we can calculate the integration as follows:
$$
\begin{aligned}
H(\mu,\nu)&=\int_0^Te^{-j2\pi[\mu x_0(t)+\nu y_0(t)]}dt \\
&=\int_0^Te^{-j2\pi[\mu a+\nu b]t/T}dt \\
&=\frac{T}{\pi(\mu a+\nu b)}\sin[\pi(\mu a+\nu b)]e^{-j\pi(\mu a+\nu b)} 
\end{aligned}
$$
And this is the desired motion blur filter.

### Filtering

There are generally three types of filters that can handle images undergoing degradation: full inverse filter, Wiener filter, and constraint least square filter.

#### Full Inverse Filter

The full inverse filter is based on the degradation model where the observed image $g(x, y)$ is the convolution of the true image $f(x, y)$ with a blur kernel $h(x, y)$, corrupted by additive noise $n(x, y)$: 𝑔(𝑥,𝑦)=ℎ(𝑥,𝑦)∗𝑓(𝑥,𝑦)+𝑛(𝑥,𝑦)*g*(*x*,*y*)=*h*(*x*,*y*)∗*f*(*x*,*y*)+*n*(*x*,*y*) In the frequency domain, this equation becomes: 𝐺(𝑢,𝑣)=𝐻(𝑢,𝑣)𝐹(𝑢,𝑣)+𝑁(𝑢,𝑣)*G*(*u*,*v*)=*H*(*u*,*v*)*F*(*u*,*v*)+*N*(*u*,*v*) The full inverse filter attempts to recover $F(u, v)$ by dividing $G(u, v)$ by $H(u, v)$: 𝐹^(𝑢,𝑣)=𝐺(𝑢,𝑣)𝐻(𝑢,𝑣)*F*^(*u*,*v*)=*H*(*u*,*v*)*G*(*u*,*v*) This assumes that $H(u, v) \neq 0$ and does not account for the noise, which can make the solution unstable or prone to amplifying noise, particularly where $H(u, v)$ is small.