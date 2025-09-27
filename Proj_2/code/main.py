import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import cv2
import time
import os

def pad_image(image, pad_h, pad_w, mode='constant', cval=0.0):
    if mode == 'constant':
        return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode, constant_values=cval)
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode=mode)

def conv2d_naive(image, kernel, padding=True, pad_mode='constant'):
    ih, iw = image.shape
    kh, kw = kernel.shape
    ph, pw = (kh // 2, kw // 2) if padding else (0, 0)
    padded = pad_image(image, ph, pw, mode=pad_mode, cval=0.0)
    out_h, out_w = (ih, iw) if padding else (ih - kh + 1, iw - kw + 1)
    out = np.zeros((out_h, out_w), dtype=np.float64)
    kflip = np.flip(kernel, axis=(0, 1))
    for i in range(out_h):
        for j in range(out_w):
            for m in range(kh):
                for n in range(kw):
                    out[i, j] += kflip[m, n] * padded[i + m, j + n]
    return out

def conv2d_two_loops(image, kernel, padding=True, pad_mode='constant'):
    ih, iw = image.shape
    kh, kw = kernel.shape
    ph, pw = (kh // 2, kw // 2) if padding else (0, 0)
    padded = pad_image(image, ph, pw, mode=pad_mode, cval=0.0)
    out_h, out_w = (ih, iw) if padding else (ih - kh + 1, iw - kw + 1)
    out = np.zeros((out_h, out_w), dtype=np.float64)
    kflip = np.flip(kernel, axis=(0, 1))
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(padded[i:i+kh, j:j+kw] * kflip)
    return out

def convolve_color(image, kernel, mode='same'):
    if image.ndim == 2:
        return convolve2d(image, kernel, mode=mode)
    h, w, c = image.shape
    out = np.zeros_like(image, dtype=np.float64)
    for ch in range(c):
        out[:, :, ch] = convolve2d(image[:, :, ch], kernel, mode=mode)
    return out

def save_image(image, filename, title=None, cmap='gray'):
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", filename)
    plt.figure(figsize=(5, 5))

    if image.ndim == 2:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(np.clip(image, 0, 1))
    plt.axis('off')
    if title:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()

img_path_selfie = '/Users/yuxuancai/cs180/cs180/Proj_2/code/images/selfie2.jpeg'
img = cv2.imread(img_path_selfie, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Ensure '{img_path_selfie}' exists.")
if len(img.shape) == 3:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img.astype(np.float64) / 255.0
img = cv2.resize(img, (500, 500))

box_kernel = np.ones((9, 9)) / 81.0
dx = np.array([[1, -1]])
dy = np.array([[1], [-1]])

start = time.time()
box_naive = conv2d_naive(img, box_kernel)
naive_time = time.time() - start

start = time.time()
box_two = conv2d_two_loops(img, box_kernel)
two_time = time.time() - start

start = time.time()
box_scipy = convolve2d(img, box_kernel, mode='same')
scipy_time = time.time() - start

dx_naive = conv2d_naive(img, dx)
dy_naive = conv2d_naive(img, dy)

save_image(img, 'part1_1_grayscale.jpg', 'Grayscale Selfie')
save_image(box_naive, 'part1_1_box_naive.jpg', 'Box (4 loops)')
save_image(box_two, 'part1_1_box_two_loops.jpg', 'Box (2 loops)')
save_image(box_scipy, 'part1_1_box_scipy.jpg', 'Box (SciPy)')
save_image(dx_naive, 'part1_1_dx.jpg', 'Dx (Horizontal Edges)')
save_image(dy_naive, 'part1_1_dy.jpg', 'Dy (Vertical Edges)')

img_path_cameraman = '/Users/yuxuancai/cs180/cs180/Proj_2/code/images/cameraman.png'
img2 = cv2.imread(img_path_cameraman, cv2.IMREAD_GRAYSCALE)
if img2 is None:
    raise FileNotFoundError(f"Ensure '{img_path_cameraman}' exists.")
if len(img2.shape) == 3:
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2 = img2.astype(np.float64) / 255.0

dx_img = convolve2d(img2, dx, mode='same')
dy_img = convolve2d(img2, dy, mode='same')
gradient_mag = np.sqrt(dx_img**2 + dy_img**2)
gradient_mag_disp = gradient_mag / (gradient_mag.max() + 1e-12)
edges = (gradient_mag_disp > 0.1).astype(np.float64)

save_image(img2, 'part1_2_cameraman.jpg', 'Cameraman')
save_image(dx_img, 'part1_2_dx.jpg', 'Dx')
save_image(dy_img, 'part1_2_dy.jpg', 'Dy')
save_image(edges, 'part1_2_edges.jpg', 'Binarized Edge')

ksize = 9
sigma = 2.0
gaussian_1d = cv2.getGaussianKernel(ksize, sigma)
gaussian_2d = gaussian_1d @ gaussian_1d.T

img_for_dog = img2
img_gaussian = convolve2d(img_for_dog, gaussian_2d, mode='same')
dx_gaussian = convolve2d(img_gaussian, dx, mode='same')
dy_gaussian = convolve2d(img_gaussian, dy, mode='same')
gradient_mag_gaussian = np.sqrt(dx_gaussian**2 + dy_gaussian**2)
gradient_mag_gaussian_disp = gradient_mag_gaussian / (gradient_mag_gaussian.max() + 1e-12)
edges_gaussian = (gradient_mag_gaussian_disp > 0.1).astype(np.float64)

dx_flipped = np.flip(dx)
dy_flipped = np.flip(dy)
dog_dx = convolve2d(gaussian_2d, dx_flipped, mode='same')
dog_dy = convolve2d(gaussian_2d, dy_flipped, mode='same')
dx_dog = convolve2d(img_for_dog, dog_dx, mode='same')
dy_dog = convolve2d(img_for_dog, dog_dy, mode='same')

orientation = np.arctan2(dy_gaussian, dx_gaussian)
orientation_normalized = (orientation + np.pi) / (2 * np.pi)
h, w = img_for_dog.shape
hsv = np.zeros((h, w, 3), dtype=np.float64)
hsv[:, :, 0] = orientation_normalized
hsv[:, :, 1] = 1.0
hsv[:, :, 2] = gradient_mag_gaussian_disp
hsv_uint8 = (hsv * 255).astype(np.uint8)
hsv_uint8[:, :, 0] = (hsv[:, :, 0] * 179).astype(np.uint8)
rgb_img = cv2.cvtColor(hsv_uint8, cv2.COLOR_HSV2RGB)
rgb_img = rgb_img.astype(np.float64) / 255.0

save_image(img_gaussian, 'part1_3_gaussian.jpg', 'Gaussian Blurred')
save_image(dx_gaussian, 'part1_3_dx_gaussian.jpg', 'Dx (Gaussian then Dx)')
save_image(dy_gaussian, 'part1_3_dy_gaussian.jpg', 'Dy (Gaussian then Dy)')
save_image(gradient_mag_gaussian_disp, 'part1_3_gradient_mag.jpg', 'Gradient Magnitude (Gaussian)')
save_image(edges_gaussian, 'part1_3_edges_gaussian.jpg', 'Binarized Edge (Gaussian)')
dog_dx_display = (dog_dx - dog_dx.min()) / (dog_dx.max() - dog_dx.min() + 1e-12)
dog_dy_display = (dog_dy - dog_dy.min()) / (dog_dy.max() - dog_dy.min() + 1e-12)
save_image(dog_dx_display, 'part1_3_dog_dx.jpg', 'DoG Dx Filter')
save_image(dog_dy_display, 'part1_3_dog_dy.jpg', 'DoG Dy Filter')
save_image(dx_dog, 'part1_3_dx_dog.jpg', 'Dx (DoG Filter)')
save_image(dy_dog, 'part1_3_dy_dog.jpg', 'Dy (DoG Filter)')
save_image(rgb_img, 'part1_3_hsv.jpg', 'Gradient Orientation (HSV)', cmap=None)

def sharpen_image_color(image, gaussian_kernel, alpha=1.0):
    if image.ndim == 2:
        blurred = convolve2d(image, gaussian_kernel, mode='same')
        high = image - blurred
        sharpened = image + alpha * high
        return blurred, high, np.clip(sharpened, 0, 1)
    else:
        blurred = convolve_color(image, gaussian_kernel, mode='same')
        high = image - blurred
        sharpened = image + alpha * high
        return blurred, high, np.clip(sharpened, 0, 1)

def create_unsharp_mask_kernel(gaussian_kernel, alpha=1.0):
    center = gaussian_kernel.shape[0] // 2
    impulse = np.zeros_like(gaussian_kernel)
    impulse[center, center] = 1.0
    unsharp_kernel = impulse + alpha * (impulse - gaussian_kernel)
    kernel_sum = unsharp_kernel.sum()
    if kernel_sum != 0:
        unsharp_kernel = unsharp_kernel / kernel_sum
    return unsharp_kernel

taj_path = '/Users/yuxuancai/cs180/cs180/Proj_2/code/images/taj.jpg'
taj_bgr = cv2.imread(taj_path, cv2.IMREAD_COLOR)
if taj_bgr is None:
    raise FileNotFoundError(f"Ensure '{taj_path}' exists.")
taj_rgb = cv2.cvtColor(taj_bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0

blur_sigma = 2.5
blur_size = 15
blur_1d = cv2.getGaussianKernel(blur_size, blur_sigma)
blur_2d = blur_1d @ blur_1d.T

sharpen_sigma = 0.5
sharpen_size = 11
alpha = 3.0
sharpen_1d = cv2.getGaussianKernel(sharpen_size, sharpen_sigma)
sharpen_2d = sharpen_1d @ sharpen_1d.T

blurred_component, high_freq_component, sharpened_multistep = sharpen_image_color(taj_rgb, sharpen_2d, alpha)

unsharp_kernel = create_unsharp_mask_kernel(sharpen_2d, alpha)
sharpened_single = np.zeros_like(taj_rgb)
for ch in range(3):
    sharpened_single[:, :, ch] = convolve2d(taj_rgb[:, :, ch], unsharp_kernel, mode='same')
sharpened_single = np.clip(sharpened_single, 0, 1)

diff = np.mean(np.abs(sharpened_multistep - sharpened_single))
evaluation_blurred = np.zeros_like(taj_rgb)
for ch in range(3):
    evaluation_blurred[:, :, ch] = convolve2d(taj_rgb[:, :, ch], blur_2d, mode='same')
evaluation_sharpened = np.zeros_like(taj_rgb)
for ch in range(3):
    evaluation_sharpened[:, :, ch] = convolve2d(evaluation_blurred[:, :, ch], unsharp_kernel, mode='same')
evaluation_sharpened = np.clip(evaluation_sharpened, 0, 1)

def calculate_metrics(original, processed):
    mse = np.mean((original - processed)**2)
    psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
    return mse, psnr

save_image(taj_rgb, 'part2_1_original_color.jpg', 'Original Taj (Color)', cmap=None)
save_image(evaluation_blurred, 'part2_1_blurry_color.jpg', 'Blurry Test Image (Color)', cmap=None)
high_freq_display = np.clip(high_freq_component + 0.5, 0, 1)
save_image(high_freq_display, 'part2_1_high_freq_color.jpg', 'High Frequencies (+0.5 offset)', cmap=None)
save_image(sharpened_multistep, 'part2_1_sharpened_color.jpg', 'Sharpened (Multi-step)', cmap=None)
unsharp_display = (unsharp_kernel - unsharp_kernel.min()) / (unsharp_kernel.max() - unsharp_kernel.min() + 1e-12)
save_image(unsharp_display, 'part2_1_unsharp_kernel.jpg', 'Unsharp Mask Kernel')

save_image(evaluation_blurred, 'part2_1_eval_blurred_color.jpg', 'Evaluation: Blurred (Color)', cmap=None)
save_image(evaluation_sharpened, 'part2_1_eval_sharpened_color.jpg', 'Evaluation: Sharpened (Color)', cmap=None)
difference_image = np.abs(taj_rgb - evaluation_sharpened)
difference_display = difference_image / (difference_image.max() + 1e-12)
save_image(difference_display, 'part2_1_difference_color.jpg', 'Difference: Original vs Sharpened', cmap=None)

def load_color_image(path, resize_to=None):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    if resize_to is not None:
        rgb = cv2.resize(rgb, resize_to)
    return rgb

YOUNG_ME_PATH = '/Users/yuxuancai/cs180/cs180/Proj_2/code/images/image3.JPG'  
CAMPANILE_PATH = '/Users/yuxuancai/cs180/cs180/Proj_2/code/images/image4.PNG'  

young_me = load_color_image(YOUNG_ME_PATH)
campanile = load_color_image(CAMPANILE_PATH)

if young_me is not None:
    young_me = cv2.resize(young_me, (taj_rgb.shape[1], taj_rgb.shape[0]))
    blurred_ym, high_ym, sharpened_ym = sharpen_image_color(young_me, sharpen_2d, alpha=alpha)
    save_image(young_me, 'image3_original.jpg', 'Original Image 3', cmap=None)
    save_image(sharpened_ym, 'image3_sharpened.jpg', 'Sharpened Image 3', cmap=None)
    
if campanile is not None:
    campanile = cv2.resize(campanile, (taj_rgb.shape[1], taj_rgb.shape[0]))
    blurred_camp = convolve_color(campanile, blur_2d, mode='same')
    sharpened_camp = np.zeros_like(campanile)
    for ch in range(3):
        sharpened_camp[:, :, ch] = convolve2d(blurred_camp[:, :, ch], unsharp_kernel, mode='same')
    sharpened_camp = np.clip(sharpened_camp, 0, 1)
    save_image(campanile, 'image4_original.jpg', 'Original Image 4', cmap=None)
    save_image(blurred_camp, 'image4_blurred.jpg', 'Blurred Image 4', cmap=None)
    save_image(sharpened_camp, 'image4_blur_then_sharpen.jpg', 'Blur then Sharpen Image 4', cmap=None)