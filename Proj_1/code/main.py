import numpy as np
import skimage.io as skio
import skimage
from skimage import transform, color, filters
import matplotlib.pyplot as plt
import sys
import os
import glob

def load_and_split_image(imgname):
    img = skio.imread(imgname)
    img = skimage.img_as_float(img)
    # 3d colored to 2d grayscale
    if len(img.shape) == 3:
        img = skimage.color.rgb2gray(img)
    height = img.shape[0] // 3
    # bgr each 1/3
    b = img[:height, :]
    g = img[height:2*height, :]
    r = img[2*height:3*height, :]
    return b, g, r

def crop_borders(img, border_percent=0.1):
    # avoid misalignment at the edges
    h, w = img.shape
    h_crop = int(h * border_percent)
    w_crop = int(w * border_percent)
    return img[h_crop:h-h_crop, w_crop:w-w_crop]

def shift_img_noncircular(img, dx, dy):
    """Shift image by (dx, dy) with zero padding. dx = cols (x), dy = rows (y)."""
    h, w = img.shape
    shifted = np.zeros_like(img)
    # Compute valid slice ranges
    x_start_src = max(0, -dx)
    x_end_src = min(w, w - dx)
    y_start_src = max(0, -dy)
    y_end_src = min(h, h - dy)
    x_start_dst = max(0, dx)
    x_end_dst = min(w, w + dx)
    y_start_dst = max(0, dy)
    y_end_dst = min(h, h + dy)
    shifted[y_start_dst:y_end_dst, x_start_dst:x_end_dst] = img[y_start_src:y_end_src, x_start_src:x_end_src]
    return shifted

def preprocess_for_alignment(img, use_edges=True):
    """Optional: Sobel edges + normalize (zero-mean, unit-var) for robustness to brightness."""
    if use_edges:
        img = filters.sobel(img)
    mu = np.mean(img)
    sigma = np.std(img) + 1e-8 # avoid div0
    return (img - mu) / sigma

def compute_ncc(img1, img2):
    """normalized cross correlation - higher is better"""
    # flatten to 1d arrays
    img1_flat = img1.flatten()
    img2_flat = img2.flatten()
    # normalize by magnitude to make it brightness invariant
    norm1 = np.linalg.norm(img1_flat)
    norm2 = np.linalg.norm(img2_flat)
    if norm1 == 0 or norm2 == 0:
        return 0
    img1_norm = img1_flat / norm1
    img2_norm = img2_flat / norm2
    return np.dot(img1_norm, img2_norm)

def compute_ssd(img1, img2):
    """sum of squared differences - lower is better"""
    return np.sum((img1 - img2) ** 2)

def find_best_alignment(cropped_ref, src, max_shift, metric, use_edges=True):
    """Search best (dx, dy) where dx = x shift, dy = y shift."""
    if metric == 'ncc':
        best_score = -float('inf')
        is_better = lambda new, best: new > best
    else: # ssd
        best_score = float('inf')
        is_better = lambda new, best: new < best
    best_dx, best_dy = 0, 0
    ref_pre = preprocess_for_alignment(cropped_ref, use_edges)

    for dy in range(-max_shift, max_shift + 1): # y shift (rows)
        for dx in range(-max_shift, max_shift + 1): # x shift (cols)
            shifted_full = shift_img_noncircular(src, dx, dy)
            shifted_cropped = crop_borders(shifted_full)
            shifted_pre = preprocess_for_alignment(shifted_cropped, use_edges)
            score = compute_ncc(ref_pre, shifted_pre) if metric == 'ncc' else compute_ssd(ref_pre, shifted_pre)

            if is_better(score, best_score):
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score

def align_single_scale(src, ref, max_shift=15, metric='ncc', use_edges=True):
    """Single-scale alignment using brute-force search (returns dx, dy)."""
    cropped_ref = crop_borders(ref)
    best_dx, best_dy, best_score = find_best_alignment(cropped_ref, src, max_shift, metric, use_edges)
    aligned = shift_img_noncircular(src, best_dx, best_dy)
    return aligned, (best_dx, best_dy), best_score

def build_pyramid(img, levels):
    """create image pyramid by downsampling"""
    pyramid = [img] # start with original
    # downsample by 2x each level
    for level in range(levels - 1):
        img_small = transform.rescale(pyramid[-1], 0.5, anti_aliasing=True, channel_axis=None)
        pyramid.append(img_small)
    return pyramid

def align_pyramid(src, ref, max_levels=4, max_shift_base=50, metric='ncc', use_edges=True):
    src_pyramid = build_pyramid(src, max_levels)
    ref_pyramid = build_pyramid(ref, max_levels)
    total_dx, total_dy = 0, 0
    for level in range(max_levels - 1, -1, -1):
        src_level = src_pyramid[level]
        ref_level = ref_pyramid[level]

        if level == max_levels - 1:
            max_shift = max_shift_base
            current_dx, current_dy = 0, 0
        else:
            max_shift = 2
            scale = 2
            current_dx, current_dy = total_dx * scale, total_dy * scale

        src_level = shift_img_noncircular(src_level, current_dx, current_dy)
        aligned_level, (dx, dy), _ = align_single_scale(src_level, ref_level, max_shift, metric, use_edges)

        if level == max_levels - 1:
            total_dx, total_dy = dx, dy
        else:
            total_dx = total_dx * 2 + dx
            total_dy = total_dy * 2 + dy

    aligned = shift_img_noncircular(src, total_dx, total_dy)
    return aligned, (total_dx, total_dy)

def auto_crop_borders(img):
    """crop borders automatically - just remove 5% from each side"""
    if len(img.shape) == 3:
        h, w = img.shape[:2]
    else:
        h, w = img.shape
    crop_top = crop_bottom = int(h * 0.05)
    crop_left = crop_right = int(w * 0.05)
    if len(img.shape) == 3:
        return img[crop_top:h-crop_bottom, crop_left:w-crop_right, :]
    return img[crop_top:h-crop_bottom, crop_left:w-crop_right]

def auto_contrast(img):
    """stretch contrast to use full 0-1 range"""
    img_min = np.min(img)
    img_max = np.max(img)
    if img_max > img_min:
        img_contrasted = (img - img_min) / (img_max - img_min)
    else:
        img_contrasted = img # avoid div by zero
    return np.clip(img_contrasted, 0, 1)

def combine_channels(r, g, b):
    """stack channels in rgb order"""
    return np.dstack([r, g, b])

def should_use_pyramid(img_shape, threshold=500):
    """decide if image is big enough to need pyramid"""
    return max(img_shape) > threshold

def process_image(imgname, use_pyramid=None, show_result=False, metric='ncc', use_edges=True):
    """main pipeline - load, align, combine, save"""
    print(f"Processing {imgname}...")

    # load and split
    b, g, r = load_and_split_image(imgname)

    if use_pyramid is None:
        use_pyramid = should_use_pyramid(b.shape)
    print(f"Image size: {b.shape}, Using pyramid: {use_pyramid}, Metric: {metric}, Edges: {use_edges}")

    # align channels to blue
    if use_pyramid:
        aligned_g, g_shift = align_pyramid(g, b, metric=metric, use_edges=use_edges)
        aligned_r, r_shift = align_pyramid(r, b, metric=metric, use_edges=use_edges)
    else:
        aligned_g, g_shift, _ = align_single_scale(g, b, metric=metric, use_edges=use_edges)
        aligned_r, r_shift, _ = align_single_scale(r, b, metric=metric, use_edges=use_edges)

    print(f"G:{g_shift}, R:{r_shift}")

    # combine into rgb image
    img_out = combine_channels(aligned_r, aligned_g, b)

    # post processing
    img_out = auto_crop_borders(img_out)
    img_out = auto_contrast(img_out)

    # save result
    output_name = f"result_{imgname.split('/')[-1].split('.')[0]}.jpg"
    img_out_save = np.clip(img_out, 0, 1) 
    img_out_save = (img_out_save * 255).astype(np.uint8) # Convert to uint8
    skio.imsave(output_name, img_out_save)
    print(f"Saved result to {output_name}")

    return img_out, g_shift, r_shift

if __name__ == "__main__":
    metric = 'ncc' # Or 'ssd'
    use_edges = True
    use_pyramid = None 

    # data directory (contains input images)
    data_dir = '/Users/yuxuancai/cs180/cs180/Proj_1/code/img'


    def list_images_in_dir(directory: str):
        patterns = ["*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.png"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(directory, pattern)))
        # Sort for deterministic order
        return sorted(files)

    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.lower() == 'all':
            image_list = list_images_in_dir(data_dir)
        else:
            image_list = [arg]
    else:
        image_list = list_images_in_dir(data_dir)

    if not image_list:
        print(f"No images found to process in: {data_dir}")
        sys.exit(1)

    print(f"Found {len(image_list)} image(s) to process.")
    for imgname in image_list:
        try:
            process_image(imgname, use_pyramid=use_pyramid, metric=metric, use_edges=use_edges)
        except Exception as e:
            print(f"Failed to process {imgname}: {str(e)}", file=sys.stderr)