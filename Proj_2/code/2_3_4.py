import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from align_image_code import align_images

def load_image(filename):
    img = plt.imread(filename) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img.astype(np.float32)

def save_image(image, filename, title=None):
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", filename)
    plt.imsave(path, np.clip(image, 0, 1))

def create_gaussian_kernel(size, sigma):
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

def gaussian_stack(image, levels, sigma=2):
    stack = [image]
    for _ in range(1, levels):
        blurred = cv2.GaussianBlur(stack[-1], (0, 0), sigma)
        stack.append(blurred)
    return stack

def laplacian_stack(image, levels, sigma=2):
    g_stack = gaussian_stack(image, levels, sigma)
    l_stack = []
    for i in range(levels - 1):
        l_stack.append(g_stack[i] - g_stack[i + 1])
    l_stack.append(g_stack[-1])
    return [lvl.astype(np.float32) for lvl in l_stack]

def visualize_stack_grid(stacks, titles, filename, normalize=True):
    n_levels = len(stacks[0])
    n_cols = len(stacks)
    fig, axes = plt.subplots(n_levels, n_cols, figsize=(3*n_cols, 2*n_levels))

    for col, (stack, title) in enumerate(zip(stacks, titles)):
        for row, img in enumerate(stack):
            if normalize:
                img_min, img_ptp = np.min(img), np.ptp(img)
                disp = (img - img_min) / (img_ptp + 1e-8) if img_ptp > 1e-8 else np.zeros_like(img)
            else:
                disp = img
            axes[row, col].imshow(disp)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title)

    plt.tight_layout()
    save_path = os.path.join("outputs", filename)
    plt.savefig(save_path, dpi=200)
    plt.close()

def reconstruct_from_laplacian(lap_pyr):
    current = lap_pyr[-1]
    for lvl in reversed(lap_pyr[:-1]):
        up = cv2.pyrUp(current)
        if up.shape[:2] != lvl.shape[:2]:
            up = cv2.resize(up, (lvl.shape[1], lvl.shape[0]))
        current = up + lvl
    return np.clip(current, 0, 1)

def create_smooth_mask(rows, cols, transition=150):
    mask = np.zeros((rows, cols, 3), dtype=np.float32)
    center = cols // 2
    left = center - transition // 2
    right = center + transition // 2
    if left < 0: left = 0
    if right > cols: right = cols
    ramp = np.linspace(1, 0, right - left)
    mask[:, :left, :] = 1
    if right > left:
        mask[:, left:right, :] = ramp[None, :, None]
    mask[:, right:, :] = 0
    return cv2.GaussianBlur(mask, (101, 101), 20)

def adjust_lighting(img_day, img_night):
    night_mean = np.mean(img_night)
    day_mean = np.mean(img_day)
    factor = day_mean / night_mean if night_mean > 0 else 1.0
    adjusted_night = img_night * factor
    return img_day, np.clip(adjusted_night, 0, 1)

def multires_blend(im1, im2, mask, levels=9, sigma=7):
    im1, im2 = adjust_lighting(im1, im2)
    
    lap1 = laplacian_stack(im1, levels, sigma)
    lap2 = laplacian_stack(im2, levels, sigma)
    gmask = gaussian_stack(mask, levels, sigma * 2)

    blended_lap = []
    for i in range(levels):
        tiger_weight = gmask[i] * 1.2
        lion_weight = (1 - gmask[i]) * 0.8
        blended = (tiger_weight * lap1[i] + lion_weight * lap2[i]) / (tiger_weight + lion_weight + 1e-8)
        blended_lap.append(blended)
    
    result = reconstruct_from_laplacian(blended_lap)
    result = cv2.GaussianBlur(result, (5, 5), 1.5)
    return np.clip(result, 0, 1), blended_lap, lap1, lap2, gmask

def create_personal_blends():
    try:
        day_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/building_day.jpg')
        night_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/building_night.jpg')
        
        day_aligned, night_aligned = align_images(day_img, night_img)
        
        target_size = (min(day_aligned.shape[1], night_aligned.shape[1]), 
                      min(day_aligned.shape[0], night_aligned.shape[0]))
        day_resized = cv2.resize(day_aligned, target_size)
        night_resized = cv2.resize(night_aligned, target_size)
        
        mask = create_smooth_mask(day_resized.shape[0], day_resized.shape[1], transition=200)

        blended, blend_stack, day_lap, night_lap, mask_stack = multires_blend(
            day_resized, night_resized, mask, levels=9, sigma=7
        )
        
        save_image(day_resized, 'personal_blend1_day.jpg', 'Building Day')
        save_image(night_resized, 'personal_blend1_night.jpg', 'Building Night') 
        save_image(blended, 'personal_blend1_result.jpg', 'Day/Night Blend')
        save_image(mask[:, :, 0], 'personal_blend1_mask.jpg', 'Vertical Mask')
        
        visualize_stack_grid(
            [day_lap, night_lap, mask_stack, blend_stack],
            ["Day Laplacian", "Night Laplacian", "Mask Stack", "Blended Stack"],
            "personal_blend1_process.jpg",
            normalize=True
        )

    except Exception as e:
        pass

    try:
        tiger_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/tiger.jpeg')
        lion_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/lionn.jpeg')
        
        if tiger_img is None or lion_img is None:
            raise ValueError("One or both images failed to load. Check file paths.")
    
        try:
            tiger_aligned, lion_aligned = align_images(tiger_img, lion_img)
        except Exception as align_error:
            target_size = (min(tiger_img.shape[1], lion_img.shape[1]), 
                          min(tiger_img.shape[0], lion_img.shape[0]))
            tiger_aligned = cv2.resize(tiger_img, target_size)
            lion_aligned = cv2.resize(lion_img, target_size)
        
        target_size = (min(tiger_aligned.shape[1], lion_aligned.shape[1]), 
                      min(tiger_aligned.shape[0], lion_aligned.shape[0]))
        tiger_resized = cv2.resize(tiger_aligned, target_size)
        lion_resized = cv2.resize(lion_aligned, target_size)
        
        h, w = tiger_resized.shape[:2]
        center = (w//2, h//3)
        radius = min(w, h) // 6
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        mask_circular = (dist_from_center <= radius).astype(np.float32)
        mask_smooth = cv2.GaussianBlur(mask_circular, (201, 201), 50)
        mask_smooth = np.stack([mask_smooth] * 3, axis=2)

        blended_tiger_lion, blend_stack_tl, lap1_tl, lap2_tl, mask_stack_tl = multires_blend(
            tiger_resized, lion_resized, mask_smooth, levels=6, sigma=3
        )
        
        save_image(tiger_resized, 'personal_blend3_tiger.jpg', 'Tiger Face')
        save_image(lion_resized, 'personal_blend3_lion.jpg', 'Lion Face')
        save_image(blended_tiger_lion, 'personal_blend3_result.jpg', 'Tiger-Lion Blend')
        save_image(mask_smooth[:, :, 0], 'personal_blend3_mask.jpg', 'Circular Mask')
        
        visualize_stack_grid(
            [lap1_tl, lap2_tl, mask_stack_tl, blend_stack_tl],
            ["Tiger Laplacian", "Lion Laplacian", "Mask Stack", "Blended Stack"],
            "personal_blend3_process.jpg",
            normalize=True
        )

    except Exception as e:
        pass

if __name__ == "__main__":
    apple = load_image("/Users/yuxuancai/cs180/cs180/Proj_2/code/images/apple.jpeg")
    orange = load_image("/Users/yuxuancai/cs180/cs180/Proj_2/code/images/orange.jpeg")
    levels = 9

    lap_apple = laplacian_stack(apple, levels, sigma=2)
    lap_orange = laplacian_stack(orange, levels, sigma=2)

    rows, cols, _ = apple.shape
    mask = create_smooth_mask(rows, cols, transition=200)
    gmask = gaussian_stack(mask, levels, sigma=10)

    visualize_stack_grid(
        [lap_apple, lap_orange, gmask],
        ["Apple Laplacian", "Orange Laplacian", "Gaussian Mask"],
        "part2_3_stacks.jpg",
        normalize=True
    )

    blended_img, _, _, _, _ = multires_blend(apple, orange, mask, levels=levels, sigma=7)
    save_image(blended_img, "part2_4_oraple.jpg", "Final Oraple Blend")

    selected_levels = [0, 2, 4]
    titles = ["High (g0)", "Mid (g1)", "Low (g2)"]

    fig, axes = plt.subplots(len(selected_levels), 3, figsize=(10, 6))

    for row, lvl in enumerate(selected_levels):
        apple_disp = (lap_apple[lvl] - np.min(lap_apple[lvl])) / (np.ptp(lap_apple[lvl]) + 1e-8)
        orange_disp = (lap_orange[lvl] - np.min(lap_orange[lvl])) / (np.ptp(lap_orange[lvl]) + 1e-8)
        avg_disp = 0.5 * (apple_disp + orange_disp)

        axes[row, 0].imshow(apple_disp)
        axes[row, 1].imshow(orange_disp)
        axes[row, 2].imshow(avg_disp)

        for col in range(3):
            axes[row, col].axis("off")
        axes[row, 0].set_ylabel(titles[row], fontsize=12)

    axes[0, 0].set_title("Apple")
    axes[0, 1].set_title("Orange")
    axes[0, 2].set_title("Average")

    plt.tight_layout()
    save_path = os.path.join("outputs", "part3_3_freqs.jpg")
    plt.savefig(save_path, dpi=200)
    plt.close()
    create_personal_blends()