import numpy as np
import matplotlib.pyplot as plt
import cv2
from align_image_code import align_images

def load_image(filename):
    img = plt.imread(filename) / 255.0
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    return img

def create_gaussian_kernel(size, sigma):
    return cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T

def hybrid_image(im1, im2, sigma1, sigma2, hf_gain=1.5, lf_gain=1.0):
    ksize = max(15, int(6 * max(sigma1, sigma2) + 1))
    if ksize % 2 == 0:
        ksize += 1

    gaussian1_2d = create_gaussian_kernel(ksize, sigma1)
    gaussian2_2d = create_gaussian_kernel(ksize, sigma2)

    hybrid = np.zeros_like(im1)
    for c in range(3):
        low_freq = cv2.filter2D(im2[:, :, c], -1, gaussian2_2d, borderType=cv2.BORDER_REFLECT) * lf_gain
        high_freq = (im1[:, :, c] - cv2.filter2D(im1[:, :, c], -1, gaussian1_2d, borderType=cv2.BORDER_REFLECT)) * hf_gain
        hybrid[:, :, c] = np.clip(low_freq + high_freq, 0, 1)
    return hybrid

def gaussian_pyramid(image, levels):
    pyramid = [image]
    current = image.copy()
    for _ in range(levels - 1):
        gaussian_2d = create_gaussian_kernel(5, 1.0)
        current = cv2.filter2D(current, -1, gaussian_2d, borderType=cv2.BORDER_REFLECT)
        current = current[::2, ::2, :]
        pyramid.append(current)
    return pyramid

def save_image(image, filename, title):
    plt.imsave(filename, image)
    plt.close()

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
        lap = g_stack[i] - g_stack[i + 1]
        l_stack.append(lap)
    l_stack.append(g_stack[-1])
    return l_stack

def visualize_stack_grid(stacks, titles, filename, normalize=True):
    n_levels = len(stacks[0])
    n_cols = len(stacks)
    fig, axes = plt.subplots(n_levels, n_cols, figsize=(3*n_cols, 2*n_levels))

    if n_levels == 1:
        axes = np.expand_dims(axes, 0)

    for col, (stack, title) in enumerate(zip(stacks, titles)):
        for row, img in enumerate(stack):
            if normalize:
                img_min, img_ptp = np.min(img), np.ptp(img)
                if img_ptp < 1e-8:
                    disp = np.zeros_like(img)
                else:
                    disp = (img - img_min) / (img_ptp + 1e-8)
            else:
                disp = img
            axes[row, col].imshow(disp)
            axes[row, col].axis("off")
            if row == 0:
                axes[row, col].set_title(title)

    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def create_complete_pipeline_visualization(im1, im2, im1_aligned, im2_aligned, sigma1, sigma2, hybrid, filename_prefix="pipeline"):
    ksize1 = max(15, int(6 * sigma1 + 1))
    ksize2 = max(15, int(6 * sigma2 + 1))
    if ksize1 % 2 == 0: ksize1 += 1
    if ksize2 % 2 == 0: ksize2 += 1
    
    gaussian1_2d = create_gaussian_kernel(ksize1, sigma1)
    gaussian2_2d = create_gaussian_kernel(ksize2, sigma2)
    
    low_freq_img = np.zeros_like(im2_aligned)
    for c in range(3):
        low_freq_img[:, :, c] = cv2.filter2D(im2_aligned[:, :, c], -1, gaussian2_2d, borderType=cv2.BORDER_REFLECT)
    
    high_freq_img = np.zeros_like(im1_aligned)
    for c in range(3):
        blurred = cv2.filter2D(im1_aligned[:, :, c], -1, gaussian1_2d, borderType=cv2.BORDER_REFLECT)
        high_freq_img[:, :, c] = im1_aligned[:, :, c] - blurred
    
    high_freq_display = np.clip(high_freq_img + 0.5, 0, 1)
    
    freq_im1 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned[:, :, 0]))) + 1e-10)
    freq_im2 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned[:, :, 0]))) + 1e-10)
    freq_low = np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_freq_img[:, :, 0]))) + 1e-10)
    freq_high = np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_freq_img[:, :, 0]))) + 1e-10)
    freq_hybrid = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid[:, :, 0]))) + 1e-10)
    
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    
    axes[0, 0].imshow(im1_aligned)
    axes[0, 0].set_title(f'Image 1 (High-freq source)\nσ={sigma1}', fontsize=10)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(im2_aligned)
    axes[0, 1].set_title(f'Image 2 (Low-freq source)\nσ={sigma2}', fontsize=10)
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(high_freq_display)
    axes[0, 2].set_title('High-freq component\n(+0.5 for visibility)', fontsize=10)
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(low_freq_img)
    axes[0, 3].set_title('Low-freq component\n(Gaussian filtered)', fontsize=10)
    axes[0, 3].axis('off')
    
    axes[0, 4].imshow(hybrid)
    axes[0, 4].set_title('Hybrid Result', fontsize=10)
    axes[0, 4].axis('off')
    
    axes[1, 0].imshow(freq_im1, cmap='gray')
    axes[1, 0].set_title('FFT: Image 1', fontsize=10)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(freq_im2, cmap='gray')
    axes[1, 1].set_title('FFT: Image 2', fontsize=10)
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(freq_high, cmap='gray')
    axes[1, 2].set_title('FFT: High-freq', fontsize=10)
    axes[1, 2].axis('off')
    
    axes[1, 3].imshow(freq_low, cmap='gray')
    axes[1, 3].set_title('FFT: Low-freq', fontsize=10)
    axes[1, 3].axis('off')
    
    axes[1, 4].imshow(freq_hybrid, cmap='gray')
    axes[1, 4].set_title('FFT: Hybrid', fontsize=10)
    axes[1, 4].axis('off')
    
    axes[2, 0].imshow(gaussian1_2d, cmap='hot')
    axes[2, 0].set_title(f'High-pass kernel\n{ksize1}×{ksize1}, σ={sigma1}', fontsize=10)
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(gaussian2_2d, cmap='hot')
    axes[2, 1].set_title(f'Low-pass kernel\n{ksize2}×{ksize2}, σ={sigma2}', fontsize=10)
    axes[2, 1].axis('off')
    
    center = freq_im1.shape[0] // 2
    profile_im1 = freq_im1[center, :]
    profile_im2 = freq_im2[center, :]
    profile_hybrid = freq_hybrid[center, :]
    
    axes[2, 2].plot(profile_im1, 'b-', label='Image 1', alpha=0.7)
    axes[2, 2].plot(profile_im2, 'r-', label='Image 2', alpha=0.7)
    axes[2, 2].set_title('Frequency Profiles\n(horizontal slice)', fontsize=10)
    axes[2, 2].legend(fontsize=8)
    axes[2, 2].grid(True, alpha=0.3)
    
    axes[2, 3].plot(profile_hybrid, 'g-', label='Hybrid', linewidth=2)
    axes[2, 3].set_title('Hybrid Frequency Profile', fontsize=10)
    axes[2, 3].legend(fontsize=8)
    axes[2, 3].grid(True, alpha=0.3)
    
    axes[2, 4].text(0.1, 0.8, f'Cutoff Parameters:', fontsize=12, weight='bold', transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.65, f'High-pass σ = {sigma1}', fontsize=10, transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.55, f'Low-pass σ = {sigma2}', fontsize=10, transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.4, 'Chosen to balance:', fontsize=10, weight='bold', transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.3, '• Close-up detail', fontsize=9, transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.2, '• Distance visibility', fontsize=9, transform=axes[2, 4].transAxes)
    axes[2, 4].text(0.1, 0.1, '• Minimal artifacts', fontsize=9, transform=axes[2, 4].transAxes)
    axes[2, 4].set_xlim(0, 1)
    axes[2, 4].set_ylim(0, 1)
    axes[2, 4].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_complete_pipeline.jpg', dpi=300, bbox_inches='tight')
    plt.close()

def create_personal_hybrids():
    def simple_align_images(im1, im2, target_size=(400, 400)):
        im1_resized = cv2.resize(im1, target_size)
        im2_resized = cv2.resize(im2, target_size)
        return im1_resized, im2_resized
    
    try:
        mbappe_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/mbappe.jpg')
        turtle_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/turtle.jpeg')
        
        mbappe_aligned, turtle_aligned = simple_align_images(mbappe_img, turtle_img)
        
        sigma_high_1 = 2.5
        sigma_low_1 = 8
        
        mbappe_turtle_hybrid = hybrid_image(mbappe_aligned, turtle_aligned, sigma_high_1, sigma_low_1)
        
        save_image(mbappe_aligned, 'personal_hybrid1_img1_aligned.jpg', 'Mbappé Aligned')
        save_image(turtle_aligned, 'personal_hybrid1_img2_aligned.jpg', 'Ninja Turtle Aligned')
        save_image(mbappe_turtle_hybrid, 'personal_hybrid1_result.jpg', 'Mbappé-Ninja Turtle Hybrid')
        
        create_complete_pipeline_visualization(
            mbappe_img, turtle_img, mbappe_aligned, turtle_aligned, 
            sigma_high_1, sigma_low_1, mbappe_turtle_hybrid, 
            filename_prefix="personal_hybrid1"
        )
        
    except Exception as e:
        pass
    
    try:
        shrek_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/shrek.jpeg')
        donkey_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/donkey.jpg')
        
        shrek_aligned, donkey_aligned = simple_align_images(shrek_img, donkey_img)
        
        sigma_high_2 = 3.0
        sigma_low_2 = 9
        
        shrek_donkey_hybrid = hybrid_image(shrek_aligned, donkey_aligned, sigma_high_2, sigma_low_2)
        
        save_image(shrek_aligned, 'personal_hybrid2_img1_aligned.jpg', 'Shrek Aligned')
        save_image(donkey_aligned, 'personal_hybrid2_img2_aligned.jpg', 'Donkey Aligned')
        save_image(shrek_donkey_hybrid, 'personal_hybrid2_result.jpg', 'Shrek-Donkey Hybrid')
        
        create_complete_pipeline_visualization(
            shrek_img, donkey_img, shrek_aligned, donkey_aligned, 
            sigma_high_2, sigma_low_2, shrek_donkey_hybrid, 
            filename_prefix="personal_hybrid2"
        )
        
    except Exception as e:
        pass
if __name__ == "__main__":
    im1 = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/nutmeg.jpg')  
    im2 = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/DerekPicture.jpg')       

    im1_aligned, im2_aligned = align_images(im1, im2)

    sigma1 = 2
    sigma2 = 6
    hybrid1 = hybrid_image(im1_aligned, im2_aligned, sigma1, sigma2)
    save_image(hybrid1, 'part2_2_hybrid1.jpg', 'Hybrid Image 1 (Derek-Nutmeg)')

    freq_im1 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned[:, :, 0]))))
    freq_im2 = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im2_aligned[:, :, 0]))))
    freq_low = np.log(np.abs(np.fft.fftshift(np.fft.fft2(cv2.filter2D(im2_aligned[:, :, 0], -1, create_gaussian_kernel(max(15, int(6*sigma2+1)), sigma2), borderType=cv2.BORDER_REFLECT)))))
    freq_high = np.log(np.abs(np.fft.fftshift(np.fft.fft2(im1_aligned[:, :, 0] - cv2.filter2D(im1_aligned[:, :, 0], -1, create_gaussian_kernel(max(15, int(6*sigma1+1)), sigma1), borderType=cv2.BORDER_REFLECT)))))
    freq_hybrid = np.log(np.abs(np.fft.fftshift(np.fft.fft2(hybrid1[:, :, 0]))))
    save_image(freq_im1, 'part2_2_freq_derek.jpg', 'Fourier Transform (Derek)')
    save_image(freq_im2, 'part2_2_freq_nutmeg.jpg', 'Fourier Transform (Nutmeg)')
    save_image(freq_low, 'part2_2_freq_low.jpg', 'Fourier Transform (Low Freq)')
    save_image(freq_high, 'part2_2_freq_high.jpg', 'Fourier Transform (High Freq)')
    save_image(freq_hybrid, 'part2_2_freq_hybrid.jpg', 'Fourier Transform (Hybrid)')

    N = 5
    pyramid = gaussian_pyramid(hybrid1, N)
    for i in range(N):
        save_image(pyramid[i], f'part2_2_pyramid_level_{i}.jpg', f'Pyramid Level {i}')
    
    apple = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/apple.jpeg')
    orange = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/orange.jpeg')

    levels = 5
    lap_apple = laplacian_stack(apple, levels, sigma=2)
    lap_orange = laplacian_stack(orange, levels, sigma=2)

    rows, cols, _ = apple.shape
    mask = np.zeros((rows, cols, 3))
    mask[:, :cols//2, :] = 1
    gmask = gaussian_stack(mask, levels, sigma=2)

    visualize_stack_grid(
        [lap_apple, lap_orange, gmask],
        ["Apple Laplacian", "Orange Laplacian", "Gaussian Mask"],
        "part2_3_stacks.jpg",
        normalize=True
    )
    
    create_complete_pipeline_visualization(
        im1, im2, im1_aligned, im2_aligned, 
        sigma1, sigma2, hybrid1, 
        filename_prefix="derek_nutmeg"
    )

    def simple_resize_align(im1, im2, target_size=(400, 400)):
        im1_resized = cv2.resize(im1, target_size)
        im2_resized = cv2.resize(im2, target_size)
        return im1_resized, im2_resized

    def create_personal_hybrids():
        try:
            mbappe_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/mbapper.jpeg')
            turtle_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/ninjat.jpeg')
            
            mbappe_aligned, turtle_aligned = simple_resize_align(mbappe_img, turtle_img)
            
            sigma_high_1 = 2.5
            sigma_low_1 = 8
            
            mbappe_turtle_hybrid = hybrid_image(mbappe_aligned, turtle_aligned, sigma_high_1, sigma_low_1)
            
            save_image(mbappe_aligned, 'personal_hybrid1_img1_aligned.jpg', 'Mbappé Aligned')
            save_image(turtle_aligned, 'personal_hybrid1_img2_aligned.jpg', 'Ninja Turtle Aligned')
            save_image(mbappe_turtle_hybrid, 'personal_hybrid1_result.jpg', 'Mbappé-Ninja Turtle Hybrid')
            
            create_complete_pipeline_visualization(
                mbappe_img, turtle_img, mbappe_aligned, turtle_aligned, 
                sigma_high_1, sigma_low_1, mbappe_turtle_hybrid, 
                filename_prefix="personal_hybrid1"
            )
            
        except Exception as e:
            pass
        
        try:
            shrek_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/shrek.jpeg')
            donkey_img = load_image('/Users/yuxuancai/cs180/cs180/Proj_2/code/images/donkey.jpg')
            
            shrek_aligned, donkey_aligned = simple_resize_align(shrek_img, donkey_img)
            
            sigma_high_2 = 3.0
            sigma_low_2 = 9
            
            shrek_donkey_hybrid = hybrid_image(shrek_aligned, donkey_aligned, sigma_high_2, sigma_low_2)
            
            save_image(shrek_aligned, 'personal_hybrid2_img1_aligned.jpg', 'Shrek Aligned')
            save_image(donkey_aligned, 'personal_hybrid2_img2_aligned.jpg', 'Donkey Aligned')
            save_image(shrek_donkey_hybrid, 'personal_hybrid2_result.jpg', 'Shrek-Donkey Hybrid')
            
            create_complete_pipeline_visualization(
                shrek_img, donkey_img, shrek_aligned, donkey_aligned, 
                sigma_high_2, sigma_low_2, shrek_donkey_hybrid, 
                filename_prefix="personal_hybrid2"
            )
            
        except Exception as e:
            pass

    create_personal_hybrids()