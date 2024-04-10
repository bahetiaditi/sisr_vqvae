import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def show_images(original, downsampled, bilinear, index, 
                title1='Original Image', title2='Downsampled Image', title3='Bilinear Interpolated Image'):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(np.clip(original.squeeze(), 0, 1), cmap='gray')
    ax[0].set_title(f"{title1} - Pair {index+1}")
    ax[0].axis('off')

    ax[1].imshow(np.clip(downsampled.squeeze(), 0, 1), cmap='gray')
    ax[1].set_title(f"{title2} - Pair {index+1}")
    ax[1].axis('off')

    ax[2].imshow(np.clip(bilinear.squeeze(), 0, 1), cmap='gray')
    ax[2].set_title(f"{title3} - Pair {index+1}")
    ax[2].axis('off')

    plt.show()

def calculate_metrics(original, reconstructed):
    psnr = compare_psnr(original, reconstructed, data_range=1)
    ssim = compare_ssim(original, reconstructed, data_range=1)
    return psnr, ssim