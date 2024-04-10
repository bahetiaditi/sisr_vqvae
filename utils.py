import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

def add_artifacts(image, num_shapes):
    if image is None:
        print("Error: Unable to load the image.")
        return None

    # Ensure the image is a numpy array with the correct type
    if image.dtype != np.uint8:
        # Convert the image to 8-bit if it's not already
        image = np.clip(image, 0, 255).astype(np.uint8)

    # Directly use the image if it's grayscale
    image_with_artifacts = image.copy()

    height, width = image_with_artifacts.shape[:2]

    for _ in range(num_shapes):
        shape_type = random.choice(['circle', 'rectangle'])
        color = 255  # White color for grayscale
        thickness = 2

        if shape_type == 'circle':
            x = np.random.randint(5, width - 10)  # Avoid edges
            y = np.random.randint(5, height - 10)
            radius = np.random.randint(8, 10)  # Limit size
            cv2.circle(image_with_artifacts, (x, y), radius, color, thickness)

        elif shape_type == 'rectangle':
            x1 = np.random.randint(7, width - 15)
            y1 = np.random.randint(7, height - 15)
            x2 = x1 + np.random.randint(7, 15)
            y2 = y1 + np.random.randint(7, 15)
            cv2.rectangle(image_with_artifacts, (x1, y1), (x2, y2), color, thickness)

    return image_with_artifacts


def show_images(original, downsampled, bilinear, bilinear_with_artifacts, index, 
                title1='Original Image', title2='Downsampled Image', 
                title3='Bilinear Interpolated Image', title4='Bilinear Interpolated with Artifacts'):
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))  # Adjusted for four subplots

    # Display original image
    ax[0].imshow(original.squeeze(), cmap='gray')
    ax[0].set_title(f"{title1} - Pair {index+1}")
    ax[0].axis('off')

    # Display downsampled image
    ax[1].imshow(downsampled.squeeze(), cmap='gray')
    ax[1].set_title(f"{title2} - Pair {index+1}")
    ax[1].axis('off')

    # Display bilinear interpolated image
    ax[2].imshow(bilinear.squeeze(), cmap='gray')
    ax[2].set_title(f"{title3} - Pair {index+1}")
    ax[2].axis('off')

    # Display bilinear interpolated image with artifacts
    ax[3].imshow(bilinear_with_artifacts.squeeze(), cmap='gray')
    ax[3].set_title(f"{title4} - Pair {index+1}")
    ax[3].axis('off')

    plt.show()

def calculate_metrics(original, reconstructed):
    psnr = compare_psnr(original, reconstructed, data_range=1)
    ssim = compare_ssim(original, reconstructed, data_range=1)
    return psnr, ssim
