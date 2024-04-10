import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from data_loader import CustomDataset
from models import Model
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np

def train_epoch(model, device, train_loader, optimizer, epoch):
    model.train()
    train_res_total_loss = []
    train_res_recon_error = []
    train_res_vq_loss = []
    train_res_psnr = []
    train_res_ssim = []
    
    for batch_idx, (original, _, _, bilinear_with_artifacts) in enumerate(train_loader):
        original = original.float().to(device)
        bilinear_with_artifacts = bilinear_with_artifacts.float().to(device)
    
        optimizer.zero_grad()
        vq_loss, data_recon, _ = model(bilinear_with_artifacts)  
        
        recon_error = F.mse_loss(data_recon, original)
        total_loss = recon_error + vq_loss
        total_loss.backward()
        optimizer.step()
        
        train_res_total_loss.append(total_loss.item())
        train_res_recon_error.append(recon_error.item())
        train_res_vq_loss.append(vq_loss.item())
        
        # Calculate and store PSNR and SSIM for each image in batch
        for i in range(data_recon.shape[0]):
            psnr = compare_psnr(original[i].cpu().detach().numpy(), data_recon[i].cpu().detach().numpy(), data_range=1)
            ssim = compare_ssim(original[i].cpu().detach().numpy().squeeze(), data_recon[i].cpu().detach().numpy().squeeze(), data_range=1)
            train_res_psnr.append(psnr)
            train_res_ssim.append(ssim)

    # Log epoch summary
    print('Epoch Summary:')
    print(f'Average Total Loss: {np.mean(train_res_total_loss)}')
    print(f'Average Reconstruction Error: {np.mean(train_res_recon_error)}')
    print(f'Average VQ Loss: {np.mean(train_res_vq_loss)}')
    print(f'Average PSNR: {np.mean(train_res_psnr)}')
    print(f'Average SSIM: {np.mean(train_res_ssim)}')

def test_epoch(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        for batch_idx, (original, _, _, bilinear_with_artifacts) in enumerate(test_loader):
            original = original.float().to(device)
            bilinear_with_artifacts = bilinear_with_artifacts.float().to(device)
            
            # Forward pass
            vq_loss, data_recon, _ = model(bilinear_with_artifacts)
            fig, axes = plt.subplots(len(original), 8, figsize=(15, len(original)*3))
            for i in range(len(original)):
               # Calculate metrics and loss for each image
               recon_error = F.mse_loss(data_recon[i].unsqueeze(0), original[i].unsqueeze(0))
               total_loss = recon_error.item() + vq_loss.item()
               psnr = compare_psnr(tensor_to_image(original[i]), tensor_to_image(data_recon[i]), data_range=1)
               ssim = compare_ssim(tensor_to_image(original[i]), tensor_to_image(data_recon[i]), data_range=1)

              # Display images and metrics
              axes[i, 0].imshow(tensor_to_image(original[i]), cmap='gray')
              axes[i, 1].imshow(tensor_to_image(downsampled[i]), cmap='gray')
              axes[i, 2].imshow(tensor_to_image(bilinear[i]), cmap='gray')
              axes[i, 3].imshow(tensor_to_image(bilinear_with_artifacts[i]), cmap='gray')
              axes[i, 4].imshow(tensor_to_image(data_recon[i]), cmap='gray')
              axes[i, 5].text(0.5, 0.5, f'{total_loss:.4f}', horizontalalignment='center', verticalalignment='center')
              axes[i, 6].text(0.5, 0.5, f'{psnr:.4f}', horizontalalignment='center', verticalalignment='center')
              axes[i, 7].text(0.5, 0.5, f'{ssim:.4f}', horizontalalignment='center', verticalalignment='center')

              # Set titles and turn off axes
              for j in range(8):
                 axes[i, j].axis('off')
              axes[i, 0].set_title("Original")
              axes[i, 1].set_title("Downsampled")
              axes[i, 2].set_title("bilinear")
              axes[i, 3].set_title("bilinear with artifacts")
              axes[i, 4].set_title("Reconstructed")
              axes[i, 5].set_title("Total Loss")
              axes[i, 6].set_title("PSNR Score")
              axes[i, 7].set_title("SSIM Score")

   plt.tight_layout()
   plt.show()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = "/path/to/your/dataset"  # Update this path
    dataset = CustomDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = Model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32, num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_epoch(model, device, train_loader, optimizer, epoch)
        test_epoch(model, device, test_loader)

if __name__ == '__main__':
    main()
