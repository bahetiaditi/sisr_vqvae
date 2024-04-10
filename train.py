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

def train(model, device, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (original, _, bilinear) in enumerate(train_loader):
        original, bilinear = original.to(device), bilinear.to(device)
        optimizer.zero_grad()
        vq_loss, data_recon, _ = model(bilinear)
        recon_error = F.mse_loss(data_recon, original)
        total_loss = recon_error + vq_loss
        total_loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(bilinear), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), total_loss.item()))

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for original, _, bilinear in test_loader:
            original, bilinear = original.to(device), bilinear.to(device)
            vq_loss, data_recon, _ = model(bilinear)
            test_loss += F.mse_loss(data_recon, original, reduction='sum').item()
    test_loss /= len(test_loader.dataset)
    print('\\nTest set: Average loss: {:.4f}\\n'.format(test_loss))

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    dataset = CustomDataset('/path/to/your/dataset')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model = Model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)