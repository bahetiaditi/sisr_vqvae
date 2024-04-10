from data_loader import CustomDataset
from models import Model
from train import train, test
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_path = "/path/to/your/dataset"  # Update this path
    dataset = CustomDataset(dataset_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # Initialize the model
    model = Model(num_hiddens=128, num_residual_layers=2, num_residual_hiddens=32,
                  num_embeddings=512, embedding_dim=64, commitment_cost=0.25).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Number of epochs
    num_epochs = 10

    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)

if __name__ == '__main__':
    main()