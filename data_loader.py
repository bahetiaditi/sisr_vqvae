import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_filenames = [
            os.path.join(dp, f) for dp, _, filenames in os.walk(image_dir) 
            for f in filenames if os.path.splitext(f)[1] in ['.png', '.jpg', '.jpeg']
        ]
        self.original_transform = transforms.Compose([
            transforms.Resize((512, 512)),  # Resize original images to 512x512
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])
        self.downsample_transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize images to 256*256
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])
        self.bilinear_transform = transforms.Compose([
            transforms.Resize((256,256)),  # Resize images to 256*256
            transforms.Resize((512, 512), interpolation=Image.BILINEAR),  # Upsample back to 512x512
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        image = Image.open(image_path)

        original_image = self.original_transform(image)
        downsampled_image = self.downsample_transform(image)
        bilinear_image = self.bilinear_transform(image)

        return original_image, downsampled_image, bilinear_image
