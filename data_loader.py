import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, image_dir, num_shapes=5):  # Added num_shapes argument
        self.image_dir = image_dir
        self.image_filenames = [
            os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) 
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
            transforms.Resize((256, 256)),  # Resize images to 256*256
            transforms.Resize((512, 512), interpolation=Image.BILINEAR),  # Upsample back to 512x512
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
        ])
        self.num_shapes = num_shapes  # Store the number of shapes to add

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_path = self.image_filenames[idx]
        image = Image.open(image_path)

        # Apply transformations
        original_image = self.original_transform(image)
        downsampled_image = self.downsample_transform(image)
        bilinear_image = self.bilinear_transform(image)

        # Convert PIL Image to numpy array for artifact addition
        bilinear_image_np = np.array(transforms.ToPILImage()(bilinear_image))
        bilinear_image_with_artifacts_np = add_artifacts(bilinear_image_np, self.num_shapes)
        
        # Convert numpy array with artifacts back to PIL Image and then to Tensor
        bilinear_image_with_artifacts = transforms.ToTensor()(Image.fromarray(bilinear_image_with_artifacts_np))
        bilinear_image_with_artifacts = transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))(bilinear_image_with_artifacts)

        return original_image, downsampled_image, bilinear_image, bilinear_image_with_artifacts
