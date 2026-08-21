import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class HumanSegmentationDataset(Dataset):
    """
    A custom dataset class for loading image data

    This class is designed to work with PyTorch's Dataset and DataLoader
    abstractions. It handles loading images and their corressponding masks
    from a specific directory structure
    """

    def __init__(self, df, dataset_root, transform=None):
        """
        Initializes the dataset object.

        Args:
            df (pd.DataFrame): DataFrame with images and masks partial paths
            dataset_root (str): The root directory where the dataset is stored.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df
        self.dataset_root = dataset_root
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        img_partial_path = self.data.iloc[idx]['images']
        mask_partial_path = self.data.iloc[idx]['masks']
        
        # 2. Create the full absolute paths
        image_path = os.path.join(self.dataset_root, img_partial_path)
        mask_path = os.path.join(self.dataset_root, mask_partial_path)
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Albumentations ToTensorV2 leaves 2D masks as [H, W]. 
        # PyTorch BCEWithLogitsLoss expects [1, H, W], so we add the channel dimension:
        mask = mask.unsqueeze(0) 
        
        # Normalize mask to 0 and 1, and ensure it's a float tensor
        mask = (mask > 0).float()
        
        # Scale image pixels from [0, 255] to [0, 1] 
        # (Albumentations ToTensorV2 doesn't auto-divide by 255 like torchvision does)
        image = image / 255.0

        return image, mask
