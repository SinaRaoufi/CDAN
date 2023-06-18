import os

from torch.utils.data.dataset import Dataset
from PIL import Image


class LLIDataset(Dataset):
    def __init__(self, low_light_root, normal_light_root, low_light_transforms, normal_light_transforms):
        super().__init__()
        self.low_light_dataset = [os.path.join(low_light_root, image) for image in os.listdir(low_light_root)]
        self.normal_light_dataset = [os.path.join(normal_light_root, image) for image in os.listdir(normal_light_root)]

        self.low_light_transform = low_light_transforms
        self.normal_light_transform = normal_light_transforms
        
    def __getitem__(self, idx):
        low_light = Image.open(self.low_light_dataset[idx]).convert('RGB')
        normal_light = Image.open(self.normal_light_dataset[idx]).convert('RGB')
        low_light = self.low_light_transform(low_light)
        normal_light = self.normal_light_transform(normal_light)
        
        return low_light, normal_light
    
    def __len__(self): 
        return len(self.low_light_dataset)