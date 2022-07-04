import os
from torch.utils.data import Dataset
from PIL import Image, ImageOps
from torchvision.transforms import PILToTensor, ToTensor


class CustomDataset(Dataset):
    def __init__(self, img_path, mask_path, num_classes, transform=None):
        self.img_path = img_path
        self.file_img = os.listdir(self.img_path)
        self.mask_path = mask_path
        self.file_mask = os.listdir(self.mask_path)
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.file_img)

    def __getitem__(self, idx):
        img = self.file_img[idx]
        mask = self.file_mask[idx]
        img = Image.open(self.img_path + img)
        mask = Image.open(self.mask_path + mask)
        mask = ImageOps.grayscale(mask)
        # open mask (niveaux de gris)
        # preparer box si mask_rcnn
        # faire tensor
        convert_tensor = ToTensor()
        return {'img': convert_tensor(img),
                'mask': convert_tensor(mask)}
