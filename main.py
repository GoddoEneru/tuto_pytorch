import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import draw_segmentation_masks
import model
from train import train
from test import test
from custom_dataset import CustomDataset
from torchvision.models.segmentation import fcn_resnet50

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as f

IMAGE_PATH = 'img/dataA/dataA/CameraRGB/'
MASK_PATH = 'img/dataA/dataA/CameraSeg/'

data = CustomDataset(IMAGE_PATH, MASK_PATH, 4)


# def show(imgs):
#     if not isinstance(imgs, list):
#         imgs = [imgs]
#     fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = f.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img))
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


# result = draw_segmentation_masks(d[0], d[1], alpha=0.6)
# show(result)


batch_size = 10

# Create data loader.
dataloader = DataLoader(data, batch_size=batch_size)

for batch_id, batch_sample in enumerate(dataloader):
    print(f"Batch's ID: {batch_id}")
    # print(f"Numbers of classes: {batch_sample['mask'].max()}")
    print(f"Shape of X [N, C, H, W]: {batch_sample['img'].shape}")
    print(f"Shape of y: {batch_sample['mask'].shape} {batch_sample['mask'].dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

model = fcn_resnet50(pretrained=True, progress=False)
model = model.eval()
#
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#
# epochs = 5
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train(train_dataloader, model, loss_fn, optimizer, device)
#     test(test_dataloader, model, loss_fn, device)
# print("Done!")
#
for batch_id, batch_sample in enumerate(dataloader):
    output = model(batch_sample['img'])['out']
    print(output.shape, output.min().item(), output.max().item())
    print(output)
