root_dir = "data"

import os
import random
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io.image import decode_image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from torchvision.transforms.functional import to_pil_image
from tqdm.auto import tqdm

# Determine if this Colab instand has access to a GPU (CUDA), which makes ML code faster
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



# Configure the random seed to make results more reproducible
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

class RUGDDataset(Dataset):

    # SUBDIR_SPLIT = {
    #     'train': ['park-1', 'trail-15', 'trail-3', 'trail-4', 'trail-5', 'trail-6', 'trail-7', 'trail-9'],
    #     'validation': ['village', 'park-2', 'trail-14'],
    #     # 'test': ['creek', 'park-8', 'trail'],
    #     # 'test': ['creek'],
    #     # 'test': ['trail'],
    #     'test': ['park-8'],
    # }
# switched around the files
    SUBDIR_SPLIT = {
        'train': ['park-1', 'trail-15', 'trail-3', 'trail-4', 'trail-5', 'trail-6', 'trail-7', 'trail-10'],
        'validation': ['village', 'park-2', 'trail-14'],
        'test': ['creek', 'park-8', 'trail-9'],
        # 'test': ['creek'],
        # 'test': ['trail-9'],
        # 'test': ['park-8'],
    }


    def __init__(self, split, root_dir=root_dir, tform=None):
        self.im_dir = os.path.join(root_dir, 'RUGD_frames-with-annotations')
        self.label_dir = os.path.join(root_dir, 'RUGD_annotations')
        self.split = split
        self.tform = tform

        self.paths = self.create_path_list()  # paired filenames for RGB and label images
        self.int_to_class_map = self.load_color_class_map()  # (R, G, B) -> class index

    def load_color_class_map(self):
        int_to_class_map = {}
        classes = []
        with open(os.path.join(self.label_dir, 'RUGD_annotation-colormap.txt'), 'r') as file:
            for line in file:
                i, title, r, g, b = line.rstrip().split(' ')
                i, r, g, b = int(i), int(r), int(g), int(b)
                classes.append(title)
                int_to_class_map[self.color_to_int((r, g, b))] = i
        return int_to_class_map

    def color_to_int(self, color):
        r, g, b = color[0], color[1], color[2]
        return r + g * 256 + b * 256**2

    def color_to_class(self, color):
        color_int = self.color_to_int(color)
        is_ndim = type(color_int) in [torch.Tensor, np.ndarray]

        if is_ndim:
            class_label = np.copy(color_int)
            for k, v in self.int_to_class_map.items():
                class_label[color_int == k] = v
            return class_label
        else:
            return self.int_to_class_map[color_int]

    def create_path_list(self):
        # todo: make sure file exists in both directories?
        # subdirs = [f.path for f in os.scandir(self.im_dir) if f.is_dir()]
        subdirs = [
            os.path.join(self.im_dir, subdir)
            for subdir in RUGDDataset.SUBDIR_SPLIT[self.split]
        ]
        paths = []
        for dir in subdirs:
            paths_ = glob(os.path.join(self.im_dir, dir, '*.png'))
            for path in paths_:
                fname = os.path.basename(path)

                if path.endswith('trail/trail_02551.png'):
                    # Hardcoded fix for corrupted file
                    continue

                path1 = os.path.join(self.label_dir, os.path.basename(dir), fname)
                path2 = path
                paths.append([path1, path2])
        return pd.DataFrame(paths, columns=['label_path', 'im_path'])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        row = self.paths.iloc[i]
        im = decode_image(row['im_path'])
        color_label = decode_image(row['label_path'])
        class_label = torch.from_numpy(self.color_to_class(color_label))
        if self.tform:
            im = self.tform(im)
        return im, class_label

from segmentation_models_pytorch.losses import DiceLoss
from segmentation_models_pytorch.losses import FocalLoss

weights = DeepLabV3_ResNet50_Weights.DEFAULT
PREPROCESS_FN = weights.transforms()
RESIZE_FN = torchvision.transforms.Resize((550, 688))

def validation_acc(val_loader, model):
    running_acc = 0
    model.eval()
    for batch in val_loader:
        im, label = batch
        im = im.to(device)
        im = PREPROCESS_FN(im)
        label = label.to(device)

        with torch.no_grad():
            pred = model(im)
        pred = RESIZE_FN(pred)
        pred_label = pred.argmax(axis=1)
        acc = (pred_label == label).sum() / label.numel()
        running_acc += acc
    model.train()
    return running_acc / len(val_loader)


class CustomModel_3(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.pretrained_model = deeplabv3_resnet50(weights=weights)
        self.one_layer = nn.Conv2d(21, 32, kernel_size, padding=2)
        self.two_layer = nn.Conv2d(32, 32, kernel_size, padding=2)
        self.final_layer = nn.Conv2d(32, 25, kernel_size, padding=2)

        self.one_batchnorm = nn.BatchNorm2d(32)
        self.two_batchnorm = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.pretrained_model(x)['out']

        y = self.one_layer(y)
        y = self.one_batchnorm(y)
        y = self.relu(y)

        y = self.two_layer(y)
        y = self.two_batchnorm(y)
        y = self.relu(y)

        return self.final_layer(y)

# Hyperparameters ################
BATCH_SIZE = 30
NUM_EPOCHS = 40
LR = 0.0001
KERNEL_SIZE3 = 5
##################################

train_loader = DataLoader(
    RUGDDataset('train'),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    RUGDDataset('validation'),
    batch_size=BATCH_SIZE, shuffle=False,
)

model = CustomModel_3(KERNEL_SIZE3).to(device) # 3 layers

#### DO NOT change the code below ###
for param in model.pretrained_model.parameters():
    param.requires_grad = False
for param in model.pretrained_model.aux_classifier.parameters():
    param.requires_grad = True
for param in model.pretrained_model.aux_classifier[-1].parameters():
    param.requires_grad = True
#####################################

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()
loss_fn2 = FocalLoss('multiclass')

losses = []
accs = []

for epoch in range(NUM_EPOCHS):
    running_loss = 0
    pbar_prefix = f'Epoch {epoch + 1}/{NUM_EPOCHS}'
    pbar = tqdm(train_loader, desc=pbar_prefix)
    for i, batch in enumerate(pbar):
        optimizer.zero_grad()

        im, label = batch
        im = im.to(device)
        im_original_size = im.shape[-2:]
        im = PREPROCESS_FN(im)
        model_input_size = im.shape[-2:]
        label = label.to(device)

        pred = model(im)
        model_output_size = pred.shape[-2:]
        pred = RESIZE_FN(pred)
        final_output_size = pred.shape[-2:]

        assert im_original_size == final_output_size
        assert model_input_size == model_output_size

        loss1 = loss_fn(pred, label.long())
        loss2 = loss_fn2(pred, label.long())
        loss = loss1 + 0.5 * loss2
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f'{pbar_prefix} loss={running_loss / (i + 1):.4f}')

    if (epoch % 2 == 0) or (epoch == NUM_EPOCHS - 1):
        acc = validation_acc(val_loader, model)

    losses.append(running_loss / len(train_loader))
    accs.append(acc.item())
    print(acc.item())

# don't use until done training model
ds = RUGDDataset('test')
test_loader = DataLoader(
    ds, batch_size=16, shuffle=True,
)

acc = validation_acc(test_loader, model)
print(acc)

#to save model weights:
torch.save(model.state_dict(), "custom_model_3_weights.pth")

fig, ax = plt.subplots()
ax.plot(losses)
ax.set_ylabel('loss')
ax2 = ax.twinx()
ax2.plot(accs, color='orange')
ax2.set_ylabel('val acc')
ax.set_xlabel('epoch')
fig.legend(['train loss', 'val acc'])