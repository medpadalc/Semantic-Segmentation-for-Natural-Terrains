# Set-Up:
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
import zipfile
import urllib.request

data_dir = "data/rugd"
os.makedirs(data_dir, exist_ok=True)

annotations_zip = os.path.join(data_dir, "RUGD_annotations.zip")
annotations_url = "http://rugd.vision/data/RUGD_annotations.zip"

if not os.path.exists(annotations_zip):
    urllib.request.urlretrieve(annotations_url, annotations_zip)

with zipfile.ZipFile(annotations_zip, "r") as zf:
    zf.extractall(data_dir)

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


    def __init__(self, split, root_dir='data/rugd', tform=None):
        # self.im_dir = os.path.join(root_dir, 'RUGD_frames-with-annotations')
        frames_zip = os.path.join(data_dir, "RUGD_frames-with-annotations.zip")
        frames_url = "http://rugd.vision/data/RUGD_frames-with-annotations.zip"
        
        if not os.path.exists(frames_zip):
            urllib.request.urlretrieve(frames_url, frames_zip)
        
        with zipfile.ZipFile(frames_zip, "r") as zf:
            zf.extractall(data_dir)
    
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
        # subdirs = [f.path for f in os.scandir(self.im_dir) if f.is_dir()]
        # subdirs = [
        #     os.path.join(self.im_dir, subdir)
        #     for subdir in RUGDDataset.SUBDIR_SPLIT[self.split]
        # ]
        # paths = []
        # for dir in subdirs:
        #     paths_ = glob(os.path.join(self.im_dir, dir, '*.png'))
        paths_ = glob(os.path.join(dir, '*.png'))
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

!pip install -U segmentation-models-pytorch

# Benchmarking
PASCAL_TO_RUGD = {
    0: 17,   # person → person
    1: 16,   # bicycle → bicycle
    2: 8,    # car → vehicle
    3: 8,    # motorbike → vehicle
    4: 0,    # aeroplane → void
    5: 8,    # bus → vehicle
    6: 8,    # train → vehicle
    7: 8,    # boat → vehicle
    8: 0,    # bird → void
    9: 0,    # cat → void
    10: 0,   # dog → void
    11: 0,   # horse → void
    12: 0,   # sheep → void
    13: 0,   # cow → void
    14: 0,   # elephant → void
    15: 0,   # bear → void
    16: 0,   # zebra → void
    17: 0,   # giraffe → void
    18: 0,   # potted plant → void
    19: 0,   # void → void
    20: 0,
}

def convert_to_rugd_labels(pascal_labels):
    # converts PASCAL labels to RUGD labels
    # input: pascal_labels torch.tensor of size (h, w), 21(or 20?) classes
    # output: rugd labels, 25 classes

    # ex. if PASCAL class bike = 2 and RUGD class bike = 4, convert all 2s to 4s
    # if PASCAL class isn't present, then convert it to the RUGD void class

    rugd_labels = torch.zeros_like(pascal_labels)

    for pascal_id, rugd_id in PASCAL_TO_RUGD.items():
        rugd_labels[pascal_labels == pascal_id] = rugd_id

    return rugd_labels

def accuracy_fn(pred, target):
    num_correct = ((pred == target) & (pred == 0)).sum()
    if (pred == 0).sum() == 0:
        return 1
    return num_correct / (pred == 0).sum()

def classwise_acc_fn(pred, target, num_classes=25):
    correct = pred == target
    class_accs = []
    for i in range(num_classes):
        target_i = target == i
        num_correct = (correct * target_i).sum()
        acc = num_correct / target_i.sum()
        if torch.isnan(acc):
            acc = 0
        class_accs.append(acc)
    return torch.tensor(class_accs)

weights = DeepLabV3_ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
model = deeplabv3_resnet50(weights=weights).to(device).eval()
resize_fn = torchvision.transforms.Resize((550, 688))

criteria = [
    # classwise_acc_fn,
    accuracy_fn,
    # torch.nn.CrossEntropyLoss(),
]
running_criteria = [0 for _ in criteria]

ds = RUGDDataset('train')
train_loader = DataLoader(
    ds, batch_size=16, shuffle=True,
)
ds = RUGDDataset('test')
test_loader = DataLoader(
    ds, batch_size=16, shuffle=True,
)

# Evaluation loop for benchmarking
for i, batch in enumerate(tqdm(test_loader)):
    im, label = batch
    im = im.to(device)
    im = preprocess(im)
    label = label.to(device)

    with torch.no_grad():
        pred = model(im)['out']

    pred = resize_fn(pred)
    pred_label = torch.argmax(pred, axis=1)

    pred_label = convert_to_rugd_labels(pred_label)

    running_criteria = [
        c_val + criterion(pred_label, label.long())
        for (c_val, criterion) in zip(running_criteria, criteria)
    ]
running_criteria = [c_val / len(train_loader) for c_val in running_criteria]

# Model
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

# One Layer Model
class CustomModel_1(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.pretrained_model = deeplabv3_resnet50(weights=weights)
        self.final_layer = nn.Conv2d(21, 25, kernel_size, padding=5, stride=1)
        # nn.Conv2d(21, 25, kernel_size, padding=9)

    def forward(self, x):
        y = self.pretrained_model(x)['out']
        return self.final_layer(y)

# FINAL MODEL
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

# Five Layer Model
class CustomModel_5(nn.Module):
    def __init__(self, kernel_size=1):
        super().__init__()
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.pretrained_model = deeplabv3_resnet50(weights=weights)
        self.one_layer = nn.Conv2d(21, 32, kernel_size)
        self.two_layer = nn.Conv2d(32, 32, kernel_size)
        self.three_layer = nn.Conv2d(32, 32, kernel_size)
        self.four_layer = nn.Conv2d(32, 32, kernel_size)
        self.final_layer = nn.Conv2d(32, 25, kernel_size)

        self.one_batchnorm = nn.BatchNorm2d(32)
        self.two_batchnorm = nn.BatchNorm2d(32)
        self.three_batchnorm = nn.BatchNorm2d(32)
        self.four_batchnorm = nn.BatchNorm2d(32)

        self.relu = nn.ReLU()


    def forward(self, x):
        y = self.pretrained_model(x)['out']

        y = self.one_layer(y)
        y = self.one_batchnorm(y)
        y = self.relu(y)

        y = self.two_layer(y)
        y = self.two_batchnorm(y)
        y = self.relu(y)

        y = self.three_layer(y)
        y = self.three_batchnorm(y)
        y = self.relu(y)

        y = self.four_layer(y)
        y = self.four_batchnorm(y)
        y = self.relu(y)

        return self.final_layer(y)

# Hyperparameters ################
# LR = 0.0008
# Final Model:
BATCH_SIZE = 30
NUM_EPOCHS = 40
KERNEL_SIZE3 = 5
LR = 0.0001 # for different kernel size!

# LR = 0.0006
# BATCH_SIZE = 35
# NUM_EPOCHS = 10
KERNEL_SIZE1 = 11
##################################

train_loader = DataLoader(
    RUGDDataset('train'),
    batch_size=BATCH_SIZE, shuffle=True,
)
val_loader = DataLoader(
    RUGDDataset('validation'),
    batch_size=BATCH_SIZE, shuffle=False,
)

# ds = RUGDDataset('test')
# test_loader = DataLoader(
#     ds, batch_size=16, shuffle=True,
# )

# model = CustomModel_1(KERNEL_SIZE1).to(device) # 1 layers
model = CustomModel_3(KERNEL_SIZE3).to(device) # 3 layers Final Model
# model = CustomModel_5(KERNEL_SIZE3).to(device) # 5 layers
# model = OnlyModel(KERNEL_SIZE1).to(device) # 0 layers

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
# loss_fn2 = DiceLoss('multiclass') # ADDED FOR DICE LOSS
loss_fn2 = FocalLoss('multiclass') # ADDED FOR FOCAL LOSS

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
        # loss = loss1 # comment when using more losses
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_description(f'{pbar_prefix} loss={running_loss / (i + 1):.4f}')

        # Displays example image and its prediction
        if epoch == 2:
            # Detach tensors from GPU and convert to numpy for plotting
            input_image_display = im[0].cpu().permute(1, 2, 0).numpy() # Convert C,H,W to H,W,C
            predicted_mask_display = pred.argmax(axis=1)[0].cpu().numpy()

            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(input_image_display) # Display original image
            plt.title('Input Image (First Batch, First Epoch)')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(predicted_mask_display, cmap='viridis') # Display predicted mask
            plt.title('Predicted Mask (First Batch, First Epoch)')
            plt.axis('off')
            plt.show()

    if (epoch % 2 == 0) or (epoch == NUM_EPOCHS - 1):
        acc = validation_acc(val_loader, model)

    losses.append(running_loss / len(train_loader))
    accs.append(acc.item())
    print(acc.item())

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

validation_acc(test_loader, model)

# Testing
ds = RUGDDataset('test')
test_loader = DataLoader(
    ds, batch_size=16, shuffle=True,
)

acc = validation_acc(test_loader, model)
print(acc)

# to save model weights:
torch.save(model.state_dict(), "custom_model_3_weights.pth")

# Graphing
fig, ax = plt.subplots()
ax.plot(losses)
ax.set_ylabel('loss')
ax2 = ax.twinx()
ax2.plot(accs, color='orange')
ax2.set_ylabel('val acc')
ax.set_xlabel('epoch')
fig.legend(['train loss', 'val acc'])
