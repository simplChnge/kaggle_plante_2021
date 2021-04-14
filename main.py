package_paths = [
    '../input/pytorch-image-library/pytorch-image-models-master/pytorch-image-models-master',
]
import sys;

for pth in package_paths:
    sys.path.append(pth)
	
import pandas as pd
import numpy as np
import cv2
import timm
import torch
import torch.nn as nn
import albumentations as A
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import torchmetrics

from torch.utils.data import Dataset, DataLoader
from albumentations.core.composition import Compose, OneOf
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import Callback
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import StratifiedKFold

DEBUG = False

class CFG:
    seed = 42
    model_name = 'resnet50'
    pretrained = True
    img_size = 512
    num_classes = 6
    lr = 1e-4
    max_lr = 1e-3
    pct_start = 0.3
    div_factor = 1.0e+3
    final_div_factor = 1.0e+3
    num_epochs = 30
    batch_size = 32
    accum = 4
    precision = 16
    n_fold = 5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	
	
	
PATH = "../input/plant-pathology-2021-fgvc8/"
​
# TRAIN_DIR = PATH + 'train_images/'
TRAIN_DIR = "../input/resized-plant2021/img_sz_640/"
TEST_DIR = PATH + 'test_images/'

df_all = pd.read_csv(PATH + "train.csv")
if DEBUG == True:
    df_all = df_all[:100]
    CFG.num_epochs = 30

df_all.shape

from collections import defaultdict


dct = defaultdict(list)

for i, label in enumerate(df_all.labels):
    for category in label.split():
        dct[category].append(i)
 
dct = {key: np.array(val) for key, val in dct.items()}
dct

new_df = pd.DataFrame(np.zeros((df_all.shape[0], len(dct.keys())), dtype=np.int8), columns=dct.keys())

for key, val in dct.items():
    new_df.loc[val, key] = 1

new_df.head()

df_all = pd.concat([df_all, new_df], axis=1)
df_all.to_csv('better_train.csv', index = False)
df_all.head()

sfk = StratifiedKFold(CFG.n_fold)
for train_idx, valid_idx in sfk.split(df_all['image'], df_all['labels']):
    df_train = df_all.iloc[train_idx]
    df_valid = df_all.iloc[valid_idx]
    break
    
print(f"train size: {len(df_train)}")
print(f"valid size: {len(df_valid)}")

class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.image_id = df['image'].values
        self.labels = df.iloc[:, 2:].values
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_id = self.image_id[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        image_path = TRAIN_DIR + image_id
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = self.transform(image=image)
        image = augmented['image']
        return {'image':image, 'target': label}
		
def get_transform(phase: str):
    if phase == 'train':
        return Compose([
            A.RandomResizedCrop(height=CFG.img_size, width=CFG.img_size),
            A.Flip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.HueSaturationValue(p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.5),
            ], p=0.5),
            A.OneOf([
                A.Blur(p=0.1),
                A.GaussianBlur(p=0.1),
                A.MotionBlur(p=0.1),
            ], p=0.1),
            A.OneOf([
                A.GaussNoise(p=0.1),
                A.ISONoise(p=0.1),
                A.GridDropout(ratio=0.5, p=0.2),
                A.CoarseDropout(max_holes=16, min_holes=8, max_height=16, max_width=16, min_height=8, min_width=8, p=0.2)
            ], p=0.2),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return Compose([
            A.Resize(height=CFG.img_size, width=CFG.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
		
train_dataset = PlantDataset(df_train, get_transform('train'))
valid_dataset = PlantDataset(df_valid, get_transform('valid'))

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)

CFG.steps_per_epoch = len(train_loader)

import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
			
class CustomResNet(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        in_features = self.model.get_classifier().in_features
#         self.model.fc = nn.Linear(in_features, CFG.num_classes)
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features, CFG.num_classes)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class LitCassava(pl.LightningModule):
    def __init__(self, model):
        super(LitCassava, self).__init__()
        self.model = model
#         self.metric = pl.metrics.F1(num_classes=CFG.num_classes)
        self.metric = torchmetrics.F1(CFG.num_classes, average='weighted')
#         self.criterion = nn.BCELoss()
        self.criterion = FocalLoss()
        self.sigmoid = nn.Sigmoid()
        self.lr = CFG.lr

    def forward(self, x, *args, **kwargs):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CFG.t_max, eta_min=CFG.min_lr)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer, 
                                                             epochs=CFG.num_epochs, steps_per_epoch=CFG.steps_per_epoch,
                                                             max_lr=CFG.max_lr, pct_start=CFG.pct_start, 
                                                             div_factor=CFG.div_factor, final_div_factor=CFG.final_div_factor)


        return [self.optimizer], [{'scheduler': self.scheduler, 'interval': 'step'}]

    def training_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']
        output = self.model(image)
        output = self.sigmoid(output)
        loss = self.criterion(output, target)
        score = self.metric(output, target)
        logs = {'train_loss': loss, 'train_f1': score, 'lr': self.optimizer.param_groups[0]['lr']}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        target = batch['target']
        output = self.model(image)
        output = self.sigmoid(output)
        loss = self.criterion(output, target)
        score = self.metric(output, target)
        logs = {'valid_loss': loss, 'valid_f1': score}
        self.log_dict(
            logs,
            on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        return loss
		
model = CustomResNet(model_name=CFG.model_name, pretrained=CFG.pretrained)
lit_model = LitCassava(model.model)

logger = CSVLogger(save_dir='logs/', name=CFG.model_name)
logger.log_hyperparams(CFG.__dict__)
checkpoint_callback = ModelCheckpoint(monitor='valid_f1',
                                      save_top_k=1,
                                      save_last=True,
                                      save_weights_only=True,
                                      filename='{epoch:02d}-{valid_loss:.4f}-{valid_f1:.4f}',
                                      verbose=False,
                                      mode='max')

trainer = Trainer(
    max_epochs=CFG.num_epochs,
    gpus=1,
    accumulate_grad_batches=CFG.accum,
    precision=CFG.precision,
#     callbacks=[EarlyStopping(monitor='valid_loss', patience=3, mode='min')],
    checkpoint_callback=checkpoint_callback,
    logger=logger,
    weights_summary='top',
)

trainer.fit(lit_model, train_dataloader=train_loader, val_dataloaders=valid_loader)

metrics = pd.read_csv(f'{trainer.logger.log_dir}/metrics.csv')

train_acc = metrics['train_f1'].dropna().reset_index(drop=True)
valid_acc = metrics['valid_f1'].dropna().reset_index(drop=True)
    
fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_acc, color="r", marker="o", label='train/f1')
plt.plot(valid_acc, color="b", marker="x", label='valid/f1')
plt.ylabel('F1', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='lower right', fontsize=18)
plt.savefig(f'{trainer.logger.log_dir}/f1.png')

train_loss = metrics['train_loss'].dropna().reset_index(drop=True)
valid_loss = metrics['valid_loss'].dropna().reset_index(drop=True)

fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(train_loss, color="r", marker="o", label='train/loss')
plt.plot(valid_loss, color="b", marker="x", label='valid/loss')
plt.ylabel('Loss', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='upper right', fontsize=18)
plt.savefig(f'{trainer.logger.log_dir}/loss.png')\

lr = metrics['lr'].dropna().reset_index(drop=True)

fig = plt.figure(figsize=(7, 6))
plt.grid(True)
plt.plot(lr, color="g", marker="o", label='learning rate')
plt.ylabel('LR', fontsize=24)
plt.xlabel('Epoch', fontsize=24)
plt.legend(loc='upper right', fontsize=18)
plt.savefig(f'{trainer.logger.log_dir}/lr.png')

