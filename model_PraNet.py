import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import random
import albumentations as A
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import SequentialLR, ConstantLR, CosineAnnealingWarmRestarts
import random
from sklearn.metrics import roc_auc_score, average_precision_score
import scipy.ndimage
import plotly.graph_objects as go
import plotly.io as pio
import math
import torch.utils.model_zoo as model_zoo
import csv
from datetime import datetime


__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b', 'res2net50_v1b_26w_4s']

model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}
class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def res2net50_v1b(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b lib.
    Res2Net-50 refers to the Res2Net-50_v1b_26w_4s.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model
def res2net101_v1b(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net50_v1b_26w_4s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
#         model_state = torch.load('/content/drive/MyDrive/ROAD TO MICCAI/MoNuSeg/MonuSeg notebooks/previous notebooks/res2net50_v1b_26w_4s-3cf99910.pth')
#         model.load_state_dict(model_state)
        model.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model


def res2net101_v1b_26w_4s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net101_v1b_26w_4s']))
    return model


def res2net152_v1b_26w_4s(pretrained=True, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Res2Net(Bottle2neck, [3, 8, 36, 3], baseWidth=26, scale=4, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['res2net152_v1b_26w_4s']))
    return model

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class PraNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(PraNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        # ---- Partial Decoder ----
        self.agg1 = aggregation(channel)
        # ---- reverse attention branch 4 ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)      # bs, 64, 88, 88
        # ---- low-level features ----
        x1 = self.resnet.layer1(x)      # bs, 256, 88, 88
        x2 = self.resnet.layer2(x1)     # bs, 512, 44, 44

        x3 = self.resnet.layer3(x2)     # bs, 1024, 22, 22
        x4 = self.resnet.layer4(x3)     # bs, 2048, 11, 11
        x2_rfb = self.rfb2_1(x2)        # channel -> 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32
        x4_rfb = self.rfb4_1(x4)        # channel -> 32

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map_5 = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear')    # NOTES: Sup-1 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_4 ----
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode='bilinear')
        x = -1*(torch.sigmoid(crop_4)) + 1
        x = x.expand(-1, 2048, -1, -1).mul(x4)
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x))
        x = F.relu(self.ra4_conv3(x))
        x = F.relu(self.ra4_conv4(x))
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(x, scale_factor=32, mode='bilinear')  # NOTES: Sup-2 (bs, 1, 11, 11) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_3 ----
        crop_3 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 1024, -1, -1).mul(x3)
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x))
        x = F.relu(self.ra3_conv3(x))
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(x, scale_factor=16, mode='bilinear')  # NOTES: Sup-3 (bs, 1, 22, 22) -> (bs, 1, 352, 352)

        # ---- reverse attention branch_2 ----
        crop_2 = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = -1*(torch.sigmoid(crop_2)) + 1
        x = x.expand(-1, 512, -1, -1).mul(x2)
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(x, scale_factor=8, mode='bilinear')   # NOTES: Sup-4 (bs, 1, 44, 44) -> (bs, 1, 352, 352)

        # return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2
        return lateral_map_2

class PiCai2DDataset(Dataset):
    def __init__(self, base_dir, sample_ids):
        self.base_dir = base_dir
        self.sample_ids = sample_ids
        self.slice_indices = []

        if not self.sample_ids:
            raise ValueError("The provided list of sample_ids is empty.")

        first_mask_path = os.path.join(self.base_dir, 'mask', f'{self.sample_ids[0]}.npy')
        data_shape = np.load(first_mask_path).shape
        self.slice_axis = np.argmin(data_shape)
        print(f"Data shape detected as {data_shape}. Deduced slice axis to be {self.slice_axis}.")
        print("Initializing dataset with balanced sampling...")

        for sample_idx, sample_id in enumerate(tqdm(self.sample_ids, desc="Finding and Balancing Slices")):
            mask_path = os.path.join(self.base_dir, 'mask', f'{sample_id}.npy')
            mask_3d = np.load(mask_path)
            num_slices_in_volume = mask_3d.shape[self.slice_axis]

            positive_slices = []
            negative_slices = []

            for slice_idx in range(num_slices_in_volume):
                slicer = [slice(None)] * 3
                slicer[self.slice_axis] = slice_idx

                if np.sum(mask_3d[tuple(slicer)]) > 0:
                    positive_slices.append((sample_idx, slice_idx))
                else:
                    negative_slices.append((sample_idx, slice_idx))

            # Add all positive slices
            self.slice_indices.extend(positive_slices)

            # Add an equal number of random negative slices
            num_to_sample = len(positive_slices)
            self.slice_indices.extend(random.sample(negative_slices, min(num_to_sample, len(negative_slices))))

        random.shuffle(self.slice_indices) # Shuffle the final list
        print(f"Found a total of {len(self.slice_indices)} (positive + negative) 2D slices for this dataset partition.")

    def __len__(self):
        return len(self.slice_indices)

    def __getitem__(self, idx):
        local_sample_idx, slice_idx = self.slice_indices[idx]
        sample_id = self.sample_ids[local_sample_idx]

        modalities = []
        for modality in ['t2w', 'adc', 'hbv']:
            img_path = os.path.join(self.base_dir, modality, f'{sample_id}.npy')
            img_data = np.load(img_path)
            modalities.append(img_data)

        img_3d = np.stack(modalities, axis=-1)

        mask_path = os.path.join(self.base_dir, 'mask', f'{sample_id}.npy')
        mask_3d = np.load(mask_path)

        slicer = [slice(None)] * 3
        slicer[self.slice_axis] = slice_idx

        image_slice, mask_slice = img_3d[tuple(slicer)], mask_3d[tuple(slicer)]
        return image_slice, mask_slice

class AugmentationWrapper(Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image_np, mask_np = self.dataset[idx]
        if self.transform:
            augmented = self.transform(image=image_np.astype(np.float32), mask=mask_np.astype(np.float32))
            image_np, mask_np = augmented['image'], augmented['mask']
        for i in range(image_np.shape[2]):
            channel = image_np[:, :, i]
            non_zero_pixels = channel[channel > 1e-6]
            if non_zero_pixels.size > 0:
                p1, p99 = np.percentile(non_zero_pixels, 1), np.percentile(non_zero_pixels, 99)
                channel = np.clip(channel, p1, p99)
            min_val, max_val = channel.min(), channel.max()
            image_np[:, :, i] = (channel - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(channel)
        image = torch.from_numpy(image_np.transpose(2, 0, 1)).float()
        mask = torch.from_numpy(mask_np).unsqueeze(0).float()
        return image, mask



class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
    def forward(self, inputs, targets, smooth=1):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets)
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat, targets_flat = inputs_prob.view(-1), targets.view(-1)
        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)
        return bce_loss + dice_loss

def check_accuracy(loader, model, device="cuda"):
    model.eval()

    all_preds_flat, all_targets_flat = [], []
    pos_preds_flat, pos_targets_flat = [], []

    total_dice_num, total_dice_den = 0, 0
    pos_dice_num, pos_dice_den = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast(device_type=str(device)):
                preds = model(x)
                preds_prob = torch.sigmoid(preds)

            preds_binary = (preds_prob > 0.5).float()
            y_binary = (y > 0.5).float()

            # --- Metrics for ALL slices ---
            all_preds_flat.append(preds_prob.view(-1).cpu().numpy())
            all_targets_flat.append(y_binary.view(-1).cpu().numpy())
            total_dice_num += (2 * (preds_binary * y_binary).sum())
            total_dice_den += (preds_binary.sum() + y_binary.sum())

            # --- Isolate and calculate metrics for POSITIVE slices ---
            # Find which slices in the batch are positive
            is_positive_slice = y_binary.view(y_binary.shape[0], -1).sum(dim=1) > 0
            if is_positive_slice.sum() > 0: # If there are any positive slices in the batch
                pos_preds_prob = preds_prob[is_positive_slice]
                pos_preds_binary = preds_binary[is_positive_slice]
                pos_y_binary = y_binary[is_positive_slice]

                pos_preds_flat.append(pos_preds_prob.view(-1).cpu().numpy())
                pos_targets_flat.append(pos_y_binary.view(-1).cpu().numpy())
                pos_dice_num += (2 * (pos_preds_binary * pos_y_binary).sum())
                pos_dice_den += (pos_preds_binary.sum() + pos_y_binary.sum())

    model.train()

    # Dice Scores
    dice_all = (total_dice_num + 1e-8) / (total_dice_den + 1e-8)
    dice_pos = (pos_dice_num + 1e-8) / (pos_dice_den + 1e-8)

    # AUROC Scores
    auroc_all, auroc_pos = 0.0, 0.0
    try:
        all_preds_np = np.concatenate(all_preds_flat)
        all_targets_np = np.concatenate(all_targets_flat)
        auroc_all = roc_auc_score(all_targets_np, all_preds_np)
    except (ValueError, IndexError):
        print("Warning: Could not compute AUROC for all slices.")

    try:
        if pos_preds_flat: # Check if any positive slices were found at all
            pos_preds_np = np.concatenate(pos_preds_flat)
            pos_targets_np = np.concatenate(pos_targets_flat)
            # Ensure there's more than one class in the targets for AUROC
            if len(np.unique(pos_targets_np)) > 1:
                auroc_pos = roc_auc_score(pos_targets_np, pos_preds_np)
            else:
                print("Warning: Only one class present in positive slice targets. Cannot compute AUROC.")
        else:
            print("Warning: No positive slices found in validation set for AUROC calculation.")
    except (ValueError, IndexError):
        print("Warning: Could not compute AUROC for positive slices.")

    return {
        "dice_all": float(dice_all), "auroc_all": auroc_all,
        "dice_pos": float(dice_pos), "auroc_pos": auroc_pos
    }

def train_fn(loader, model, optimizer, loss_fn, scaler):
    # Track total loss for the epoch
    total_loss = 0.0
    batch_count = 0

    loop = tqdm(loader, desc="Training")
    for data, targets in loop:
        data, targets = data.to(device=DEVICE), targets.to(device=DEVICE)
        with torch.amp.autocast(device_type=str(DEVICE)):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        batch_count += 1
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
    print(f"Average training loss: {avg_loss:.4f}")

    return avg_loss

def visualize_predictions(model, loader, device, num_images=5):
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
    model.to(device)
    model.eval()
    try:
        images, masks = next(iter(loader))
    except StopIteration:
        print("Validation loader is empty. Cannot visualize predictions.")
        return
    images, masks = images.to(device), masks.to(device)
    with torch.no_grad():
        preds = torch.sigmoid(model(images))
        preds_binary = (preds > 0.5).float()
    fig, axes = plt.subplots(num_images, 3, figsize=(15, num_images * 5))
    fig.suptitle("Model Predictions vs. Ground Truth", fontsize=16)
    for i in range(min(num_images, len(images))):
        ax = axes[i, 0]
        ax.imshow(images[i][0].cpu().numpy(), cmap='gray')
        ax.set_title(f"Input Image (T2W)")
        ax.axis("off")
        ax = axes[i, 1]
        ax.imshow(masks[i].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title("Ground Truth Mask")
        ax.axis("off")
        ax = axes[i, 2]
        ax.imshow(preds_binary[i].squeeze().cpu().numpy(), cmap='gray')
        ax.set_title("Predicted Mask")
        ax.axis("off")
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()



def save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, path):
    """
    Save model, optimizer, scheduler states and best metric for resuming training.
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_metric': best_val_metric
    }, path)

def load_checkpoint(path, model, optimizer, scheduler):
    """
    Load model, optimizer, scheduler states and last epoch/best metric.
    Returns: last_epoch, best_val_metric
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    last_epoch = checkpoint['epoch']
    best_val_metric = checkpoint['best_val_metric']
    return last_epoch, best_val_metric

# --- Update main training loop ---
def main():
    train_transform = A.Compose([
        A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE), A.Rotate(limit=20, p=0.7),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
        A.RandomBrightnessContrast(p=0.4), A.GaussNoise(p=0.3),
    ])
    val_transform = A.Compose([A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE)])

    print(f"Using device: {DEVICE}")
    model = PraNet().to(DEVICE)
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler1 = ConstantLR(optimizer, factor=1.0, total_iters=100)
    scheduler2 = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[100])

    try:
        mask_dir = os.path.join(DATA_DIR, 'mask')
        all_sample_ids = sorted([f.replace('.npy', '') for f in os.listdir(mask_dir) if f.endswith('.npy')])
        if not all_sample_ids: raise FileNotFoundError
    except FileNotFoundError:
        print(f"\nERROR: No mask files (.npy) found in {mask_dir}. Check DATA_DIR.")
        return

    random.seed(42)
    random.shuffle(all_sample_ids)
    split_idx = int(len(all_sample_ids) * (1 - VALIDATION_SPLIT))
    train_ids, val_ids = all_sample_ids[:split_idx], all_sample_ids[split_idx:]
    print(f"\nTotal samples: {len(all_sample_ids)}, Training: {len(train_ids)}, Validation: {len(val_ids)}\n")

    train_base_dataset = PiCai2DDataset(DATA_DIR, sample_ids=train_ids)
    val_base_dataset = PiCai2DDataset(DATA_DIR, sample_ids=val_ids)
    train_dataset = AugmentationWrapper(train_base_dataset, transform=train_transform)
    val_dataset = AugmentationWrapper(val_base_dataset, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=False)

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))
    best_val_metric = -1.0
    patience = 10
    patience_counter = 0

    # Prepare log file
    with open(log_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "date_time", "dice_all", "dice_pos", "auroc_all", "auroc_pos", "lr", "total_loss", "model_saved"
        ])

    # Resume training if checkpoint exists
    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        start_epoch, best_val_metric = load_checkpoint(MODEL_SAVE_PATH, model, optimizer, scheduler)
        print(f"Resuming training from epoch {start_epoch+1} with best_val_metric {best_val_metric:.4f}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Current time: {current_time}")

        avg_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        metrics = check_accuracy(val_loader, model, device=DEVICE)
        print("--- Validation Metrics ---")
        print(f"  ALL Slices:      Dice = {metrics['dice_all']:.4f}")
        print(f"  LESION Slices:  Dice = {metrics['dice_pos']:.4f}")

        current_metric = metrics['dice_pos']
        model_saved = False
        if current_metric > best_val_metric:
            best_val_metric = current_metric
            save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, MODEL_SAVE_PATH)
            torch.save(model.state_dict(), MODEL_SAVE_PATH2)
            print(f"==> New best model saved with Lesion Dice Score: {best_val_metric:.4f}")
            patience_counter = 0
            model_saved = True
        else:
            patience_counter += 1
            print(f"No improvement. Early stop patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break

        scheduler.step()
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.8f}")

        # Save metrics to log file
        with open(log_file, "a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                current_time,
                metrics['dice_all'],
                metrics['dice_pos'],
                metrics['auroc_all'],
                metrics['auroc_pos'],
                optimizer.param_groups[0]['lr'],
                avg_loss,
                int(model_saved)
            ])
#
def test():
    print('testing the code')
    x = torch.randn((2, 3, 224,224)) # batch, channel, H, W    
    model = PraNet()
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total__trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('total number of parameters: {}'.format(pytorch_total_params))
    print('total number of trainable parameters: {}'.format(pytorch_total__trainable_params))
    model.eval()
    model.cuda()
    with torch.no_grad():
      preds3 = model(x.cuda())
    print(f'Input shape : {x.shape}')
    print(f'Output shape : {preds3.shape}')

test()


LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 100
NUM_WORKERS = 0
IMAGE_SIZE = 384
PIN_MEMORY = True
VALIDATION_SPLIT = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


DATA_DIR = "input/processed_resampled3/"
base_dir = "models/model_PRANET3/"
MODEL_SAVE_PATH = base_dir + "best_pranet_model_balanced_eval.pth"
MODEL_SAVE_PATH2 = base_dir + "best_pranet_model_balanced_eval_state.pth"
log_file = base_dir + "training_log.csv"

os.makedirs(base_dir, exist_ok=True)

# print all
print(f"Using device: {DEVICE}")
print(f"Data directory: {DATA_DIR}")
print(f"Model save path: {MODEL_SAVE_PATH}")
print(f"Log file: {log_file}")
print(f"Batch size: {BATCH_SIZE}")
print(f"Number of epochs: {NUM_EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Validation split: {VALIDATION_SPLIT}")
print(f"Number of workers: {NUM_WORKERS}")
print(f"Pin memory: {PIN_MEMORY}")
print(f"Learning rate: {LEARNING_RATE}")

# current time

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current time: {current_time}")  

main()

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"Current time: {current_time}")
