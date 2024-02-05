from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as torch_transforms
from datasets import load_dataset
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms.functional import InterpolationMode

import os
import glob
from PIL import Image

INTERPOLATIONS = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "lanczos": InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose(
        [
            torch_transforms.Resize(size, interpolation=interpolation),
            torch_transforms.CenterCrop(size),
            _convert_image_to_rgb,
            torch_transforms.ToTensor(),
            torch_transforms.Normalize([0.5], [0.5]),
        ]
    )
    return transform


class Imagenette(Dataset):
    def __init__(self, split, class_to_forget=None, transform=None):
        self.dataset = load_dataset("frgfm/imagenette", "160px")[split]
        self.class_to_idx = {
            cls: i for i, cls in enumerate(self.dataset.features["label"].names)
        }
        self.file_to_class = {
            str(idx): self.dataset["label"][idx] for idx in range(len(self.dataset))
        }

        self.class_to_forget = class_to_forget
        self.num_classes = max(self.class_to_idx.values()) + 1
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        image = example["image"]
        label = example["label"]

        if example["label"] == self.class_to_forget:
            label = np.random.randint(0, self.num_classes)

        if self.transform:
            image = self.transform(image)
        return image, label


def setup_model(config, ckpt, device):
    """Loads a model from config and a ckpt
    if config is a path will use omegaconf to load
    """
    if isinstance(config, (str, Path)):
        config = OmegaConf.load(config)

    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    model.cond_stage_model.device = device
    return model


def setup_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", class_to_forget, transform)
    # train_set = Imagenette('train', transform)

    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_ga_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] == class_to_forget]
    # print(len(filtered_data), train_set[0], filtered_data[0])

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_remain_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    filtered_data = [data for data in train_set if data[1] != class_to_forget]
    # print(len(filtered_data), train_set[0], filtered_data[0])

    train_dl = DataLoader(filtered_data, batch_size=batch_size, shuffle=True)
    return train_dl, descriptions


def setup_forget_data(class_to_forget, batch_size, image_size, interpolation="bicubic"):
    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    train_set = Imagenette("train", transform=transform)
    descriptions = [f"an image of a {label}" for label in train_set.class_to_idx.keys()]
    print(train_set[0][1], type(train_set[0][1]))
    filtered_data = [data for data in train_set if data[1] == class_to_forget]
    print(batch_size, len(filtered_data), descriptions)
    train_dl = DataLoader(filtered_data, batch_size=batch_size)
    return train_dl, descriptions


class NSFWDataset(Dataset):

    def __init__(self, img_dir, transform, image_key='jpg', txt_key='txt', caption=None):
        super().__init__()
        self.img_dir = img_dir
        self.all_imgs = glob.glob(os.path.join(self.img_dir, "*.png"))
        if caption is None:
            self.caption = "nudity, naked, erotic, sexual"
        else:
            self.caption = caption
        self.captions = [item.strip() for item in self.caption.split(",")]
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):

        img_name = self.all_imgs[idx]
        image = Image.open(img_name)
        caption_idx = int(img_name.split('/')[-1].split('_')[0])
        text_cond =  self.captions[caption_idx]
        image = self.transform(image).permute(1,2,0) # [HxWxC]

        return {self.image_key: image, self.txt_key: text_cond}



class NotNSFWDataset(Dataset):

    def __init__(self, img_dir, transform, image_key='jpg', txt_key='txt', caption=None):
        super().__init__()
        self.img_dir = img_dir
        self.all_imgs = glob.glob(os.path.join(self.img_dir, "*.png"))
        if caption is None:
            self.caption = "a photo of a person wearing clothes"
        else:
            self.caption = caption
        self.captions = [item.strip() for item in self.caption.split(",")]
        self.image_key = image_key
        self.txt_key = txt_key
        self.transform = transform

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):

        img_name = self.all_imgs[idx]
        image = Image.open(img_name)
        text_cond =  self.captions[0]
        image = self.transform(image).permute(1,2,0) # [HxWxC]

        return {self.image_key: image, self.txt_key: text_cond}


def setup_nsfw_data(batch_size, forget_path, remain_path, image_size, interpolation="bicubic"):

    interpolation = INTERPOLATIONS[interpolation]
    transform = get_transform(interpolation, image_size)

    forget_set = NSFWDataset(img_dir=forget_path, transform=transform)
    forget_dl = DataLoader(forget_set, batch_size=batch_size)

    remain_set = NotNSFWDataset(img_dir=remain_path, transform=transform)
    remain_dl = DataLoader(remain_set, batch_size=batch_size)
    return forget_dl, remain_dl


