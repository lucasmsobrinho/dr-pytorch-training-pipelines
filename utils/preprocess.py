import torch
import torchvision
from torchvision import transforms
import multiprocessing
import cv2
import pandas as pd
import numpy as np

import os
import functools

class Lambda(torchvision.transforms.Lambda):
    """
        Lambda Class that accept parameters
    """ 
    def __init__(self, lambd, **kwargs):
        super().__init__(lambd)
        self.lambd = lambd
        self.kwargs = kwargs

    def __call__(self, img):
        return self.lambd(img, **self.kwargs)


def class_reduction_transform(new_number_classes):
    target_transform=transforms.Compose(
                                    lambda x: class_reduction(x, new_number_classes))
    return target_transform


def class_reduction(x, new_number_classes):
    if x > new_number_classes - 1:
        return new_number_classes - 1
    else:
        return x


def crop_best_square(img):
    size = min(img.shape[0], img.shape[1])
    best_x = 0
    best_y = 0
    img = img[best_y:best_y+size, best_x:best_x+size]
    return img


def threshold(img, thresh=10):
    img[img < thresh] = 0
    return img


def green_channel(img):
    return img[1]


def CLAHE(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    img = img.cpu().numpy().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img)
    return img_clahe


def subtract_local_avg_color(img, img_size=512):
    k = 51
    s = img_size/30
    return torch.clip(0.5 + 3*(img-transforms.GaussianBlur((k,k), sigma=s)(img)), 0, 1)


def mask_outer(img, img_size=512):
    base = np.zeros((img_size, img_size, 3), dtype=np.float32)
    cv2.circle(base,
            center = (img_size//2, img_size//2),
            radius = int(0.9*img_size/2),
            color = (1, 1, 1),
            thickness = -1)
    base = torch.tensor(base).permute(2,0,1).to('cuda')
    return base*img + (1-base)*.5


def adjust_radius(img, img_size=512):
    x = img[:, img.shape[1]//2,:].sum(0)
    r_x = (x > x.mean()/10).sum()//2
    r_y = img.shape[1]//2
    r = min(r_x, r_y)
    scale = img_size/(2*r)
    return transforms.functional.affine(img, scale=scale, translate=[0,0], angle=0, shear=0)


def adjust_radius_center(img, img_size=512):
    g = img[1]
    thresh = g.mean()/10

    x_fg = (g > thresh).sum(1)
    x = x_fg.shape[0]
    x_center = (x_fg.argmax().item() + x - x_fg.flip(0).argmax().item())//2

    y_fg = (g > thresh).sum(0)
    y = y_fg.shape[0]
    y_center = (y_fg.argmax().item() + y - y_fg.flip(0).argmax().item())//2

    r = min(y_fg.max(), x_fg.max())

    scale = img_size/(r)

    dx = (x//2 - x_center) * scale
    dy = (y//2 - y_center) * scale
    translate = [dy, dx]
    return transforms.functional.affine(img, scale=scale, translate=translate, angle=0, shear=0)


def transform_vanilla(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu'))
    ])


def transform_scale_and_crop(img_size=512):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        Lambda(adjust_radius_center, img_size=img_size),
        transforms.CenterCrop(img_size),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu'))
    ])


def transform_kaggle(img_size=512):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        Lambda(adjust_radius_center, img_size=img_size),
        transforms.CenterCrop(img_size),
        Lambda(subtract_local_avg_color, img_size=img_size),
        Lambda(mask_outer, img_size=img_size),
        transforms.ConvertImageDtype(torch.uint8),
        Lambda(lambda x: x.to('cpu')),
    ])


def transform_jabbar(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(threshold),
        transforms.Lambda(green_channel),
        transforms.Lambda(CLAHE), # overhead because cv2 operates on CPU
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to('cuda')),
        transforms.GaussianBlur((5,5)),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Lambda(lambda x: x.to('cpu')),
    ])


def process(df, transform, input_folder="./train", output_folder="./proc256"):
    for idx, img_name in enumerate(df.name):
        if (idx % 1000 == 0):
            print(f"{idx}/{df.name.shape[0]}, {img_name}, {output_folder}/{img_name}.jpeg")

        if(not os.path.exists(f"{output_folder}/{img_name}.jpeg")):
            img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to('cuda')
            proc = transform(img)
            torchvision.io.write_jpeg(proc, f"{output_folder}/{img_name}.jpeg", 100)


if __name__=="__main__":
    labels_path = "./sample.csv"
    input_folder="./train"
    output_folder="./kaggle256"
    img_size = 256
    proc_name = "kaggle"

    pool_size = 8

    proc_map = {
        "vanilla": transform_vanilla(img_size),
        "scale_crop": transform_scale_and_crop(img_size),
        "kaggle": transform_kaggle(img_size),
        "jabbar": transform_jabbar(img_size)
    }

    _process = functools.partial(process, 
                                transform=proc_map[proc_name],
                                input_folder=input_folder, 
                                output_folder=output_folder)

    df = pd.read_csv(labels_path, header=None, names=["name", "label"])

    chunk_size = len(df)//pool_size
    chunk_limit = [chunk_size*i for i in range(pool_size+1)]
    chunk_limit[-1] = len(df)+1
    print(len(df))
    print(chunk_limit)
    df_chunks = [df[chunk_limit[i]:chunk_limit[i+1]] for i in range(pool_size)]

    with multiprocessing.Pool(pool_size) as p:
        p.map(_process, df_chunks)
