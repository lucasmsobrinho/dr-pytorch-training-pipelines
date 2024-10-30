import torch
import torchvision
from torchvision import transforms
import multiprocessing
import cv2
import pandas as pd
import numpy as np
import os

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

def subtract_local_avg_color(img):
    k = 51
    s = 256/30
    return np.clip(0.5 + 3*(img-transforms.GaussianBlur((k,k), sigma=s)(img)), 0, 1)

def mask_outer(img):
    img_size = 256
    base = np.zeros((img_size,img_size, 3), dtype=np.float32)
    cv2.circle(base, 
            center = (img_size//2, img_size//2), 
            radius = int(0.9*img_size/2),
            color = (1, 1, 1),
            thickness = -1)
    base = torch.tensor(base).permute(2,0,1)
    return base*img + (1-base)*.5

def adjust_radius(img):
    x = img[:, img.shape[1]//2,:].sum(0)
    r = (x > x.mean()/10).sum()//2
    scale = 128/r
    return transforms.functional.affine(img, scale=scale, translate=[0,0], angle=0, shear=0)

def get_transform(img_size=512):
    preprocess = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(threshold), # threshold
        transforms.Lambda(green_channel), # green channel
        transforms.Lambda(CLAHE), # CLAHE
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to('cuda')),
        transforms.GaussianBlur((5,5)),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Lambda(lambda x: x.to('cpu')),
    ])
    return preprocess

def get_transform_kaggle1(img_size=256):
    preprocess = transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.Lambda(adjust_radius),
        transforms.CenterCrop(img_size),
        transforms.Lambda(subtract_local_avg_color),
        transforms.Lambda(mask_outer),
        transforms.ConvertImageDtype(torch.uint8),
    ])
    return preprocess

# TODO: change it to class-based.
def process_jabbar(df, input_folder="./train", output_folder="./proc256", img_size=512):
    transform = get_transform(img_size=img_size)

    for idx, img_name in enumerate(df.name):
        if (idx % 1000 == 0):
            print(f"{idx}/{df.name.shape[0]}, {img_name}, {output_folder}/{img_name}.jpeg")

        if(not os.path.exists(f"{output_folder}/{img_name}.jpeg")):
            img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to('cuda')
            proc = transform(img)
            torchvision.io.write_jpeg(proc, f"{output_folder}/{img_name}.jpeg", 100)

def process_kaggle(df, input_folder="./train", output_folder="./kaggle256", img_size=256):
    # scale radius to be equal
    # subtract average color
    # clip images to 90% to remove "boundary effects"
    transform = get_transform_kaggle(img_size=img_size)

    for idx, img_name in enumerate(df.name):
        if (idx % 1000 == 0):
            print(f"{idx}/{df.name.shape[0]}, {img_name}, {output_folder}/{img_name}.jpeg")

        if(not os.path.exists(f"{output_folder}/{img_name}.jpeg")):
            img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to('cuda')
            proc = transform(img)
            torchvision.io.write_jpeg(proc, f"{output_folder}/{img_name}.jpeg", 100)


if __name__=="__main__":
    labels_path="./sample.csv"
    input_folder="./sample"
    output_folder="./kaggle256"
    img_size = 256
    pool_size = 8
    process = process_kaggle

    df = pd.read_csv(labels_path, header=None, names=["name", "label"])

    chunk_size = len(df)//pool_size
    chunk_limit = [chunk_size*i for i in range(pool_size+1)]
    chunk_limit[-1] = len(df)+1
    print(len(df))
    print(chunk_limit)
    df_list = [df[chunk_limit[i]:chunk_limit[i+1]] for i in range(pool_size)]

    with multiprocessing.Pool(pool_size) as p:
        p.map(process, df_list)
