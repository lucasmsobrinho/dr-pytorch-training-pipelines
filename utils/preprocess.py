import torch
import torchvision
from torchvision import transforms
import multiprocessing
import cv2
import pandas as pd
import numpy as np

import os
import functools
import argparse


def get_device():
    """Get the device to use for computations (CUDA or MPS if available, else CPU)"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps') 
    return torch.device('cpu')
device = get_device()

class NoCircleException(Exception):
    pass

class CutomLambda(torchvision.transforms.Lambda):
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

# TODO: 
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
    base = torch.tensor(base).permute(2,0,1).to(device)
    return base*img + (1-base)*.5


def adjust_radius(img, img_size=512):
    x = img[:, img.shape[1]//2,:].sum(0)
    r_x = (x > x.mean()/10).sum()//2
    r_y = img.shape[1]//2
    r = min(r_x, r_y)
    scale = img_size/(2*r)
    return transforms.functional.affine(img, scale=scale, translate=[0,0], angle=0, shear=0)


def adjust_radius_center(image, img_size=512, scale_factor=4):
    '''
        This function resize the image to a small one with scale_factor
        in order to save processing and time.
        Then, it runs Hough Circles to find the best circle that represent the 
        retina image.
        Finally, it scale and crop down to the new dimension img_size x img_size
    '''
    # Scale down
    small = transforms.Resize((image.shape[1]//scale_factor, image.shape[2]//scale_factor))(image)
    gray = small[1].cpu().numpy().astype(np.uint8)

    # Perform Hough Circle Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,       # Inverse ratio of the accumulator resolution
        minDist=100,   # Minimum distance between detected centers
        param1=50,    # Higher threshold for the Canny edge detector
        param2=30,    # Threshold for center detection
        minRadius=int(min(gray.shape)*0.4), # Minimum circle radius
        maxRadius=max(gray.shape)//2 # Maximum circle radius
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        xc, yc, r = circles[:1][0]
    else:
        raise NoCircleException("No circles found")

    # get reescaled circle
    xc *= scale_factor
    yc *= scale_factor
    r *= scale_factor

    # Display the result
    scale = img_size/(2*r)
    c, y, x = image.shape
    dx = (x//2 - xc) * scale
    dy = (y//2 - yc) * scale
    translate = [dx, dy]

    return transforms.functional.affine(image, scale=scale, translate=translate, angle=0, shear=0)

def local_avg_and_mask(img, img_size=512, retention=0.9):
    img = torchvision.io.read_image(path)

    img = transforms.ConvertImageDtype(torch.float32)(img)
    mask = (img[1]>10/255).numpy().astype(np.uint8)
    mask_y = mask.sum(1)

    bounds = np.where(mask_y > 0)[0]
    mask_top = bounds[0]
    mask_bottom = bounds[-1]

    base = np.zeros((img_size, img_size), dtype=np.float32)
    cv2.circle(base,
            center = (img_size//2, img_size//2),
            radius = int(retention*img_size/2),
            color = (1,1,1),
            thickness = -1)

    d = int((img_size/2)*(1-retention))
    base[:mask_top+d,:] = 0.0
    base[mask_bottom-d:,:] = 0.0

    k = 51
    s = img_size/30
    out = torch.clip(0.5 + 3*(img-transforms.GaussianBlur((k,k), sigma=s)(img)), 0, 1)

    show(out*base, path, True)


def transform_vanilla(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu'))
    ])

def transform_scale_and_crop(img_size=512):
    return transforms.Compose([
        CutomLambda(adjust_radius_center, img_size=img_size),
        transforms.CenterCrop(img_size),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu'))
    ])


def transform_kaggle(img_size=512):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        CutomLambda(local_avg_and_mask, img_size=img_size),
        transforms.ConvertImageDtype(torch.uint8),
        CutomLambda(lambda x: x.to('cpu')),
    ])

def transform_kaggle_pre_scaled(img_size):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        CutomLambda(local_avg_and_mask, img_size=img_size),
        transforms.ConvertImageDtype(torch.uint8),
        CutomLambda(lambda x: x.to('cpu')),
    ])


def transform_jabbar(img_size=512):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Lambda(threshold),
        transforms.Lambda(green_channel),
        transforms.Lambda(CLAHE), # overhead because cv2 operates on CPU
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.to(device)),
        transforms.GaussianBlur((5,5)),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.repeat(3,1,1)),
        transforms.Lambda(lambda x: x.to('cpu')),
    ])

def process(df, proc_name="vanilla", img_size=256, input_folder="./train", output_folder="./proc256"):
    transform = get_proc(proc_name, img_size)
    
    if not os.path.exists(output_folder):
        try:
            os.mkdir(output_folder)
        except FileExistsError:
            # never thought racing conditions would be so obvious
            print("probably a sibling process ended up creating it already.")

    for idx, img_name in enumerate(df.name):
        if (idx % 1000 == 0):
            print(f"{idx}/{df.name.shape[0]}, {img_name}, {output_folder}/{img_name}.jpeg")

        if(not os.path.exists(f"{output_folder}/{img_name}.jpeg")):
            img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to(device)
            try:
                proc = transform(img)
                torchvision.io.write_jpeg(proc, f"{output_folder}/{img_name}.jpeg", 100)
            except NoCircleException as e:
                print(img_name)


def get_proc(name, img_size):
    # workaround function to avoid multiprocessing bugs when dealing with lambda functions
    proc_map = {
        "vanilla": transform_vanilla(img_size),
        "scale_crop": transform_scale_and_crop(img_size),
        "kaggle": transform_kaggle(img_size),
        "kaggle_pre_scaled": transform_kaggle_pre_scaled(img_size),
        "jabbar": transform_jabbar(img_size),
    }
    return proc_map[name]


if __name__=="__main__":
    multiprocessing.set_start_method("spawn")
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for image datasets"
    )

    parser.add_argument('-p', '--proc_name', default="vanilla", type=str,
                      choices=("vanilla", "scale_crop", "kaggle", "kaggle_pre_scaled", "jabbar"),
                      help='config file path (default: "vanilla")')
    parser.add_argument('-l', '--labels_path', default="./sample.csv", type=str,
                      help='config file path (default: "./sample.csv")')
    parser.add_argument('-i', '--input_folder', default="./train", type=str,
                      help='source folder for input images (default: "./train")')
    parser.add_argument('-o', '--output_folder', default="./proc", type=str,
                      help='target folder for processed images (default: "./proc")')
    parser.add_argument('-s', '--img_size', default=256, type=int,
                      help='output image size (default: 256)')
    parser.add_argument('--pool_size', default=1, type=int,
                      help='pool size for parallelization (default: 1)')
    
    args = parser.parse_args()
    
    proc_name = args.proc_name
    labels_path = args.labels_path
    input_folder = args.input_folder
    output_folder= args.output_folder
    img_size = args.img_size

    pool_size = args.pool_size

    _process = functools.partial(process,
                                proc_name=proc_name,
                                img_size=img_size,
                                input_folder=input_folder, 
                                output_folder=output_folder)

    df = pd.read_csv(labels_path, header=1, names=["name", "level"])


    if pool_size > 1 :
        print(f"{pool_size=}, {len(df)=}")
        chunk_size = len(df)//pool_size
        chunk_limit = [chunk_size*i for i in range(pool_size+1)]
        chunk_limit[-1] = len(df)+1
        print(f"{chunk_limit=}")
        df_chunks = [df[chunk_limit[i]:chunk_limit[i+1]] for i in range(pool_size)]

        with multiprocessing.Pool(pool_size) as p:
            p.map(_process, df_chunks)
    else:
        _preprocess(df)
