import torch
import torchvision
import multiprocessing
from torchvision import transforms
from torch import nn
import pandas as pd
import numpy as np
import cv2 

import functools
import argparse

class CutomLambda(transforms.Lambda):
    """
        Lambda Class that accept parameters
    """ 
    def __init__(self, lambd, *args, **kwargs):
        super().__init__(lambd)
        self.lambd = lambd
        self.args = args
        self.kwargs = kwargs

    def __call__(self, img):
        return self.lambd(img, *self.args, **self.kwargs)
    

class CustomCrop(nn.Module):
    def __init__(self, min_size, max_size):
        super(CustomCrop, self).__init__()
        self.min_size = min_size
        self.max_size = max_size

    def forward(self, img):
        p = np.random.uniform()
        proportion = np.random.uniform(self.min_size, self.max_size)
        size = int(img.shape[1] * proportion)
        crop = transforms.CenterCrop(size)(img) if (p < 0.5) else img
        return crop

class GST(nn.Module):
    def __init__(self):
        super(GST, self).__init__()

    def forward(self, x):
        # não entendi exatamente a ideia por detrás da implementação
        return x # NotImplementedError()

class Krizhevsky(nn.Module):
    def __init__(self):
        super(Krizhevsky, self).__init__()

    def forward(self, x):
        #
        return x # NotImplementedError()

classes = {
    0: {'name': 'Normal', 'count': 25810, 'n_operations': 0},
    1: {'name': 'Mild', 'count': 2443, 'n_operations': 11},
    2: {'name': 'Moderate', 'count': 5292, 'n_operations': 5},
    3: {'name': 'Severe', 'count': 873, 'n_operations': 29},
    4: {'name': 'PDR', 'count': 708, 'n_operations': 36}
}

def mask_outer(img, img_size=512):
    base = np.zeros((img_size, img_size, 3), dtype=np.float32)
    cv2.circle(base,
            center = (img_size//2, img_size//2),
            radius = int(0.9*img_size/2),
            color = (1, 1, 1),
            thickness = -1)
    base = torch.tensor(base).permute(2,0,1).to('cuda')
    return base*img + (1-base)*.5


def augmentation_jabbar(img_size=256):
    return transforms.Compose([
        CustomCrop(min_size=0.6, max_size=0.75),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(30/img_size, 30/img_size)),
        transforms.RandomRotation(degrees=360),
        transforms.RandomAffine(degrees=0, shear=18),
        transforms.RandomResizedCrop(size=img_size, scale=(0.7, 1.3)),
        #GST(),
        #Krizhevsky(),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu')),
    ])


def augmentation_kaggle(img_size=256):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        CutomLambda(mask_outer, img_size=img_size),
        transforms.ConvertImageDtype(torch.uint8),
        transforms.Lambda(lambda x: x.to('cpu')),
    ])


def augment(df, proc_name="vanilla", img_size=256, input_folder="./kaggle256", output_folder="./kaggle256"):
     transform = get_proc(proc_name, img_size)

     for target in classes:
        subdf = df[df.label==target].name
        n_operations = classes[target]["n_operations"]
        for operation in range(n_operations):
            if operation%1000 == 0:
                print(f"{operation}/{n_operations}")
            for img_name in subdf:
                out_name = f"{img_name}_aug_{operation}"
                img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to('cuda')
                aug = transform(img)
                torchvision.io.write_jpeg(aug, f"{output_folder}/{out_name}.jpeg", 100)

def get_proc(name, img_size):
    # workaround function to avoid multiprocessing bugs when dealing with lambda functions
    proc_map = {
        "jabbar": augmentation_jabbar(img_size),
        "kaggle": augmentation_kaggle(img_size)
    }
    return proc_map[name]


if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description="Preprocessing pipeline for image datasets"
    )

    parser.add_argument('-p', '--proc_name', default="kaggle", type=str,
                      choices=("kaggle", "jabbar"),
                      help='augmentation to run (default: "kaggle")')
    parser.add_argument('-l', '--labels_path', default="./train_labels.csv", type=str,
                      help='input labels file path (default: "./train_labels.csv")')
    parser.add_argument('-l', '--output_labels_path', default="./train_aug_labels.csv", type=str,
                      help='output labels file path (default: "./train_aug_labels.csv")')
    parser.add_argument('-i', '--input_folder', default="./proc", type=str,
                      help='source folder for input images (default: "./proc")')
    parser.add_argument('-o', '--output_folder', default="./proc", type=str,
                      help='target folder for processed images (default: "./proc")')
    parser.add_argument('-s', '--img_size', default=256, type=int,
                      help='output image size (default: 256)')
    parser.add_argument('--pool_size', default=4, type=int,
                      help='pool size for parallelization (default: 4)')
    
    args = parser.parse_args()

    proc_name = args.proc_name
    labels_path = args.labels_path
    output_labels_path = args.output_labels_path
    input_folder = args.input_folder
    output_folder= args.output_folder
    img_size = args.img_size
    
    pool_size = args.pool_size


    _augment = functools.partial(augment,
                                proc_name=proc_name,
                                img_size=img_size,
                                input_folder=input_folder, 
                                output_folder=output_folder)

    df = pd.read_csv(labels_path, header=None, names=["name", "label"])
    df = df.sample(frac=1)

    new_imgs = {'name':[], 'label':[]}
    
    for target in classes:
        subdf = df[df.label==target].name
        n_operations = classes[target]["n_operations"]
        for operation in range(n_operations):
            for img_name in subdf:
                out_name = f"{img_name}_aug_{operation}"
                new_imgs['name'].append(out_name)
                new_imgs['label'].append(target)


    chunk_size = len(df)//pool_size
    chunk_limit = [chunk_size*i for i in range(pool_size+1)]
    chunk_limit[-1] = len(df)+1
    print(len(df))
    print(chunk_limit)
    df_list = [df[chunk_limit[i]:chunk_limit[i+1]] for i in range(pool_size)]

    with multiprocessing.Pool(pool_size) as p:
        p.map(_augment, df_list)

    augs = pd.DataFrame(new_imgs)
    augs = pd.concat([df, augs])
    augs.to_csv(output_labels_path, header=False, index=False)

