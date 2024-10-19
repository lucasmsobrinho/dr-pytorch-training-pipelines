import torch
import torchvision
import multiprocessing
from torchvision import transforms
from torch import nn
import pandas as pd
import numpy as np

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

augmentation = transforms.Compose([
    CustomCrop(min_size=0.6, max_size=0.75),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(30/512, 30/512)),
    transforms.RandomRotation(degrees=360),
    transforms.RandomAffine(degrees=0, shear=18),
    transforms.RandomResizedCrop(size=512, scale=(0.7, 1.3)),
    #GST(),
    #Krizhevsky(),
    transforms.ConvertImageDtype(torch.uint8)],
    transforms.Lambda(lambda x: x.to('cpu')),
)

def augment(df, input_folder="./proc256", output_folder="./proc256"):
     for target in classes:
        subdf = df[df.label==target].name
        n_operations = classes[target]["n_operations"]
        for operation in range(n_operations):
            for img_name in subdf:
                out_name = f"{img_name}_aug_{operation}"
                img = torchvision.io.read_image(f"{input_folder}/{img_name}.jpeg").to('cuda')
                aug = augmentation(img)
                torchvision.io.write_jpeg(aug, f"{output_folder}/{out_name}.jpeg", 100)

if __name__=="__main__":
    df = pd.read_csv("trainLabels.csv", header=None, names=["name", "label"])
    df = df.sample(frac=1)
    input_folder = "./proc256"
    output_folder = "./proc256"
    pool_size = 8

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
        p.map(augment, df_list)

    augs = pd.DataFrame(new_imgs)
    augs = pd.concat([df, augs])
    augs.to_csv('./train_aug_labels.csv', header=False, index=False)