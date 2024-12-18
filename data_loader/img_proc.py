import torchvision.transforms as transforms
import torch
import cv2
import numpy as np

# utility classes and functions for transforms and processing pipelines

class CustomCrop(torch.nn.Module):
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


def mask_outer(img, img_size=512):
    base = np.zeros((img_size, img_size, 3), dtype=np.float32)
    cv2.circle(base,
            center = (img_size//2, img_size//2),
            radius = int(0.9*img_size/2),
            color = (1, 1, 1),
            thickness = -1)
    base = torch.tensor(base).permute(2,0,1).to(img.device)
    return base*img + (1-base)*.5


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
    gray = small[1].cpu().numpy()

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


# transformations utils

def transform_scale_and_crop(img_size=512):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        CutomLambda(adjust_radius_center, img_size=img_size),
        transforms.CenterCrop(img_size),
    ])

# preprocessing functions used in train and test
def preprocess_eyepacs():
    return transforms.Compose([
        transforms.Normalize(mean=[0.4466, 0.3089, 0.2198], std=[0.2080, 0.1455, 0.1045])
    ])

def preprocess_eyepacs_kaggle():
    return transforms.Compose([
        transforms.Normalize(mean=[0.5037, 0.5010, 0.5000], std=[0.0650, 0.0681, 0.0500])
    ])

def preprocess_imagenet():
    return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def preprocess_cifar10(resize=False, img_size=224):
    t_list = []
    if resize:
        t_list.append(transforms.Resize(size=(img_size, img_size)))

    t_list.extend([transforms.ToTensor(),
                   transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ])
    return transforms.Compose(t_list)

def preprocess_vit(resize=False, img_size=224):
    if resize:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
        ])

def preprocess_swin(resize=False, img_size=224):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet mean and std
    ])

def augmentation_cifar10():
    return transforms.Compose([
                transforms.RandomAffine(degrees=30, translate=[0.1, 0.1], scale=(0.9, 1.1), shear=18),
                transforms.RandomHorizontalFlip(0.5)
        ])


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
        transforms.ConvertImageDtype(torch.uint8)
    ])


def augmentation_kaggle(img_size=256):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
        CutomLambda(mask_outer, img_size=img_size)
    ])


def augmentation_simple(img_size=256):
    return transforms.Compose([
        transforms.ConvertImageDtype(torch.float32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=360),
    ])

def none():
    return None