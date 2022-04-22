from os import listdir
from os.path import join

from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, InterpolationMode

import random

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        ToTensor(),
    ])


def train_lr_transform(crop_size, upscale_factor):
    return Compose([
        ToPILImage(),
        Resize(crop_size // upscale_factor, interpolation=InterpolationMode.BICUBIC),
        ToTensor()
    ])


def display_transform():
    return Compose([
        ToPILImage(),
        Resize(400),
        CenterCrop(400),
        ToTensor()
    ])


class PairedTrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, lr_dataset_dir, crop_size, upscale_factor):
        super(PairedTrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in sorted(listdir(dataset_dir)) if is_image_file(x)]
        self.lr_image_filenames = [join(lr_dataset_dir, x) for x in sorted(listdir(lr_dataset_dir)) if is_image_file(x)]

        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
        # self.lr_transform = train_hr_transform(crop_size)
        self.hr_crop_size = crop_size
        self.lr_crop_size = crop_size // upscale_factor
        self.upscale_factor = upscale_factor

    def __getitem__(self, index):
        # hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        # lr_image = self.lr_transform(hr_image)
        # lr_image = self.lr_transform(Image.open(self.lr_image_filenames[index]))
        lr_image = Image.open(self.lr_image_filenames[index])
        w_lr, h_lr = lr_image.size
        lr_top = random.randint(0, h_lr - self.lr_crop_size)
        lr_left = random.randint(0, w_lr - self.lr_crop_size)
        lr_box = (lr_top, lr_left, lr_top+self.lr_crop_size, lr_left+self.lr_crop_size)
        lr_image = lr_image.crop(lr_box)

        hr_image = Image.open(self.image_filenames[index])
        hr_top = int(lr_top * self.upscale_factor)
        hr_left = int(lr_left * self.upscale_factor)
        hr_box = (hr_top, hr_left, hr_top+self.hr_crop_size, hr_left+self.hr_crop_size)
        hr_image = hr_image.crop(hr_box)
        

        hr_image = ToTensor()(hr_image)
        lr_image = ToTensor()(lr_image)

        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)

class PairedValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, lr_dataset_dir, upscale_factor):
        super(PairedValDatasetFromFolder, self).__init__()
        self.upscale_factor = upscale_factor
        self.image_filenames = [join(dataset_dir, x) for x in sorted(listdir(dataset_dir)) if is_image_file(x)]
        self.lr_image_filenames = [join(lr_dataset_dir, x) for x in sorted(listdir(lr_dataset_dir)) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        lr_image = Image.open(self.lr_image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        # lr_scale = Resize(crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=InterpolationMode.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = CenterCrop(crop_size //self.upscale_factor)(lr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)

class TrainDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor):
        super(TrainDatasetFromFolder, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        crop_size = calculate_valid_crop_size(crop_size, upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)

    def __getitem__(self, index):
        hr_image = self.hr_transform(Image.open(self.image_filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image

    def __len__(self):
        return len(self.image_filenames)


class ValDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(ValDatasetFromFolder, self).__init__()
        
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]

    def __getitem__(self, index):
        hr_image = Image.open(self.image_filenames[index])
        w, h = hr_image.size
        crop_size = calculate_valid_crop_size(min(w, h), self.upscale_factor)
        lr_scale = Resize(crop_size // self.upscale_factor, interpolation=InterpolationMode.BICUBIC)
        hr_scale = Resize(crop_size, interpolation=InterpolationMode.BICUBIC)
        hr_image = CenterCrop(crop_size)(hr_image)
        lr_image = lr_scale(hr_image)
        hr_restore_img = hr_scale(lr_image)
        return ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetFromFolder(Dataset):
    def __init__(self, dataset_dir, upscale_factor):
        super(TestDatasetFromFolder, self).__init__()
        self.lr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/data/'
        self.hr_path = dataset_dir + '/SRF_' + str(upscale_factor) + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):
        image_name = self.lr_filenames[index].split('/')[-1]
        lr_image = Image.open(self.lr_filenames[index])
        w, h = lr_image.size
        hr_image = Image.open(self.hr_filenames[index])
        hr_scale = Resize((self.upscale_factor * h, self.upscale_factor * w), interpolation=InterpolationMode.BICUBIC)
        hr_restore_img = hr_scale(lr_image)
        return image_name, ToTensor()(lr_image), ToTensor()(hr_restore_img), ToTensor()(hr_image)

    def __len__(self):
        return len(self.lr_filenames)