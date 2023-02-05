import torch
from torchvision import datasets, transforms
from constants import norm_mean, norm_std, img_size

class Datasets:
    def __init__(self, data_dir):
        train_dir = data_dir + '/train'
        valid_dir = data_dir + '/valid'
        test_dir = data_dir + '/test'

        resize_transform = transforms.Resize(img_size)
        center_crop_transform = transforms.CenterCrop(img_size)
        normalize_transform = transforms.Normalize(norm_mean, norm_std)

        training_transform = transforms.Compose([resize_transform,
                                                  center_crop_transform,
                                                  transforms.RandomHorizontalFlip(),
                                                  transforms.RandomRotation(30),
                                                  transforms.ToTensor(),
                                                  normalize_transform])

        validation_transform = transforms.Compose([resize_transform,
                                                   center_crop_transform,
                                                   transforms.ToTensor(),
                                                   normalize_transform])
        
        testing_transform = transforms.Compose([resize_transform,
                                                   center_crop_transform,
                                                   transforms.ToTensor(),
                                                   normalize_transform])

        self.training_dataset = datasets.ImageFolder(train_dir, transform=training_transform)
        self.validation_dataset = datasets.ImageFolder(valid_dir, transform=validation_transform)
        self.testing_dataset = datasets.ImageFolder(test_dir, transform=testing_transform)

        self.training_loader = torch.utils.data.DataLoader(self.training_dataset, batch_size=32, shuffle=True)
        self.validation_loader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=32, shuffle=True)
        self.testing_loader = torch.utils.data.DataLoader(self.testing_dataset, batch_size=32, shuffle=True)
