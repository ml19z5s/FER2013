from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from sklearn.model_selection import StratifiedShuffleSplit

from torchvision import transforms
from PIL import ImageOps, ImageEnhance
import numpy as np


class RandomGrayscale(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = ImageOps.grayscale(img)
            img = ImageOps.colorize(img, "black", "white")
        return img


class RandomBrightnessContrastSaturation(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, img):
        img = ImageEnhance.Brightness(img).enhance(np.random.uniform(1 - self.brightness, 1 + self.brightness))
        img = ImageEnhance.Contrast(img).enhance(np.random.uniform(1 - self.contrast, 1 + self.contrast))
        img = ImageEnhance.Color(img).enhance(np.random.uniform(1 - self.saturation, 1 + self.saturation))
        return img


class CustomTransformDataset(Dataset):
    def __init__(self, dataset, transform=None, indices=None):
        self.dataset = dataset
        self.transform = transform
        self.indices = indices

    def __getitem__(self, index):
        if self.indices is not None:
            index = self.indices[index]

        sample, label = self.dataset[index]

        if self.transform:
            sample = self.transform(sample)

        return sample, label

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.dataset)


def get_mean_image_size(dataset_dir):
    dataset = datasets.ImageFolder(dataset_dir)
    sizes = []
    for sample, _ in dataset:
        sizes.append(sample.size)

    return np.mean(sizes, axis=0)


def load_datasets(train_dir, test_dir, args, batch_size, num_workers=4, test_only=False):
    size = get_mean_image_size(train_dir)
    print('Mean image size:', size)
    args.image_size = int(max(size))
    # args.image_size = 224
    image_size = args.image_size
    print('Loading datasets...')

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ]),
        'test': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # transforms.Normalize(0.5, 0.5)
        ]),
    }

    # CK+
    # split the data into train and test sets
    # train_dir: datasets/CK+/train
    # -> datasets/CK+
    # all_data = datasets.ImageFolder(os.path.dirname(train_dir))
    #
    # sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    # train_idx, test_idx = next(iter(sss.split(all_data, all_data.targets)))
    # train_idx_idx, val_idx = next(iter(sss.split(train_idx, np.array(all_data.targets)[train_idx])))
    # train_idx = train_idx[train_idx_idx]
    #
    # train_dataset = CustomTransformDataset(all_data, transform=data_transforms['train'], indices=train_idx)
    # val_dataset = CustomTransformDataset(all_data, transform=data_transforms['val'], indices=val_idx)
    # test_dataset = CustomTransformDataset(all_data, indices=test_idx)

    # Stratified sampling for train/val split
    full_train_dataset = datasets.ImageFolder(train_dir)
    test_dataset = datasets.ImageFolder(test_dir, transform=data_transforms['test'])
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_idx, val_idx = next(iter(sss.split(full_train_dataset, full_train_dataset.targets)))
    # Truncated data is used for testing
    if test_only:
        train_idx = train_idx[:1024]
        val_idx = val_idx[:256]
        test_dataset = CustomTransformDataset(test_dataset, indices=range(256))

    train_dataset = CustomTransformDataset(full_train_dataset, transform=data_transforms['train'], indices=train_idx)
    val_dataset = CustomTransformDataset(full_train_dataset, transform=data_transforms['val'], indices=val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    print('Done!')
    args.num_classes = len(set(full_train_dataset.targets))
    return train_loader, val_loader, test_loader
