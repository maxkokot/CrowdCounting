import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
import pytorch_lightning as pl


class CustomDataset(Dataset):

    def __init__(self, annotations, root_dir, transform=None):

        self.annotations = annotations
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_info = self.annotations.iloc[idx]
        img_name = img_info['image_name']
        counts = img_info['count']

        image = Image.open(img_name).convert(mode='RGB')
        image = self.transform(image)
        counts = torch.tensor(counts, dtype=torch.float32)
        sample = (image, counts)
        return sample


class DataModule(pl.LightningDataModule):

    def __init__(self, trainval_annotations, test_annotations,
                 root_dir, train_transform=None, val_transform=None,
                 test_transform=None, trainval_size=0.8,
                 batch_size=32, random_state=42):
        super().__init__()

        self.trainval_annotations = trainval_annotations
        self.test_annotations = test_annotations
        self.root_dir = root_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.random_state = random_state
        self.batch_size = batch_size
        self.trainval_size = trainval_size

    def prepare_data(self):
        pass

    def setup(self, stage: str):

        if stage == "fit" or stage is None:
            train_set_full = CustomDataset(self.trainval_annotations,
                                           self.root_dir,
                                           self.train_transform)
            generator = torch.Generator().manual_seed(self.random_state)
            lengths = [round(len(train_set_full) * self.trainval_size),
                       len(train_set_full) -
                       round(len(train_set_full) * self.trainval_size)]
            self.train, self.validate = random_split(train_set_full,
                                                     lengths,
                                                     generator=generator)

            self.validate.transform = self.val_transform
        if stage == "test" or stage is None:
            self.test = CustomDataset(self.test_annotations,
                                      self.root_dir,
                                      self.test_transform)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validate, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


def prepare_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform


def prepare_augmented_transforms():

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        transforms.RandomHorizontalFlip()
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform


