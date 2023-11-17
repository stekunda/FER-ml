import numpy as np
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToPILImage
from torch.utils.data import ConcatDataset
from skimage.feature import local_binary_pattern
from sklearn import svm
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

IMG_HEIGHT = 48
IMG_WIDTH = 48

# Path to the training data
TRAIN_DATA_PATH = os.path.join(os.getcwd(), '../FER with DL/data', 'train')

# Path to the test data
TEST_DATA_PATH = os.path.join(os.getcwd(), '../FER with DL/data', 'test')

# Define your transformations
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor()
])


class LimitedImageFolder(ImageFolder):
    def __init__(self, root, transform=None, limit=1000):
        super().__init__(root, transform=transform)
        self.limit = limit
        self.samples = self._limit_samples()

    def _limit_samples(self):
        class_to_samples = {}
        for s, c in super().samples:
            if c not in class_to_samples:
                class_to_samples[c] = []
            if len(class_to_samples[c]) < self.limit:
                class_to_samples[c].append((s, c))
        self.class_to_idx = {c: i for i,
                             c in enumerate(class_to_samples.keys())}
        limited_samples = [sample for samples in class_to_samples.values()
                           for sample in samples]
        return limited_samples


# Load the datasets
train_dataset = LimitedImageFolder(
    TRAIN_DATA_PATH, transform=transform, limit=1000)
test_dataset = LimitedImageFolder(
    TEST_DATA_PATH, transform=transform, limit=1000)

# Create the dataloader for validation set only, train data still needs to be augmented
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
