from torchvision import transforms
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    def __init__(self, images):
        self.images = images

        self.transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        return self.transforms(self.images[idx])
