import torch
from torchvision import datasets, transforms
import torch.utils.data


class FashionMNISTDataset:
    def __init__(self):
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dataset = datasets.FashionMNIST(
            "../data",
            train=True,
            download=True,
            transform=transform,
        )
        self.test_dataset = datasets.FashionMNIST(
            "../data",
            train=False,
            transform=transforms.Compose(
                [transforms.ToTensor()]
            ),
        )

        for x, y in torch.utils.data.DataLoader(
            self.train_dataset, batch_size=len(self.train_dataset), shuffle=True
        ):
            self.train_data, self.train_targets = x, y
        for x, y in torch.utils.data.DataLoader(
            self.test_dataset, batch_size=len(self.train_dataset), shuffle=True
        ):
            self.test_data, self.test_targets = x, y

    def get_train_set_targets(self):
        return torch.ones_like(self.train_dataset.targets).float()

    def get_train_images(self):
        return self.train_data

    def get_test_set_targets(self):
        return torch.ones_like(self.test_dataset.targets).float()

    def get_test_images(self):
        return self.test_data

    def get_number_of_train_samples(self):
        return len(self.train_data)

    def get_number_of_test_samples(self):
        return len(self.test_data)

    def get_random_from_test(self, n_samples):
        perm = torch.randperm(self.test_data.size(0))
        idx = perm[:n_samples]
        return self.test_data[idx], torch.ones(len(idx))
