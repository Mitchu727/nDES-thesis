import torch

from src.classic.utils import shuffle_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader


def create_merged_train_dataloader(real_dataset, fake_dataset, batch_size, device):
    train_data_real = real_dataset.train_data
    train_targets_real = real_dataset.get_train_set_targets()
    train_data_fake = fake_dataset.train_dataset
    train_targets_fake = fake_dataset.get_train_set_targets()
    train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
    train_targets_merged = torch.cat([train_targets_fake, train_targets_real], 0).unsqueeze(1)
    train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
    train_loader = MyDatasetLoader(
        x_train=train_data_merged.to(device),
        y_train=train_targets_merged.to(device),
        batch_size=batch_size
    )
    return train_loader


def create_merged_test_dataloader(real_dataset, fake_dataset, batch_size, device):
    test_data_real = real_dataset.test_data
    test_targets_real = real_dataset.get_test_set_targets()
    test_data_fake = fake_dataset.test_dataset
    test_targets_fake = fake_dataset.get_test_set_targets()
    test_data_merged = torch.cat([test_data_fake, test_data_real], 0)
    test_targets_merged = torch.cat([test_targets_fake, test_targets_real], 0).unsqueeze(1)
    test_data_merged, test_targets_merged = shuffle_dataset(test_data_merged, test_targets_merged)
    test_loader = MyDatasetLoader(
        x_train=test_data_merged.to(device),
        y_train=test_targets_merged.to(device),
        batch_size=batch_size
    )
    return test_loader


def create_discriminator_visualisation_dataloader(real_part, fake_part):
    visualisation_data = torch.cat([real_part[0], fake_part[0]], 0)
    visualisation_targets = torch.cat([real_part[1], fake_part[1]], 0)
    loader = MyDatasetLoader(
        x_train=visualisation_data,
        y_train=torch.unsqueeze(visualisation_targets, 1),
        batch_size=len(visualisation_data)
    )
    return loader


def train_discriminator_adam(model, criterion, train_dataset):
    pass