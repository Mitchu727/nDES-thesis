import torch

from src.classic.utils import shuffle_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader


def create_merged_dataloader(real_dataset, fake_dataset, batch_size, device):
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