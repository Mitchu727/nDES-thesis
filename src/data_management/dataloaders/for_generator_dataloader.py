import torch

from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader
from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn


class ForGeneratorDataloader:
    @staticmethod
    def for_generator(generator, number_of_samples, batch_num):
        train_data = get_noise_for_nn(generator.get_latent_dim(), number_of_samples, generator.device)
        train_targets = torch.ones(number_of_samples)
        return MyDatasetLoader(train_data, train_targets, batch_num)