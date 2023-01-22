import torch
import torch.nn as nn

from src.classic.utils import shuffle_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader
from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorOutputManager, DiscriminatorSample
from src.data_management.output.generator_output import GeneratorSample, GeneratorOutputManager

from src.gan.networks.discriminator import Discriminator
from src.gan.networks.generator import Generator

if __name__ == '__main__':
    device = torch.device("cuda:0")
    BATCH_SIZE = 64

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(device)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(device)
    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    generator.load_state_dict(torch.load("../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    train_data_real = fashionMNIST.train_data
    train_targets_real = fashionMNIST.get_train_set_targets()

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real))
    train_data_fake = generated_fake_dataset.train_dataset
    train_targets_fake = generated_fake_dataset.get_train_set_targets()

    train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
    train_targets_merged = torch.cat(
        [train_targets_fake, train_targets_real], 0).unsqueeze(1)
    train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
    train_loader = MyDatasetLoader(
        x_train=train_data_merged.to(device),
        y_train=train_targets_merged.to(device),
        batch_size=BATCH_SIZE
    )

    criterion = nn.MSELoss()

    discriminator_output_manager = DiscriminatorOutputManager(criterion)
    generator_output_manager = GeneratorOutputManager()

    disc_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, train_loader)
    discriminator_output_manager.visualise(disc_sample)
    gen_sample = GeneratorSample.sample_from_generator(generator, discriminator, 64)
    generator_output_manager.visualise(gen_sample)

