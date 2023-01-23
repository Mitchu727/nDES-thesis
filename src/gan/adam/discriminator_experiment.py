import torch
import torch.optim as optim
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import TensorDataset
import numpy as np

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.gan.networks.discriminator import Discriminator
from src.gan.networks.generator import Generator
from src.gan.utils import create_merged_test_dataloader
from src.loggers.logger import Logger

DEVICE = torch.device("cuda:0")
PRE_TRAINED_DISCRIMINATOR = False
PRE_TRAINED_GENERATOR = False
MODEL_NAME = "gan_adam_discriminator"


if __name__ == "__main__":
    logger = Logger("adam_logs/", MODEL_NAME, "adam")
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    train_data_real = fashionMNIST.train_data

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real), fashionMNIST.get_number_of_test_samples())
    train_data_fake = generated_fake_dataset.train_dataset

    train_targets_real = fashionMNIST.get_train_set_targets()
    train_targets_fake = generated_fake_dataset.get_train_set_targets()
    criterion = nn.MSELoss()

    discriminator_output_manager = DiscriminatorOutputManager(criterion)

    merged_train_loader = data.DataLoader(
        TensorDataset(train_data_real, train_targets_real, train_data_fake, train_targets_fake), batch_size=256,
        shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    merged_test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 32, DEVICE)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.99)

    num_epochs = 10
    logger.start_training()
    sample_1 = DiscriminatorSample.from_discriminator_and_loader(discriminator, merged_test_loader)
    logger.log_discriminator_sample(sample_1, "begin")
    discriminator_output_manager.visualise(sample_1, logger.dir + "/discriminator_begin.png")
    for epoch in range(num_epochs):
        discriminator_real_acc = []
        discriminator_fake_acc = []
        discriminator_error_real = []
        discriminator_error_fake = []

        for i, batch in enumerate(merged_train_loader, 0):
            # Train with all-real batch
            discriminator_optimizer.zero_grad()
            images_real = batch[0].to(DEVICE)
            label_real = batch[1].float().to(DEVICE)
            # Forward pass real batch through D
            output = discriminator(images_real).view(-1)
            # Calculate loss on all-real batch
            error_discriminator_real = criterion(output, label_real)
            error_discriminator_real.backward()
            discriminator_error_real.append(error_discriminator_real.mean().item())
            discriminator_real_acc.append(output.mean().item())
            discriminator_optimizer.step()

            discriminator_optimizer.zero_grad()
            fake_offset = 2
            fake_images = batch[fake_offset + 0].to(DEVICE)
            label_fake = batch[fake_offset + 1].float().to(DEVICE)
            # Classify all fake batch with Discriminator
            output = discriminator(fake_images).view(-1)
            # Calculate D's loss on the all-fake batch
            error_discriminator_fake = criterion(output, label_fake)
            # Calculate the gradients for this batch, accumulated (summed) with previous gradients
            error_discriminator_fake.backward()
            discriminator_error_fake.append(error_discriminator_fake.mean().item())
            discriminator_fake_acc.append(output.mean().item())
            discriminator_optimizer.step()

        print(f"Epoch: {epoch}, discriminator real mean output: {np.mean(discriminator_real_acc):.3}")
        print(f"Epoch: {epoch}, discriminator real mean error: {np.mean(discriminator_error_real):.3}")
        print(f"Epoch: {epoch}, discriminator fake mean output: {np.mean(discriminator_fake_acc):.3}")
        print(f"Epoch: {epoch}, discriminator fake mean error: {np.mean(discriminator_error_fake):.3}")
    sample_2 = DiscriminatorSample.from_discriminator_and_loader(discriminator, merged_test_loader)
    logger.log_discriminator_sample(sample_2, "begin")
    discriminator_output_manager.visualise(sample_2, logger.dir + "/discriminator_end.png")
    logger.end_training()




