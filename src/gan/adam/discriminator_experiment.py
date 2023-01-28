import copy

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
from src.gan.utils import create_merged_test_dataloader, create_discriminator_visualisation_dataloader
from src.loggers.logger import Logger

DEVICE = torch.device("cuda:0")
PRE_TRAINED_DISCRIMINATOR = False
PRE_TRAINED_GENERATOR = False
MODEL_NAME = "gan_adam_discriminator"
FAKE_DATASET_SIZE = 60000


def evaluate_discriminator(discriminator, test_loader, info):
    evaluation_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, test_loader)
    discriminator_output_manager.calculate_metrics(evaluation_sample)
    logger.log_discriminator_sample(evaluation_sample, info)


if __name__ == "__main__":
    logger = Logger("adam_logs/", MODEL_NAME, 20, "adam")
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    logger.log_conf("FAKE_DATASET_SIZE", FAKE_DATASET_SIZE)

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    criterion = nn.MSELoss()

    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    generated_fake_dataset = GeneratedFakeDataset(generator, FAKE_DATASET_SIZE, 10000)


    train_data_real = fashionMNIST.train_data
    train_data_fake = generated_fake_dataset.train_dataset

    train_targets_real = fashionMNIST.get_train_set_targets()
    train_targets_fake = generated_fake_dataset.get_train_set_targets()

    test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 20000, DEVICE)

    visualisation_loader = create_discriminator_visualisation_dataloader(
        fashionMNIST.get_random_from_test(12),
        generated_fake_dataset.get_random_from_test(12)
    )

    discriminator_output_manager = DiscriminatorOutputManager(criterion, logger)

    merged_train_loader = data.DataLoader(
        TensorDataset(train_data_real, train_targets_real, train_data_fake, train_targets_fake), batch_size=256,
        shuffle=True, drop_last=True, pin_memory=True, num_workers=4)
    merged_test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 32, DEVICE)

    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0001)
    discriminator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=discriminator_optimizer, gamma=0.99)


    logger.start_training()

    evaluate_discriminator(discriminator, test_loader, "begin")
    vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, visualisation_loader)
    discriminator_output_manager.visualise(vis_sample, "/discriminator_end.png")

    num_epochs=100
    for epoch in range(num_epochs):
        discriminator_real_acc = []
        discriminator_fake_acc = []
        discriminator_error_real = []
        discriminator_error_fake = []
        error = []
        discriminator_original = copy.deepcopy(discriminator)
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
        # for i, batch in enumerate(merged_train_loader, 0):
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

            # error = error_discriminator_fake + error_discriminator_real
            # error.append(error_discriminator_fake.mean().item() + error_discriminator_real.mean().item())

        logger.log_iter("iter", epoch)
        logger.log_iter("error", np.mean(discriminator_error_real + discriminator_error_fake))
        logger.log_iter("discriminator real mean error", np.mean(discriminator_error_real))
        logger.log_iter("discriminator fake mean error", np.mean(discriminator_error_fake))
        logger.end_iter()
        # logger.log_iter("error", np.mean(error)

    evaluate_discriminator(discriminator, test_loader, "end")
    vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, visualisation_loader)
    discriminator_output_manager.visualise(vis_sample, "/discriminator_end.png")
    logger.end_training()




