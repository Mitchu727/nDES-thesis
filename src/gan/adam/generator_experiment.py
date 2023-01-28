import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader
from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn
from src.data_management.output.generator_output import GeneratorSample, GeneratorOutputManager
from src.gan.networks.discriminator import Discriminator
from src.gan.networks.generator import Generator
from src.loggers.logger import Logger

def evaluate_generator(generator, discriminator, test_loader, info):
    evaluation_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, test_loader)
    generator_output_manager.calculate_metrics(evaluation_sample, info)
    logger.log_generator_sample(evaluation_sample, info)

DEVICE = torch.device("cuda:0")
PRE_TRAINED_DISCRIMINATOR = True
PRE_TRAINED_GENERATOR = False
MODEL_NAME = "gan_adam_generator"
BATCH_NUM = 600

if __name__ == "__main__":
    logger = Logger("adam_logs/generator", MODEL_NAME, 20, "adam")
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)

    basic_generator_criterion = nn.MSELoss()
    generator_output_manager = GeneratorOutputManager(basic_generator_criterion, logger)

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.99)

    train_loader = ForGeneratorDataloader.for_generator(generator, 60000, BATCH_NUM)
    test_loader = ForGeneratorDataloader.for_generator(generator, 10000, 1)
    visualisation_loader = ForGeneratorDataloader.for_generator(generator, 24, 1)

    num_epochs = 100
    sample_size = 10000
    latent_dim = 32
    criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))

    logger.start_training()
    evaluate_generator(generator, discriminator, test_loader, "begin")
    vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, visualisation_loader)
    generator_output_manager.visualise(vis_sample, "/generator_begin")

    for epoch in range(num_epochs):
        error = []
        for i in range(BATCH_NUM):
            generator_optimizer.zero_grad()
            data_batch = next(iter(train_loader))
            noise = data_batch[1][0].to(generator.device)
            label = data_batch[1][1].to(generator.device)


            # noise = get_noise_for_nn(latent_dim, sample_size, DEVICE)
            # Generate fake image batch with Generator
            fake_images = generator(noise)
            # label = torch.ones((sample_size,), dtype=torch.float, device=DEVICE)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            # Calculate G's loss based on this output


            # difference criterion
            error_generator = criterion(fake_images, label)
            error_generator.backward()
            error.append(error_generator.item())
            # print(f"Epoch: {epoch}")
            # print(f"Generator mean error {error_generator.item()}")
        logger.log_iter("iter", epoch)
        logger.log_iter("error", np.mean(error).item())
        logger.end_iter()
        generator_optimizer.step()

    evaluate_generator(generator, discriminator, test_loader, "end")
    vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, visualisation_loader)
    generator_output_manager.visualise(vis_sample, "/generator_end")

    logger.end_training()
