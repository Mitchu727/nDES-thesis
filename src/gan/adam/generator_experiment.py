import torch
import torch.optim as optim
import torch.nn as nn

from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader
from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn
from src.data_management.output.generator_output import GeneratorSample, GeneratorOutputManager
from src.gan.networks.discriminator import Discriminator
from src.gan.networks.generator import Generator
from src.loggers.logger import Logger

DEVICE = torch.device("cuda:0")
PRE_TRAINED_DISCRIMINATOR = True
PRE_TRAINED_GENERATOR = True
MODEL_NAME = "gan_adam_generator"

if __name__ == "__main__":
    logger = Logger("adam_logs/", MODEL_NAME, "adam")
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)

    generator_output_manager = GeneratorOutputManager()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=0.0001)
    generator_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=generator_optimizer, gamma=0.99)

    num_epochs = 1000
    sample_size = 10000
    latent_dim = 32
    basic_generator_criterion = nn.MSELoss()
    criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))

    sample_1 = GeneratorSample.sample_from_generator(generator, discriminator, 32)
    generator_output_manager.visualise(sample_1)
    logger.log_generator_sample(sample_1, "begin")

    logger.start_training()
    for epoch in range(num_epochs):

        # Generate batch of latent vectors
        noise = get_noise_for_nn(latent_dim, sample_size, DEVICE)
        # Generate fake image batch with Generator
        fake_images = generator(noise)
        generator_optimizer.zero_grad()
        label = torch.ones((sample_size,), dtype=torch.float, device=DEVICE)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        # Calculate G's loss based on this output

        # mean_discriminator_output = -output.mean()
        # mean_discriminator_output.backward()
        # difference criterion
        error_generator = criterion(fake_images, label)
        error_generator.backward()
        print(f"Epoch: {epoch}")
        print(f"Generator mean error {error_generator.item()}")
        generator_optimizer.step()
    logger.end_training()

    sample_2 = GeneratorSample.sample_from_generator(generator, discriminator, 32)
    generator_output_manager.visualise(sample_2)
    logger.log_generator_sample(sample_2, "begin")
