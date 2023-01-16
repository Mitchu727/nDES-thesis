import torch
import torch.nn as nn
import wandb

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset
from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset, get_noise_for_nn
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.data_management.output.generator_output import GeneratorOutputManager, GeneratorSample
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.gan.utils import create_merged_dataloader

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 1000)
EPOCHS = int(POPULATION) * 50
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_experiment"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
BATCH_NUM = 50
VALIDATION_SIZE = 10000
STRATIFY = False


if __name__ == "__main__":
    seed_everything(SEED_OFFSET+20)

    generator_ndes_config = {
        'history': 3,
        'worst_fitness': 3,
        'Ft': 1,
        'ccum': 0.96,
        # 'cp': 0.1,
        'lower': -50.0,
        'upper': 50.0,
        'log_dir': "ndes_logs/",
        'tol': 1e-6,
        'budget': EPOCHS,
        'device': DEVICE
    }
    discriminator_ndes_config = {
        'history': 3,
        'worst_fitness': 3,
        'Ft': 1,
        'ccum': 0.96,
        # 'cp': 0.1,
        'lower': -50.0,
        'upper': 50.0,
        'log_dir': "ndes_logs/",
        'tol': 1e-6,
        'budget': EPOCHS,
        'device': DEVICE
    }
    wandb.init(project=MODEL_NAME, entity="mmatak", config={**discriminator_ndes_config})

    discriminator_criterion = nn.MSELoss()
    basic_generator_criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    generator.load_state_dict(torch.load("../../pre-trained/generator"))

    discriminator_output_manager = DiscriminatorOutputManager(discriminator_criterion)
    generator_output_manager = GeneratorOutputManager()

    fashionMNIST = FashionMNISTDataset()
    number_of_samples = fashionMNIST.get_number_of_samples()

    train_generated_images_number = 100000

    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        for _ in range(10):
            generated_fake_dataset = GeneratedFakeDataset(generator, number_of_samples)
            discriminator_data_loader = create_merged_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)

            discriminator_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_data_loader)
            discriminator_output_manager.visualise(discriminator_sample)

            discriminator_ndes_optim = BasenDESOptimizer(
                model=discriminator,
                criterion=discriminator_criterion,
                data_gen=discriminator_data_loader,
                ndes_config=discriminator_ndes_config,
                use_fitness_ewma=False,
                restarts=None,
                lr=0.00001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                lambda_=POPULATION,
                device=DEVICE,
            )
            discriminator = train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)

            discriminator_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_data_loader)
            discriminator_output_manager.visualise(discriminator_sample)

            generator_train_loader = ForGeneratorDataloader.for_generator(generator, train_generated_images_number, BATCH_NUM)
            generator_criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))

            generator_sample = GeneratorSample.sample_from_generator(generator, discriminator, 64)
            generator_output_manager.visualise(generator_sample)

            generator_ndes_optim = BasenDESOptimizer(
                model=generator,
                criterion=generator_criterion,
                data_gen=generator_train_loader,
                ndes_config=generator_ndes_config,
                use_fitness_ewma=False,
                restarts=None,
                lr=0.001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                lambda_=POPULATION,
                device=DEVICE,
            )
            generator = train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)

            generator_sample = GeneratorSample.sample_from_generator(generator, discriminator, 64)
            generator_output_manager.visualise(generator_sample)
    else:
        raise Exception("Not yet implemented")
    wandb.finish()