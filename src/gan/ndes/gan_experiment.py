import torch
import torch.nn as nn
import wandb

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.population_initializers import XavierMVNPopulationInitializerV2
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset
from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.data_management.output.generator_output import GeneratorOutputManager, GeneratorSample
from src.gan.networks.generator import Generator
from src.gan.networks.discriminator import Discriminator
from src.gan.utils import create_merged_train_dataloader, create_merged_test_dataloader
from src.loggers.logger import Logger

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 5000)
EPOCHS = int(POPULATION) * 5
DISCRIMINATOR_MULTIPLIER = 10
NDES_TRAINING = True
CYCLES = 50

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_experiment"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
BATCH_NUM = 50
VALIDATION_SIZE = 10000
STRATIFY = False
PRE_TRAINED_DISCRIMINATOR = False
PRE_TRAINED_GENERATOR = False


if __name__ == "__main__":
    seed_everything(SEED_OFFSET+20)
    discriminator_logger = Logger("ndes_logs/", MODEL_NAME)
    generator_logger = Logger("ndes_logs/", MODEL_NAME)

    discriminator_logger.log_conf("DEVICE", DEVICE)
    discriminator_logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    discriminator_logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    discriminator_logger.log_conf("BATCH_NUM", BATCH_NUM)
    discriminator_logger.log_conf("VALIDATION_SIZE", VALIDATION_SIZE)
    discriminator_logger.log_conf("STRATIFY", STRATIFY)
    discriminator_logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    discriminator_logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    discriminator_logger.log_conf("POPULATION", POPULATION)
    discriminator_logger.log_conf("EPOCHS", EPOCHS*DISCRIMINATOR_MULTIPLIER)

    generator_logger.log_conf("DEVICE", DEVICE)
    generator_logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    generator_logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    generator_logger.log_conf("BATCH_NUM", BATCH_NUM)
    generator_logger.log_conf("VALIDATION_SIZE", VALIDATION_SIZE)
    generator_logger.log_conf("STRATIFY", STRATIFY)
    generator_logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    generator_logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    generator_logger.log_conf("POPULATION", POPULATION)
    generator_logger.log_conf("EPOCHS", EPOCHS)

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
        'budget': EPOCHS * DISCRIMINATOR_MULTIPLIER,
        'device': DEVICE
    }
    wandb.init(project=MODEL_NAME, entity="mmatak", config={**discriminator_ndes_config})

    discriminator_criterion = nn.MSELoss()
    basic_generator_criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    discriminator_output_manager = DiscriminatorOutputManager(discriminator_criterion)
    generator_output_manager = GeneratorOutputManager()

    fashionMNIST = FashionMNISTDataset()
    number_of_samples = fashionMNIST.get_number_of_train_samples()

    train_generated_images_number = 100000

    discriminator_logger.start_training()
    generator_logger.start_training()
    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        for i in range(CYCLES):
            generated_fake_dataset = GeneratedFakeDataset(generator, number_of_samples, 10000)
            discriminator_data_loader = create_merged_train_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)
            discriminator_merged_test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 24, DEVICE)

            discriminator_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_merged_test_loader)
            discriminator_output_manager.visualise(discriminator_sample, f"ndes_logs/{i}_discriminator_begin")

            discriminator_ndes_optim = BasenDESOptimizer(
                model=discriminator,
                criterion=discriminator_criterion,
                data_gen=discriminator_data_loader,
                logger=discriminator_logger,
                ndes_config=discriminator_ndes_config,
                use_fitness_ewma=False,
                restarts=None,
                population_initializer=XavierMVNPopulationInitializerV2,
                lr=0.00001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                lambda_=POPULATION,
                device=DEVICE,
            )
            discriminator = train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)

            discriminator_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_merged_test_loader)
            discriminator_output_manager.visualise(discriminator_sample, f"ndes_logs/{i}_discriminator_begin")

            generator_train_loader = ForGeneratorDataloader.for_generator(generator, train_generated_images_number, BATCH_NUM)
            generator_criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))

            generator_sample = GeneratorSample.sample_from_generator(generator, discriminator, 24)
            generator_output_manager.visualise(generator_sample, f"ndes_logs/{i}_generator_begin")

            generator_ndes_optim = BasenDESOptimizer(
                model=generator,
                criterion=generator_criterion,
                data_gen=generator_train_loader,
                logger=generator_logger,
                ndes_config=generator_ndes_config,
                use_fitness_ewma=False,
                restarts=None,
                population_initializer=XavierMVNPopulationInitializerV2,
                lr=0.001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                lambda_=POPULATION,
                device=DEVICE,
            )
            generator = train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)

            generator_sample = GeneratorSample.sample_from_generator(generator, discriminator, 24)
            generator_output_manager.visualise(generator_sample, f"ndes_logs/{i}_generator_end")
    else:
        raise Exception("Not yet implemented")
    discriminator_logger.end_training()
    generator_logger.start_training()
