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
from src.gan.utils import create_merged_train_dataloader, create_merged_test_dataloader, \
    create_discriminator_visualisation_dataloader
from src.loggers.logger import Logger

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 10000)
EPOCHS = int(POPULATION) * 5
DISCRIMINATOR_EPOCH_MULTIPLIER = 5
NDES_TRAINING = True
CYCLES = 25

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_experiment"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
BATCH_NUM = 600
STRATIFY = False
PRE_TRAINED_DISCRIMINATOR = False
PRE_TRAINED_GENERATOR = False
GENERATOR_TRAIN_IMAGES_NUMBER = 60000
DISCRIMINATOR_GENERATED_TEST_IMAGES_NUMBER = 10000

def evaluate_discriminator(discriminator, test_loader, info):
    evaluation_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, test_loader)
    discriminator_output_manager.calculate_metrics(evaluation_sample)
    discriminator_logger.log_discriminator_sample(evaluation_sample, info)


def evaluate_generator(generator, discriminator, test_loader, info):
    evaluation_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, test_loader)
    generator_output_manager.calculate_metrics(evaluation_sample, info)
    generator_logger.log_generator_sample(evaluation_sample, info)


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)
    discriminator_logger = Logger("ndes_logs/gan/discriminator", MODEL_NAME)
    generator_logger = Logger("ndes_logs/gan/generator", MODEL_NAME)

    discriminator_logger.log_conf("DEVICE", DEVICE)
    discriminator_logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    discriminator_logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    discriminator_logger.log_conf("BATCH_NUM", BATCH_NUM)
    discriminator_logger.log_conf("STRATIFY", STRATIFY)
    discriminator_logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    discriminator_logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    discriminator_logger.log_conf("POPULATION", POPULATION)
    discriminator_logger.log_conf("EPOCHS", EPOCHS * DISCRIMINATOR_EPOCH_MULTIPLIER)
    discriminator_logger.log_conf("DISCRIMINATOR_GENERATED_TEST_IMAGES_NUMBER", DISCRIMINATOR_GENERATED_TEST_IMAGES_NUMBER)

    generator_logger.log_conf("DEVICE", DEVICE)
    generator_logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    generator_logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    generator_logger.log_conf("BATCH_NUM", BATCH_NUM)
    generator_logger.log_conf("STRATIFY", STRATIFY)
    generator_logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    generator_logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    generator_logger.log_conf("POPULATION", POPULATION)
    generator_logger.log_conf("EPOCHS", EPOCHS)
    generator_logger.log_conf("TRAIN_GENERATED_IMAGES_NUMBER", GENERATOR_TRAIN_IMAGES_NUMBER)


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
        'budget': EPOCHS * DISCRIMINATOR_EPOCH_MULTIPLIER,
        'device': DEVICE,
        'lambda_': POPULATION,
    }

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
        'device': DEVICE,
        'lambda_': POPULATION,
    }
    discriminator_criterion = nn.MSELoss()
    basic_generator_criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("pre-trained/generator"))

    discriminator_output_manager = DiscriminatorOutputManager(discriminator_criterion, discriminator_logger)
    generator_output_manager = GeneratorOutputManager(basic_generator_criterion, generator_logger)

    fashionMNIST = FashionMNISTDataset()
    number_of_samples = fashionMNIST.get_number_of_train_samples()

    train_generated_images_number = 60000
    test_generated_images_number = 10000


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
            # =====================================
            # NEW SETS CREATION
            # =====================================
            generated_fake_dataset = GeneratedFakeDataset(generator, number_of_samples, DISCRIMINATOR_GENERATED_TEST_IMAGES_NUMBER)

            # discriminator sets

            discriminator_data_loader = create_merged_train_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)
            discriminator_test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 20000, DEVICE)
            discriminator_visualisation_loader = create_discriminator_visualisation_dataloader(
                fashionMNIST.get_random_from_test(12),
                generated_fake_dataset.get_random_from_test(12)
            )

            # generator sets

            generator_train_loader = ForGeneratorDataloader.for_generator(generator, GENERATOR_TRAIN_IMAGES_NUMBER , BATCH_NUM)
            generator_test_loader = ForGeneratorDataloader.for_generator(generator, 10000, 1)
            generator_visualisation_loader = ForGeneratorDataloader.for_generator(generator, 24, 1)

            # =====================================
            # DISCRIMINATOR TRAINING
            # =====================================
            evaluate_discriminator(discriminator, discriminator_test_loader, str(i))

            vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_visualisation_loader)
            discriminator_output_manager.visualise(vis_sample, f"/{i}_discriminator_begin.png")

            discriminator_ndes_optim = BasenDESOptimizer(
                model=discriminator,
                criterion=discriminator_criterion,
                data_gen=discriminator_data_loader,
                logger=discriminator_logger,
                use_fitness_ewma=False,
                restarts=None,
                population_initializer=XavierMVNPopulationInitializerV2,
                lr=0.00001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                **discriminator_ndes_config
            )
            discriminator = train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)

            evaluate_discriminator(discriminator, discriminator_test_loader, str(i))

            vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, discriminator_visualisation_loader)
            discriminator_output_manager.visualise(vis_sample, f"/{i}_discriminator_end.png")

            # =====================================
            # GENERATOR TRAINING
            # =====================================

            generator_criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))
            evaluate_generator(generator, discriminator, generator_test_loader, str(i))

            vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, generator_visualisation_loader)
            generator_output_manager.visualise(vis_sample, f"/{i}_generator_begin.png")

            generator_ndes_optim = BasenDESOptimizer(
                model=generator,
                criterion=generator_criterion,
                data_gen=generator_train_loader,
                logger=generator_logger,
                use_fitness_ewma=False,
                restarts=None,
                population_initializer=XavierMVNPopulationInitializerV2,
                lr=0.00001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                **generator_ndes_config
            )
            generator = train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)

            evaluate_generator(generator, discriminator, generator_test_loader, str(i))

            vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, generator_visualisation_loader)
            generator_output_manager.visualise(vis_sample, f"/{i}_generator_end.png")
    else:
        raise Exception("Not yet implemented")
    discriminator_logger.end_training()
    generator_logger.end_training()
