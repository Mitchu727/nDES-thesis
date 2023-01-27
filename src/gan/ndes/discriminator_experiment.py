import copy

import torch
import torch.nn as nn

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.gan.networks.generator import Generator
from src.gan.networks.discriminator import Discriminator
from src.gan.utils import create_merged_train_dataloader, create_merged_test_dataloader, \
    create_discriminator_visualisation_dataloader
from src.loggers.logger import Logger

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 10)
EPOCHS = int(POPULATION) * 10
NDES_TRAINING = True

# TODO
# zbiór testowy

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_discriminator"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
VALIDATION_SIZE = 10000
STRATIFY = False
PRE_TRAINED_DISCRIMINATOR = False
PRE_TRAINED_GENERATOR = False
FAKE_DATASET_SIZE = 60000


def evaluate_discriminator(discriminator, test_loader, info):
    evaluation_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, test_loader)
    discriminator_output_manager.calculate_metrics(evaluation_sample)
    logger.log_discriminator_sample(evaluation_sample, info)


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)
    logger = Logger("ndes_logs/discriminator", MODEL_NAME)
    logger.log_conf("DEVICE", DEVICE)
    logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    logger.log_conf("VALIDATION_SIZE", VALIDATION_SIZE)
    logger.log_conf("STRATIFY", STRATIFY)
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    logger.log_conf("FAKE_DATASET_SIZE", FAKE_DATASET_SIZE)
    logger.log_conf("POPULATION", POPULATION)
    logger.log_conf("EPOCHS", EPOCHS)

    ndes_config = {
        'history': 3,
        'worst_fitness': 3,
        'Ft': 1,
        'ccum': 0.96,
        # 'cp': 0.1,
        'lower': -50.0,
        'upper': 50.0,
        'tol': 1e-6,
        'budget': EPOCHS,
        'device': DEVICE,
        'lambda_': POPULATION,
    }

    criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    generated_fake_dataset = GeneratedFakeDataset(generator, FAKE_DATASET_SIZE, 10000)

    train_loader = create_merged_train_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)
    test_loader = create_merged_test_dataloader(fashionMNIST, generated_fake_dataset, 20000, DEVICE)

    visualisation_loader = create_discriminator_visualisation_dataloader(
        fashionMNIST.get_random_from_test(12),
        generated_fake_dataset.get_random_from_test(12)
    )

    original_discriminator = copy.deepcopy(discriminator)

    discriminator_output_manager = DiscriminatorOutputManager(criterion, logger)
    logger.start_training()
    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        discriminator_ndes_optim = BasenDESOptimizer(
            model=discriminator,
            criterion=criterion,
            data_gen=train_loader,
            logger=logger,
            use_fitness_ewma=False,
            restarts=None,
            lr=0.00001,
            secondary_mutation=SecondaryMutation.RandomNoise,
            **ndes_config
        )

        evaluate_discriminator(discriminator, test_loader, "begin")
        vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, visualisation_loader)
        discriminator_output_manager.visualise(vis_sample, "/discriminator_begin.png")

        train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)

        evaluate_discriminator(discriminator, test_loader, "end")
        vis_sample = DiscriminatorSample.from_discriminator_and_loader(discriminator, visualisation_loader)
        discriminator_output_manager.visualise(vis_sample, "/discriminator_end.png")
    else:
        raise Exception("Not yet implemented")
    logger.end_training()


