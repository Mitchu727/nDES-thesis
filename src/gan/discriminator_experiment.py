import torch
import torch.nn as nn

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.gan.utils import create_merged_train_dataloader
from src.loggers.logger import Logger

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 10)
EPOCHS = int(POPULATION) * 100
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_discriminator"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 64
BATCH_NUM = 50
VALIDATION_SIZE = 10000
STRATIFY = False
PRE_TRAINED_DISCRIMINATOR = True
PRE_TRAINED_GENERATOR = True
FAKE_DATASET_SIZE = 60000


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)
    logger = Logger("ndes_logs/", MODEL_NAME)
    logger.log_conf("DEVICE", DEVICE)
    logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    logger.log_conf("BATCH_SIZE", BATCH_SIZE)
    logger.log_conf("BATCH_NUM", BATCH_NUM)
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
        'device': DEVICE
    }

    criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    generated_fake_dataset = GeneratedFakeDataset(generator, FAKE_DATASET_SIZE)

    train_loader = create_merged_train_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)

    discriminator_output_manager = DiscriminatorOutputManager(criterion)

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
            ndes_config=ndes_config,
            logger=logger,
            use_fitness_ewma=False,
            restarts=None,
            lr=0.00001,
            secondary_mutation=SecondaryMutation.RandomNoise,
            lambda_=POPULATION,
            device=DEVICE,
        )
        sample_1 = DiscriminatorSample.from_discriminator_and_loader(discriminator, train_loader)
        logger.log_discriminator_sample(sample_1, "begin")
        discriminator_output_manager.visualise(sample_1, logger.dir + "/discriminator_begin.png")
        train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
        sample_2 = DiscriminatorSample.from_discriminator_and_loader(discriminator, train_loader)
        logger.log_discriminator_sample(sample_2, "end")
        discriminator_output_manager.visualise(sample_2, logger.dir + "/discriminator_end.png")
    else:
        raise Exception("Not yet implemented")
    logger.end_training()
