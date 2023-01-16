import torch
import torch.nn as nn
import wandb

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.data_management.output.discriminator_output import DiscriminatorSample, DiscriminatorOutputManager
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.gan.utils import create_merged_dataloader

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 5000)
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


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)

    ndes_config = {
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
    wandb.init(project=MODEL_NAME, entity="mmatak", config={**ndes_config})

    criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    generator.load_state_dict(torch.load("../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    generated_fake_dataset = GeneratedFakeDataset(generator, 60000)

    train_loader = create_merged_dataloader(fashionMNIST, generated_fake_dataset, BATCH_SIZE, DEVICE)

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
            use_fitness_ewma=False,
            restarts=None,
            lr=0.00001,
            secondary_mutation=SecondaryMutation.RandomNoise,
            lambda_=POPULATION,
            device=DEVICE,
        )
        sample_1 = DiscriminatorSample.from_discriminator_and_loader(discriminator, train_loader)
        discriminator_output_manager.visualise(sample_1)
        train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
        sample_2 = DiscriminatorSample.from_discriminator_and_loader(discriminator, train_loader)
        discriminator_output_manager.visualise(sample_2)
    else:
        raise Exception("Not yet implemented")
    wandb.finish()
