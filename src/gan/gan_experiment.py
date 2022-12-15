import torch
import torch.nn as nn
import wandb

from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.ndes import SecondaryMutation
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset, train_via_ndes, shuffle_dataset
from src.classic.fashion_mnist_experiment import MyDatasetLoader
from src.data_management.datasource import show_images_from_tensor

from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset, get_noise_for_nn
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 100)
EPOCHS = int(POPULATION) * 2
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


def show_sample_predictions(discriminator, my_data_loader_batch):
    # TODO to można podzielić na funkcje
    show_images_from_tensor(my_data_loader_batch[1][0].cpu())
    predictions = discriminator(my_data_loader_batch[1][0].to(DEVICE)).cpu()
    print(f"Loss: {discriminator_criterion(predictions.cuda(), my_data_loader_batch[1][1].cuda())}")
    print(f"Predictions: {predictions}")
    print(f"Targets: {my_data_loader_batch[1][1]}")


if __name__ == "__main__":
    seed_everything(SEED_OFFSET+20)

    generator_ndes_config = {
        'history': 16,
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
        'history': 16,
        'worst_fitness': 3,
        'Ft': 1,
        'ccum': 0.96,
        # 'cp': 0.1,
        'lower': -50.0,
        'upper': 50.0,
        'log_dir': "ndes_logs/",
        'tol': 1e-6,
        'budget': EPOCHS*10,
        'device': DEVICE
    }
    wandb.init(project=MODEL_NAME, entity="mmatak", config={**discriminator_ndes_config})

    discriminator_criterion = nn.MSELoss()
    basic_generator_criterion = nn.MSELoss()

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    # discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    # generator.load_state_dict(torch.load("../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    train_data_real = fashionMNIST.train_data
    train_targets_real = fashionMNIST.get_train_set_targets()

    train_generated_images_number = 100000

    if LOAD_WEIGHTS:
        raise Exception("Not yet implemented")

    if NDES_TRAINING:
        if STRATIFY:
            raise Exception("Not yet implemented")
        if BOOTSTRAP:
            raise Exception("Not yet implemented")
        for _ in range(10):
            generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real))
            train_data_fake = generated_fake_dataset.train_dataset
            train_targets_fake = generated_fake_dataset.get_train_set_targets()

            train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
            train_targets_merged = torch.cat(
                [train_targets_fake, train_targets_real], 0).unsqueeze(1)
            train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
            train_loader = MyDatasetLoader(
                x_train=train_data_merged.to(DEVICE),
                y_train=train_targets_merged.to(DEVICE),
                batch_size=BATCH_SIZE
            )
            discriminator_ndes_optim = BasenDESOptimizer(
                model=discriminator,
                criterion=discriminator_criterion,
                data_gen=train_loader,
                ndes_config=discriminator_ndes_config,
                use_fitness_ewma=False,
                restarts=None,
                lr=0.00001,
                secondary_mutation=SecondaryMutation.RandomNoise,
                lambda_=POPULATION,
                device=DEVICE,
            )

            discriminator = train_via_ndes_without_test_dataset(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
            show_sample_predictions(discriminator, next(iter(train_loader)))
            generator_noise = get_noise_for_nn(generator.get_latent_dim(), train_generated_images_number, generator.device)
            generator_train_loader = MyDatasetLoader(generator_noise, torch.zeros(train_generated_images_number), BATCH_NUM)
            generator_criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))
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
            generated_images = generator(get_noise_for_nn(generator.get_latent_dim(), 16, generator.device)).detach().cpu()
            show_images_from_tensor(generated_images)
        # train_via_ndes(discriminator, discriminator_ndes_optim, DEVICE, MODEL_NAME)
        # print(discriminator(train_loader.get_sample_images_gpu()))
        # print(discriminator(train_loader.get_sample_images_gpu()))
    else:
        raise Exception("Not yet implemented")
    wandb.finish()