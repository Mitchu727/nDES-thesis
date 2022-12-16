import torch.utils.data
import torch.nn as nn
import wandb

from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn
from src.data_management.datasource import show_images_from_tensor
from src.classic.ndes import SecondaryMutation
from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader


POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 50)
EPOCHS = int(POPULATION) * 10
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_generator"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_SIZE = 1000
BATCH_NUM = 50
VALIDATION_SIZE = 10000
STRATIFY = False


if __name__ == "__main__":
    seed_everything(0)

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)
    generator.load_state_dict(torch.load("../../pre-trained/generator"))
    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    # generate images from generator
    ndes_config = {
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

    wandb.init(project=MODEL_NAME, entity="mmatak", config={**ndes_config})

    basic_criterion = nn.MSELoss()

    def discriminator_criterion(out, targets):
        return basic_criterion(discriminator(out), targets.to(DEVICE))
        # return -discriminator(out).mean()

    # criterion = lambda out, targets: -discriminator(out).unsqueeze(1).sum()/out.size()[0]
    criterion = discriminator_criterion

    train_generated_images_number = 100000
    train_data = get_noise_for_nn(generator.get_latent_dim(), train_generated_images_number, generator.device)
    train_targets = torch.ones(train_generated_images_number)

    # 2 wersje
    # wersja no. 1 - low effort -> wsadzić dyskryminator w kryterium, byle co w targets
    # wersja no. 2 - głębiej wejść w kod i porefactorować
    train_loader = MyDatasetLoader(train_data, train_targets, BATCH_NUM)

    generator_ndes_optim = BasenDESOptimizer(
        model=generator,
        criterion=criterion,
        data_gen=train_loader,
        ndes_config=ndes_config,
        use_fitness_ewma=False,
        restarts=None,
        lr=0.001,
        secondary_mutation=SecondaryMutation.RandomNoise,
        lambda_=POPULATION,
        device=DEVICE,
    )
    num_of_samples = 24
    generated_images = generator(get_noise_for_nn(generator.get_latent_dim(), num_of_samples, generator.device)).detach().cpu()
    print(discriminator(generated_images.cuda()).sum())  # funkcja kosztu -> maksymalizacja
    wandb.watch(generator)
    train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)
    generated_images = generator(get_noise_for_nn(generator.get_latent_dim(), num_of_samples, generator.device)).detach().cpu()
    print(discriminator(generated_images.cuda()))
    print(discriminator(generated_images.cuda()).sum())# funkcja kosztu -> maksymalizacja -> już nie
    print(discriminator(generated_images.cuda()).mean())# funkcja kosztu -> maksymalizacja
    show_images_from_tensor(generated_images)





