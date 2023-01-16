import torch.utils.data
import torch.nn as nn
import wandb

from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader
from src.data_management.output.generator_output import GeneratorOutputManager, GeneratorSample
from src.gan.generator import Generator
from src.gan.discriminator import Discriminator
from src.classic.ndes import SecondaryMutation
from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset


POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 1000)
EPOCHS = int(POPULATION) * 100
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
    generator_output_manager = GeneratorOutputManager()

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


    def discriminator_criterion(out, targets):
        basic_criterion = nn.MSELoss()
        return basic_criterion(discriminator(out), targets.to(DEVICE))

    # criterion = lambda out, targets: -discriminator(out).unsqueeze(1).sum()/out.size()[0]
    criterion = discriminator_criterion

    train_loader = ForGeneratorDataloader.for_generator(generator, BATCH_NUM, 60000)

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

    sample_1 = GeneratorSample.sample_from_generator(generator, discriminator, 64)
    generator_output_manager.visualise(sample_1)
    train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)
    sample_2 = GeneratorSample.sample_from_generator(generator, discriminator, 64)
    generator_output_manager.visualise(sample_2)





