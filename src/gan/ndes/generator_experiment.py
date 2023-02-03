import torch.utils.data
import torch.nn as nn

from src.data_management.dataloaders.for_generator_dataloader import ForGeneratorDataloader
from src.data_management.output.generator_output import GeneratorOutputManager, GeneratorSample
from src.gan.networks.generator import Generator
from src.gan.networks.discriminator import Discriminator
from src.classic.ndes import SecondaryMutation
from src.classic.ndes_optimizer import BasenDESOptimizer
from src.classic.utils import seed_everything, train_via_ndes_without_test_dataset
from src.loggers.logger import Logger

POPULATION_MULTIPLIER = 1
POPULATION = int(POPULATION_MULTIPLIER * 35000)
EPOCHS = int(POPULATION) * 100
NDES_TRAINING = True

DEVICE = torch.device("cuda:0")
BOOTSTRAP = False
MODEL_NAME = "gan_ndes_generator"
LOAD_WEIGHTS = False
SEED_OFFSET = 0
BATCH_NUM = 600
VALIDATION_SIZE = 10000
STRATIFY = False
PRE_TRAINED_DISCRIMINATOR = True
PRE_TRAINED_GENERATOR = False


def evaluate_generator(generator, discriminator, test_loader, info):
    evaluation_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, test_loader)
    generator_output_manager.calculate_metrics(evaluation_sample, info)
    logger.log_generator_sample(evaluation_sample, info)


if __name__ == "__main__":
    seed_everything(SEED_OFFSET)
    logger = Logger("ndes_logs/generator/", MODEL_NAME)
    logger.log_conf("DEVICE", DEVICE)
    logger.log_conf("SEED_OFFSET", SEED_OFFSET)
    logger.log_conf("BATCH_NUM", BATCH_NUM)
    logger.log_conf("VALIDATION_SIZE", VALIDATION_SIZE)
    logger.log_conf("STRATIFY", STRATIFY)
    logger.log_conf("PRE_TRAINED_DISCRIMINATOR", PRE_TRAINED_DISCRIMINATOR)
    logger.log_conf("PRE_TRAINED_GENERATOR", PRE_TRAINED_GENERATOR)
    logger.log_conf("POPULATION", POPULATION)
    logger.log_conf("EPOCHS", EPOCHS)

    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(DEVICE)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(DEVICE)

    if PRE_TRAINED_DISCRIMINATOR:
        discriminator.load_state_dict(torch.load("pre-trained/discriminator"))
    if PRE_TRAINED_GENERATOR:
        generator.load_state_dict(torch.load("pre-trained/generator"))

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
    }

    basic_generator_criterion = nn.MSELoss()

    generator_output_manager = GeneratorOutputManager(basic_generator_criterion, logger)

    # criterion = lambda out, targets: -discriminator(out).unsqueeze(1).sum()/out.size()[0]
    criterion = lambda out, targets: basic_generator_criterion(discriminator(out), targets.to(DEVICE))

    train_loader = ForGeneratorDataloader.for_generator(generator, 60000, BATCH_NUM)
    test_loader = ForGeneratorDataloader.for_generator(generator, 10000, 1)
    visualisation_loader = ForGeneratorDataloader.for_generator(generator, 24, 1)

    logger.start_training()
    generator_ndes_optim = BasenDESOptimizer(
        model=generator,
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
        **ndes_config
    )

    evaluate_generator(generator, discriminator, test_loader, "begin")
    vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, visualisation_loader)
    generator_output_manager.visualise(vis_sample, "/generator_begin")

    train_via_ndes_without_test_dataset(generator, generator_ndes_optim, DEVICE, MODEL_NAME)

    evaluate_generator(generator, discriminator, test_loader, "end")
    vis_sample = GeneratorSample.sample_from_generator_and_loader(generator, discriminator, visualisation_loader)
    generator_output_manager.visualise(vis_sample, "/generator_end")

    logger.end_training()





