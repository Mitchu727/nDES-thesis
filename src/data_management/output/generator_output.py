import torch

from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn
import matplotlib.pyplot as plt


class GeneratorOutputManager:
    def __init__(self, criterion, logger):
        self.metrics_manager = GeneratorMetricsManager()
        self.visualiser = GeneratorVisualiser()
        self.criterion = criterion
        self.logger = logger

    def visualise(self, generator_sample, path_to_save):
        self.calculate_metrics(generator_sample, log=False)
        self.visualiser.from_images_and_metrics(
            generator_sample=generator_sample,
            metrics=self.metrics_manager.calculated_metrics,
            path_to_save=self.logger.dir + path_to_save,
            columns_number=3
        )

    def calculate_metrics(self, generator_sample, log=True):
        self.metrics_manager.calculate_mean_output_metric(generator_sample)
        self.metrics_manager.calculate_min_output_metric(generator_sample)
        self.metrics_manager.calculate_max_output_metric(generator_sample)
        self.metrics_manager.calculate_output_std_dev_metric(generator_sample)
        self.metrics_manager.calculate_criterion_metric(generator_sample, self.criterion)
        if log:
            self.logger.log_output_metrics(self.metrics_manager.calculated_metrics)


class GeneratorMetricsManager:
    def __init__(self):
        self.calculated_metrics = {}

    def calculate_criterion_metric(self, generator_sample, criterion):
        metric_id = 'Wartość funkcji straty generatora'
        self.calculated_metrics[metric_id] = criterion(
            generator_sample.discriminator_outputs,
            torch.ones(len(generator_sample.discriminator_outputs), 1).to(generator_sample.discriminator_outputs.device)
        ).item()

    def calculate_mean_output_metric(self, generator_sample):
        metric_id = 'Średnie wskazanie'
        self.calculated_metrics[metric_id] = torch.mean(generator_sample.discriminator_outputs).item()

    def calculate_min_output_metric(self, generator_sample):
        metric_id = 'Minimalne wskazanie'
        self.calculated_metrics[metric_id] = torch.min(generator_sample.discriminator_outputs).item()

    def calculate_max_output_metric(self, generator_sample):
        metric_id = 'Maksymalne wskazanie'
        self.calculated_metrics[metric_id] = torch.max(generator_sample.discriminator_outputs).item()

    def calculate_output_std_dev_metric(self, generator_sample):
        metric_id = 'Odchylenie standardowe'
        self.calculated_metrics[metric_id] = torch.std(generator_sample.discriminator_outputs).item()

    def reset(self):
        self.calculated_metrics = {}

    @staticmethod
    def calculated_metrics_to_string(calculated_metrics):
        calculated_metrics_string = ""
        for key in calculated_metrics:
            calculated_metrics_string += f"{key}: {calculated_metrics[key]} \n"
        return calculated_metrics_string


class GeneratorSample:
    def __init__(self, generated_images, discriminator_outputs):
        self.images = generated_images
        self.discriminator_outputs = discriminator_outputs

    @classmethod
    def sample_from_generator(cls, generator, discriminator, num_of_samples):
        generated_images = generator(get_noise_for_nn(generator.get_latent_dim(), num_of_samples, generator.device)).detach().cpu()
        discriminator_outputs = discriminator(generated_images.to(discriminator.device))
        return GeneratorSample(generated_images, discriminator_outputs)

    @classmethod
    def sample_from_generator_and_loader(cls, generator, discriminator, loader):
        data_batch = next(iter(loader))
        generated_images = generator(data_batch[1][0].to(generator.device)).detach().cpu()
        discriminator_outputs = discriminator(generated_images.to(discriminator.device))
        return GeneratorSample(generated_images, discriminator_outputs)


class GeneratorVisualiser:
    def from_images_and_metrics(self, generator_sample, metrics, path_to_save, columns_number=4):
        images = generator_sample.images.permute(0, 2, 3, 1)
        if images.size(0) % columns_number == 0:
            rows_number = int(images.size(0) / columns_number)
        else:
            rows_number = images.size(0) // columns_number + 1
        fig, axs = plt.subplots(rows_number, columns_number, figsize=(3 * columns_number, 3 * rows_number))
        current_row = 0
        current_column = 0

        for i in range(images.size(0)):
            current_axs = axs[current_row, current_column]
            self.put_image(images[i], current_axs)
            current_axs.set_title(f"Discriminator output: {generator_sample.discriminator_outputs[i].item():.04f}")
            if current_column == columns_number - 1:
                current_column = 0
                current_row += 1
            else:
                current_column += 1
        plt.suptitle(GeneratorMetricsManager.calculated_metrics_to_string(metrics))
        plt.tight_layout()
        plt.subplots_adjust(top=0.70)
        plt.savefig(path_to_save)
        plt.show()

    @staticmethod
    def put_image(image, axs):
        axs.imshow(image, cmap='gray', interpolation='nearest')
        axs.axis('off')