import torch

from src.data_management.datasets.generated_fake_dataset import get_noise_for_nn
from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
import matplotlib.pyplot as plt

device = torch.device("cuda:0")
BATCH_SIZE = 64

class GeneratorSample:
    def __init__(self, generated_images, discriminator_outputs):
        self.images = generated_images
        self.discriminator_outputs = discriminator_outputs
        self.calculated_metrics = {}

    def calculated_metrics_to_string(self):
        calculated_metrcis_string = ""
        for key in self.calculated_metrics:
            calculated_metrcis_string += f"{key}: {self.calculated_metrics[key]} \n"
        return calculated_metrcis_string

    def calculate_mean_output_metric(self):
        metric_id = 'Mean'
        self.calculated_metrics[metric_id] = torch.mean(self.discriminator_outputs)

    def calculate_min_output_metric(self):
        metric_id = 'Min'
        self.calculated_metrics[metric_id] = torch.min(self.discriminator_outputs)

    def calculate_max_output_metric(self):
        metric_id = 'Max'
        self.calculated_metrics[metric_id] = torch.max(self.discriminator_outputs)


class GeneratorVisualiser:
    def from_images(self, generator_sample, columns_number=4):
        images = generator_sample.images.permute(0, 2, 3, 1)
        if images.size(0) % columns_number == 0:
            rows_number = int(images.size(0) / columns_number)
            print("hejka hejka")
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
        plt.suptitle(generator_sample.calculated_metrics_to_string())
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.show()

    def put_image(self, image, axs):
        axs.imshow(image, cmap='gray', interpolation='nearest')
        axs.axis('off')


def sample_from_generator(generator, num_of_samples):
    generator(get_noise_for_nn(generator.get_latent_dim(), num_of_samples, generator.device)).detach().cpu()


if __name__ == '__main__':
    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(device)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(device)
    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    generator.load_state_dict(torch.load("../../pre-trained/generator"))

    generator_visualiser = GeneratorVisualiser()

    num_of_samples = 24
    generated_images = generator(get_noise_for_nn(generator.get_latent_dim(), num_of_samples, generator.device)).detach().cpu()
    discriminator_outputs = discriminator(generated_images.to(device))
    sample = GeneratorSample(generated_images, discriminator_outputs)
    sample.calculate_mean_output_metric()
    sample.calculate_min_output_metric()
    sample.calculate_max_output_metric()
    generator_visualiser.from_images(sample)

    # generator_visualiser.from_images()