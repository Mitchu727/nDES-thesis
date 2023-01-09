import torchvision
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.transforms

from src.classic.utils import shuffle_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader
from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
from src.gan.discriminator import Discriminator
from src.gan.generator import Generator

from math import ceil
device = torch.device("cuda:0")
BATCH_SIZE = 64


class DiscriminatorSample:
    def __init__(self, images, targets, predictions, criterion=None):
        self.images = images
        self.targets = targets
        self.predictions = predictions
        self.criterion = criterion
        self.calculated_metrics = {}

    def calculate_criterion_metric(self):
        metric_id = 'Criterion'
        if self.criterion is None:
            raise Exception("No criterion given, cannot calculate loss")
        self.calculated_metrics[metric_id] = self.criterion(self.targets, self.predictions)

    def calculate_mean_prediction_metric(self):
        metric_id = 'Mean'
        self.calculated_metrics[metric_id] = torch.mean(self.predictions)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def calculated_metrics_to_string(self):
        calculated_metrcis_string = ""
        for key in self.calculated_metrics:
            calculated_metrcis_string += f"{key}: {self.calculated_metrics[key]} \n"
        return calculated_metrcis_string

class DiscriminatorVisualiser:

    # def from_images(self, images, targets, predictions, columns=10, criterion = None):
    #     rows = images.size(0)//columns + 1
    #     fig = plt.figure(figsize=(ceil(columns*1.8), ceil(rows*2.5)))
    #     images = images.permute(0, 2, 3, 1)
    #     for i in range(images.size(0)):
    #         # plt.subplot(rows, columns, i+1)
    #         fig.add_subplot(rows, columns, i+1)
    #         self.from_image(images[i], targets[i], predictions[i])
    #     if criterion is not None:
    #         fig.suptitle(f"Loss: {self.calculate_characteristics(criterion, targets, predictions)}")
    #     plt.show()

    def from_images(self, discriminator_sample, columns_number=4, criterion=None):
        images = discriminator_sample.images.permute(0, 2, 3, 1)
        if images.size(0) % columns_number == 0:
            rows_number = int(images.size(0) / columns_number)
            print("hejka hejka")
        else:
            rows_number = images.size(0) // columns_number + 1
        fig, axs = plt.subplots(rows_number, columns_number, figsize=(3*columns_number, 3*rows_number))
        current_row = 0
        current_column = 0

        for i in range(images.size(0)):
            current_axs = axs[current_row, current_column]
            self.put_image(images[i], current_axs)
            current_axs.set_title(f"Prediction: {discriminator_sample.targets[i].item():.2f} \n Target: {discriminator_sample.predictions[i].item():.2f}")
            if current_column == columns_number-1:
                current_column = 0
                current_row += 1
            else:
                current_column += 1
        plt.suptitle(discriminator_sample.calculated_metrics_to_string())
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

    def put_image(self, image, axs):
        axs.imshow(image, cmap='gray', interpolation='nearest')
        axs.axis('off')


def sample_discriminator(discriminator, train_loader):
    data_batch = next(iter(train_loader))
    images = data_batch[1][0].cpu()
    predictions = discriminator(data_batch[1][0].to(device)).cpu()
    targets = data_batch[1][1].cpu()
    return DiscriminatorSample(images, predictions, targets)


def show_images_from_tensor(images, n_row=8):
    grid = torchvision.utils.make_grid(images, nrow=n_row)
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    discriminator = Discriminator(hidden_dim=40, input_dim=784).to(device)
    generator = Generator(latent_dim=32, hidden_dim=40, output_dim=784).to(device)
    discriminator.load_state_dict(torch.load("../../pre-trained/discriminator"))
    generator.load_state_dict(torch.load("../../pre-trained/generator"))

    fashionMNIST = FashionMNISTDataset()
    train_data_real = fashionMNIST.train_data
    train_targets_real = fashionMNIST.get_train_set_targets()

    generated_fake_dataset = GeneratedFakeDataset(generator, len(train_data_real))
    train_data_fake = generated_fake_dataset.train_dataset
    train_targets_fake = generated_fake_dataset.get_train_set_targets()

    train_data_merged = torch.cat([train_data_fake, train_data_real], 0)
    train_targets_merged = torch.cat(
        [train_targets_fake, train_targets_real], 0).unsqueeze(1)
    train_data_merged, train_targets_merged = shuffle_dataset(train_data_merged, train_targets_merged)
    train_loader = MyDatasetLoader(
        x_train=train_data_merged.to(device),
        y_train=train_targets_merged.to(device),
        batch_size=BATCH_SIZE
    )

    criterion = nn.MSELoss()

    discriminator_visualiser = DiscriminatorVisualiser()
    # discriminator_visualiser.from_discriminator(discriminator, next(iter(train_loader)))
    discriminator_sample = sample_discriminator(discriminator, train_loader)
    # show_images_from_tensor(images)
    discriminator_sample.set_criterion(criterion)
    discriminator_sample.calculate_criterion_metric()
    discriminator_sample.calculate_mean_prediction_metric()
    discriminator_visualiser.from_images(discriminator_sample, 4, criterion)
    # discriminator_visualiser.load_data_for_visualization(images, predictions, targets)