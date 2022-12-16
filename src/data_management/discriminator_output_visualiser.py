import torchvision
import matplotlib.pyplot as plt
import torch

from src.classic.utils import shuffle_dataset
from src.data_management.dataloaders.my_data_set_loader import MyDatasetLoader
from src.data_management.datasets.fashion_mnist_dataset import FashionMNISTDataset
from src.data_management.datasets.generated_fake_dataset import GeneratedFakeDataset
# from src.data_management.datasource import show_images_from_tensor
from src.gan.discriminator import Discriminator
from src.gan.generator import Generator
from math import ceil
device = torch.device("cuda:0")
BATCH_SIZE = 64

class DiscriminatorVisualiser:
    def __init__(self):
        self.n_row = 8

    def from_images(self, images, targets, predictions):
        columns = 10
        rows = images.size(0)//columns + 1
        fig = plt.figure(figsize=(ceil(columns*1.8), ceil(rows*2.5)))
        images = images.permute(0, 2, 3, 1)
        for i in range(images.size(0)):
            # plt.subplot(rows, columns, i+1)
            fig.add_subplot(rows, columns, i+1)
            self.from_image(images[i], targets[i], predictions[i])
        plt.show()


    def from_image(self, image, target, prediction):
        plt.imshow(image, cmap='gray', interpolation='nearest')
        plt.axis('off')
        plt.title(f"Target: {target.item()} \n Prediction: {round(prediction.item(),2)}")




def sample_discriminator(discriminator, train_loader):
    data_batch = next(iter(train_loader))
    images = data_batch[1][0].cpu()
    predictions = discriminator(data_batch[1][0].to(device)).cpu()
    targets = data_batch[1][1]
    return images, predictions, targets


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

    discriminator_visualiser = DiscriminatorVisualiser()
    # discriminator_visualiser.from_discriminator(discriminator, next(iter(train_loader)))
    images, predictions, targets = sample_discriminator(discriminator, train_loader)
    # show_images_from_tensor(images)
    discriminator_visualiser.from_images(images, targets, predictions)
    # discriminator_visualiser.load_data_for_visualization(images, predictions, targets)