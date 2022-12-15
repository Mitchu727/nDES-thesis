import torchvision
import matplotlib.pyplot as plt


def show_images_from_tensor(images, n_row=8):
    grid = torchvision.utils.make_grid(images, nrow=n_row)
    grid = grid.permute(1, 2, 0)
    plt.imshow(grid)
    plt.axis('off')
    plt.show()
