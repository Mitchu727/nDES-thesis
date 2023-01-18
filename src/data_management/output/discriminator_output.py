import matplotlib.pyplot as plt
import torch


class DiscriminatorOutputManager:
    def __init__(self, criterion=None):
        self.metrics_manager = DiscriminatorMetricManager()
        self.visualiser = DiscriminatorVisualiser()
        self.criterion = criterion

    def visualise(self, discriminator_sample, path_to_save):
        self.calculate_metrics(discriminator_sample)
        self.visualiser.from_images_amd_metrics(
            discriminator_sample=discriminator_sample,
            metrics=self.metrics_manager.calculated_metrics,
            path_to_save=path_to_save
        )

    def set_criterion(self, criterion):
        self.criterion = criterion

    def calculate_metrics(self, discriminator_sample):
        self.metrics_manager.calculate_criterion_metric(discriminator_sample, self.criterion)
        self.metrics_manager.calculate_mean_prediction_metric(discriminator_sample)


class DiscriminatorMetricManager:
    def __init__(self):
        self.calculated_metrics = {}

    def calculate_criterion_metric(self, discriminator_sample, criterion):
        metric_id = 'Criterion'
        if criterion is None:
            raise Exception("No criterion given, cannot calculate loss")
        self.calculated_metrics[metric_id] = criterion(discriminator_sample.targets, discriminator_sample.predictions)

    def calculate_mean_prediction_metric(self, discriminator_sample):
        metric_id = 'Mean'
        self.calculated_metrics[metric_id] = torch.mean(discriminator_sample.predictions)

    def reset(self):
        self.calculated_metrics = {}

    @staticmethod
    def calculated_metrics_to_string(calculated_metrics):
        calculated_metrics_string = ""
        for key in calculated_metrics:
            calculated_metrics_string += f"{key}: {calculated_metrics[key]} \n"
        return calculated_metrics_string


class DiscriminatorSample:
    def __init__(self, images, targets, predictions):
        self.images = images
        self.targets = targets
        self.predictions = predictions

    @classmethod
    def from_discriminator_and_loader(cls, discriminator, loader):
        data_batch = next(iter(loader))
        images = data_batch[1][0].cpu()
        predictions = discriminator(data_batch[1][0].to(discriminator.device)).cpu()
        targets = data_batch[1][1].cpu()
        return cls(images, predictions, targets)


class DiscriminatorVisualiser:
    def from_images_amd_metrics(self, discriminator_sample, metrics, path_to_save, columns_number=4):
        images = discriminator_sample.images.permute(0, 2, 3, 1)
        if images.size(0) % columns_number == 0:
            rows_number = int(images.size(0) / columns_number)
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
        plt.suptitle(DiscriminatorMetricManager.calculated_metrics_to_string(metrics))
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.savefig(path_to_save)
        plt.show()

    @staticmethod
    def put_image(image, axs):
        axs.imshow(image, cmap='gray', interpolation='nearest')
        axs.axis('off')
