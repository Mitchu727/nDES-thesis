import torch
from torch.distributions import MultivariateNormal, Uniform
from abc import ABC, abstractmethod

from src.classic.utils import bounce_back_boundary_2d


class BasePopulationInitializer(ABC):
    def __init__(self, initial_value, xavier_coeffs, device, lambda_=None):
        self.initial_value = initial_value
        self.xavier_coeffs = xavier_coeffs.to(device)
        self.device = device
        if lambda_ is None:
            # TODO this is replicated wrt NDES class init
            self.lambda_ = self.initial_value.numel() * 4
        else:
            self.lambda_ = lambda_

    @abstractmethod
    def get_new_population(self, lower, upper):
        pass


class XavierMVNPopulationInitializer(BasePopulationInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        sd_1 = torch.eye(self.lambda_, device=self.device).cpu()
        mean_1 = (
            torch.zeros_like(self.initial_value)
            .unsqueeze(1)
            .repeat(1, self.lambda_)
            .cpu()
        )
        self.normal1 = MultivariateNormal(mean_1, sd_1)
        sd_2 = torch.eye(1, device=self.device).cpu()
        mean_2 = (
            torch.zeros_like(self.initial_value)
            .unsqueeze(1)
            # .repeat(1, self.lambda_)
            .cpu()
        )
        self.normal2 = MultivariateNormal(mean_2, sd_2)

    def get_new_population(self, lower, upper):
        population = self.normal1.sample().to(self.device)
        # population = self.normal2.sample((self.lambda_,)).squeeze().T.contiguous().to(self.device)
        population *= self.xavier_coeffs[:, None]
        self.initial_value = self.initial_value.cuda()
        population += self.initial_value[:, None]
        population[:, 0] = self.initial_value
        return bounce_back_boundary_2d(population, lower, upper)


class UniformPopulationInitializer(BasePopulationInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.uniform = Uniform(-self.xavier_coeffs.cpu(), self.xavier_coeffs.cpu())

    def get_new_population(self, lower, upper):
        return self.uniform.sample((self.lambda_,)).transpose(0, 1).to(self.device)


class StartFromUniformPopulationInitializer(BasePopulationInitializer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.args = args
        self.kwargs = kwargs
        self.uniform = UniformPopulationInitializer(*args, **kwargs)
        self.xavier_mvn = None

    def get_new_population(self, lower, upper):
        # first iteration - start from uniform
        if self.uniform is not None:
            population = self.uniform.get_new_population(lower, upper)
            del self.uniform
            self.uniform = None
            return population
        # second iteration
        elif self.xavier_mvn is None:
            self.xavier_mvn = XavierMVNPopulationInitializer(*self.args, **self.kwargs)
        # consecutive iterations
        return self.xavier_mvn.get_new_population(lower, upper)


class SimpleMultivariateNormalGenerator:
    def __init__(self, mean):
        self.mean = mean