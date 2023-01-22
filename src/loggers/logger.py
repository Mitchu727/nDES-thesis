import os
from timeit import default_timer as timer
import pandas as pd
import torch
import wandb


class Logger:
    def __init__(self, path, model_name="nDES", save_interval=50, directory_name="ndes"):
        self.model_name = model_name
        self._iteration_log = {}
        self._fitness_log = {}
        self._config_log = {}
        self.save_interval = save_interval
        self.iter_logs_collector = pd.DataFrame()
        self._fitness_log = pd.DataFrame()
        self.dir = os.path.join(path, f"{directory_name}_{timer()}")
        if os.path.exists(path):
            os.mkdir(self.dir)
        else:
            os.mkdir(path)
            os.mkdir(self.dir)

    def start_training(self):
        wandb.init(project=self.model_name, entity="mmatak", config={**self._config_log})

    def end_training(self):
        self.save_to_files()
        wandb.finish()

    def log_iter(self, key, value):
        self._iteration_log[key] = value

    def end_iter(self):
        self.print_iter_log()
        wandb.log(self._iteration_log)
        if self.iter_logs_collector.empty:
            self.iter_logs_collector = pd.DataFrame(columns=list(self._iteration_log.keys()))

        self.iter_logs_collector = pd.concat([self.iter_logs_collector, pd.DataFrame([self._iteration_log])], ignore_index=True)

        if self._iteration_log["iter"] % self.save_interval == 0:
            self.save_to_files()

        self._iteration_log = {}

    def save_to_files(self):
        self.iter_logs_collector.to_csv(f"{self.dir}/iteration_logs.csv")
        self._fitness_log.to_csv(f"{self.dir}/fitness_logs.csv")

    def log_fitness(self, fitness):
        self._fitness_log = pd.concat([self._fitness_log, pd.DataFrame([fitness])], ignore_index=True)

    def log_discriminator_sample(self, discriminator_sample, description):
        struct_to_save = {
            'images': discriminator_sample.images,
            'targets': discriminator_sample.targets,
            'predictions': discriminator_sample.predictions
        }
        torch.save(struct_to_save, f"{self.dir}/discriminator_{description}.pt")

    def log_generator_sample(self, generator_sample, description):
        struct_to_save = {
            'images': generator_sample.images,
            'outputs': generator_sample.outputs
        }
        torch.save(struct_to_save, f"{self.dir}/generator_{description}.pt")

    def print_iter_log(self):
        print(f"======= Iteration: {self._iteration_log['iter']} =======")
        print(f"Step size: {self._iteration_log['step_size']}")
        print(f"Pc: {self._iteration_log['pc']}")
        print(f"Best fitness: {self._iteration_log['best fitness']}")
        print(f"Mean fitness: {self._iteration_log['mean fitness']}")
        print(f"Cumulative function value: {self._iteration_log['fn_cum']}")
        print(f"Best found: {self._iteration_log['best_found']}")

    def log_conf(self, key, value):
        self._config_log[key] = value

    def log_conf_kwargs(self, kwargs):
        self._config_log = {**self._config_log, **kwargs}
