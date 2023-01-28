import copy
import json
import os
from timeit import default_timer as timer
import pandas as pd
import torch
import wandb
from pathlib import Path


class Logger:
    def __init__(self, path, model_name="nDES", save_interval=50, directory_name="ndes"):
        self.model_name = model_name
        self._iteration_log = {}
        self._fitness_log = {}
        self._config_log = {}
        self._output_metrics_log = {}
        self._fitness_log = pd.DataFrame()
        self.save_interval = save_interval
        self.iter_logs_collector = pd.DataFrame()
        self._output_metrics_collector = pd.DataFrame()
        self.dir = os.path.join(path, f"{directory_name}_{timer()}")
        if os.path.exists(path):
            os.mkdir(self.dir)
        else:
            # os.mkdir(path)
            Path(self.dir).mkdir(parents=True)

    def start_training(self):
        wandb.init(project=self.model_name, entity="mmatak", config={**self._config_log})

    def update_config(self):
        wandb.config.update({**self._config_log}, allow_val_change=True)

    def end_training(self):
        # self.save_config()
        self.save_metrics()
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
            'outputs': generator_sample.discriminator_outputs
        }
        torch.save(struct_to_save, f"{self.dir}/generator_{description}.pt")

    def log_output_metrics(self, metrics_dict):
        self._output_metrics_log = copy.deepcopy(metrics_dict)
        wandb.log(metrics_dict)
        if self._output_metrics_collector.empty:
            self._output_metrics_collector = pd.DataFrame(columns=list(self._output_metrics_log.keys()))
        self._output_metrics_collector = pd.concat([self._output_metrics_collector, pd.DataFrame([self._output_metrics_log])],
                                             ignore_index=True)

    def log_conf(self, key, value):
        self._config_log[key] = value

    def log_conf_kwargs(self, kwargs):
        self._config_log = {**self._config_log, **kwargs}

    def save_to_files(self):
        self.iter_logs_collector.to_csv(f"{self.dir}/iteration_logs.csv")
        self._fitness_log.to_csv(f"{self.dir}/fitness_logs.csv")

    def save_config(self):
        with open(self.dir + '/congig.json', 'w') as file:
            json.dump(self._config_log, file)

    def save_metrics(self):
        self._output_metrics_collector.to_csv(f"{self.dir}/metrics_logs.csv")
        # with open(self.dir + '/metrics.json', 'w') as file:
        #     json_string = json.dumps(self._output_metrics_log)
        #     file.write(json_string)

    def print_iter_log(self):
        for key in self._iteration_log:
            print(f"{key}:{self._iteration_log[key]}")
        # print(f"======= Iteration: {self._iteration_log['iter']} =======")
        # print(f"Step size: {self._iteration_log['step']}")
        # print(f"Pc: {self._iteration_log['pc']}")
        # print(f"Best fitness: {self._iteration_log['best_fitness']}")
        # print(f"Mean fitness: {self._iteration_log['mean_fitness']}")
        # print(f"Cumulative function value: {self._iteration_log['fn_cum']}")
        # print(f"Best found: {self._iteration_log['best_found']}")
