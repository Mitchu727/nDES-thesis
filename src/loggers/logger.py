
class Logger:
    def __init__(self, path):
        self.path = path
        self.iteration_log = {}
        self.fitness_log = {}

    def log_iter(self, key, value):
        self.iteration_log[key] = value

    def end_iter(self):
        self.iteration_log = {}

    def fitness_log(self):
        self.fitness_log = {}

    def raw_image_log(self):

