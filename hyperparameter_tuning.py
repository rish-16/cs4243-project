import torch, copy, random
import matplotlib.pyplot as plt
from model_training import *

class Tuner:
    def __init__(self, model_class, trainset, valset, param_ranges):
        self.model_class = model_class
        self.trainset = trainset
        self.valset = valset
        self.param_ranges = param_ranges
        self.best_param = {p: None for p, r in param_ranges.items()}
        self.log = {param: {p: None for p in param_range} for param, param_range in param_ranges.items()}
    def update_best_param(self, p, v):
        prev_best = self.best_param[p]
        if prev_best is None:
            self.best_param[p] = v
            return
        prev_perf = self.get_val_acc(self.log[p][prev_best])
        curr_perf = self.get_val_acc(self.log[p][v])
        if curr_perf > prev_perf:
            self.best_param[p] = v
    def init_param(self, p, v):
        current_param = self.best_param.copy()
        current_param[p] = v
        for p, v in current_param.items():
            if v is None:
                current_param[p] = self.param_ranges[p][0]
        return current_param
    def create_trainer(self, param):
        if self.model_class == MLP and isinstance(self.trainset, DoodleDataset):
            param['n_input'] = 64*64
        elif self.model_class == MLP and isinstance(self.trainset, RealDataset):
            param['n_input'] = 64*64*3
        elif self.model_class == CNN and isinstance(self.trainset, DoodleDataset):
            param['n_channels'] = 1
        elif self.model_class == CNN and isinstance(self.trainset, RealDataset):
            param['n_channels'] = 3
        return Trainer(self.model_class(**param),
                       self.trainset, self.valset, 5, 128)
    def log_param(self, param, p, hist):
        self.log[param][p] = hist
    def get_val_acc(self, hist):
        return hist['val_acc'][-1]
    def print_log(self, param):
        print(f"{param:>10}", end="")
        for p in self.param_ranges[param]:
            s = f"*{p}" if p == self.best_param[param] else f"{p}"
            print(f"{s:>10}", end="")
        print("\n{:>10}".format("val acc"), end="")
        for p in self.param_ranges[param]:
            print(f"{self.get_val_acc(self.log[param][p]):>10.3f}", end="")
        print('\n')
    def tune(self):
        for param, values in self.param_ranges.items():
            for p in values:
                curr_param = self.init_param(param, p)
                t = self.create_trainer(curr_param)
                hist = t.train()
                self.log_param(param, p, hist)
                self.update_best_param(param, p)
            self.print_log(param)