from __future__ import annotations

import os
import pathlib

import torch

from utils.logger import get_logger
from dataclasses import asdict

logger_inst = get_logger("main")

class EarlyStopSignal:
    """
    EarlyStopper signal giver counts how often
    """

    def __init__(self, patience=1, min_delta=0, relative_delta=False, num_models=1):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.relative_delta = relative_delta
        self.best_models = []


    def early_stop_signal(self, validation_loss):
        if self.relative_delta:
            current_delta = self.min_validation_loss * self.min_delta
        else:
            current_delta = self.min_delta

        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + current_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


    def reset(self):
        self.counter = 0

    def __call__(self, loss):
        return self.early_stop_signal(loss)

class EarlyStopOnPlateau:

    def __init__(self):
        self.previous_validation_loss = 100_000
        self.best_model = None
        self.steps_ago = 0
        self.current_step = 0

    def check(self, loss, model):
        # when loss is small, save model
        if self.previous_validation_loss >= loss:
            self.previous_validation_loss = loss
            self.best_model = model.state_dict().copy()
            self.steps_ago = 0
            return True
        return False

    def load_last_best_model(self, model_inst, device):
        model_inst, best_model = model_inst.to(device), self.best_model.to(device)
        return model_inst.load_state_dict(best_model)

    def __call__(self, loss, model):
        return self.check(loss, model)


class CheckPoint:
    def __init__(self, checkpoint_name, checkpoint_fold,  patience=0, delta=0, verbose=True):
        self.fold = checkpoint_fold
        self.name = checkpoint_name

        self.patience = patience # number of iteration before starting to look again
        self.delta = delta # minimum threshold that needs to be overcome
        self.verbose = verbose

        self.best_loss = None
        self.last_checkpoint = None
        self.no_improvement_count = 0

    @property
    def save_path(self):
        base_dir = os.environ["MODELS_DIR"]
        dst = (pathlib.Path(base_dir) / f"{self.name}_fold{self.fold}").with_suffix(".pt")
        return str(dst)

    def check_criteria(self, loss):
        # when loss is small, save model
        if self.no_improvement_count >= self.patience:
            if (self.best_loss is None) or (self.best_loss >= (loss - self.delta)):
                logger_inst.info(f"Checkpoint criteria trigger started at {loss:6E} - after {self.no_improvement_count} waiting epochs")
                self.no_improvement_count = 0
                self.best_loss = loss
                return True
        self.no_improvement_count +=1
        return False

    def create_checkpoint(self, model, optimizer, scheduler, current_iteration, full_config):
        checkpoint = {
            "epoch": current_iteration,
            "model_cls_module": model.__class__.__module__,
            "model_cls_name": model.__class__.__name__,
            "model_inst" : (None if model.is_parametrized else model), # when using a parametrization model_inst can't be saved, in this case return None
            "model_state_dict" : model.state_dict().copy(),
            "optimizer" : optimizer,
            "optimizer_state_dict": optimizer.state_dict().copy(),
            "lr_scheduler": scheduler,
            "iteration": current_iteration,
            "full_config": asdict(full_config),
        }

        self.last_checkpoint = checkpoint
        torch.save(self.last_checkpoint, self.save_path)
