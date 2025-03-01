import os
import time
from abc import abstractmethod

import torch


class BaseModel():
    def __init__(self, config, dataloader):
        """init model with basic input, which are from __init__(**kwargs) function in inherited class."""
        self.config = config
        self.phase = config['phase']
        self.device = config[self.phase]['device']
        self.batch_size = config[self.phase]['dataloader']["args"]['batch_size']
        self.epoch = config['train']['n_epoch']
        self.lr = config['train']['lr']
        self.is_dataset_paired = config['test']['dataset']['is_paired']
        self.dataloader = dataloader
        self.apply_post_processing = config['test']['apply_post_processing']
        self.model_path = config[self.phase]['model_path']
        self.model_name = config[self.phase]['model_name']
        self.output_images_path = config['test']['output_images_path']

    def train(self):
        since = time.time()        
        self.train_step()
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def test(self):
        self.test_step()

    @abstractmethod
    def train_step(self):
        raise NotImplementedError('You must specify how to train your model.')

    @abstractmethod
    def val_step(self):
        raise NotImplementedError('You must specify how to do validation on your model.')

    def save_model(self, model):
        """Saves the model's state dictionary."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        path = os.path.join(self.model_path, self.model_name)
        torch.save(model.state_dict(), path)