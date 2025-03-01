import json
import importlib
from functools import partial
from types import FunctionType
from collections import OrderedDict

from torch.utils.data import DataLoader


class NoneDict(dict):
    def __missing__(self, key):
        return None


def dict_to_nonedict(config):
    """ Convert to NoneDict, which return None for missing key. """
    if isinstance(config, dict):
        new_config = dict()
        for key, sub_config in config.items():
            new_config[key] = dict_to_nonedict(sub_config)
        return NoneDict(**new_config)
    elif isinstance(config, list):
        return [dict_to_nonedict(sub_config) for sub_config in config]
    else:
        return config


def parse(args):
    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str, object_pairs_hook=OrderedDict)

    # replace the config context using args
    config['phase'] = args.phase

    return dict_to_nonedict(config)


def init_obj(obj_config, *args, default_file_name='default file', given_module=None, init_type='Network', **modify_kwargs):
    """ Takes a configuration dictionary `obj_config` and additional arguments to create and initialize an object. """
    name = obj_config['name']
    # name can be list, indicates the file and class name of function
    if isinstance(name, list):
        file_name, class_name = name[0], name[1]
    else:
        file_name, class_name = default_file_name, name

    try:
        if given_module is not None:
            module = given_module
        else:
            module = importlib.import_module(file_name)

        attr = getattr(module, class_name)
        kwargs = obj_config.get('args', {})
        kwargs.update(modify_kwargs)

        # import class or function with args
        if isinstance(attr, type):
            obj = attr(*args, **kwargs)
            obj.__name__ = obj.__class__.__name__
        elif isinstance(attr, FunctionType):
            obj = partial(attr, *args, **kwargs)
            obj.__name__ = attr.__name__

    except:
        raise NotImplementedError(
            f"{init_type} [{class_name}() from {file_name}] not recognized.")

    return obj


def create_model(**cfg_model):
    """ Creates and initializes a model based on the provided configuration. """
    config = cfg_model['config']

    model_config = config['model']['which_model']
    model_config['args'].update(cfg_model)
    model = init_obj(
        model_config, default_file_name='models.model', init_type='Model')

    return model


def define_network(network_config):
    """ Creates and initializes a network based on the provided configuration. """
    return init_obj(network_config, default_file_name='models.network', init_type='Network')


def define_dataset(dataset_config):
    """ Creates and initializes a dataset based on the provided configuration. """
    return init_obj(dataset_config, default_file_name='data', init_type='Dataset')


def define_dataloader(dataset, dataloader_config):
    """ Creates and initializes a dataloader based on the provided configuration. """
    dataloader = DataLoader(dataset,
                            batch_size=dataloader_config['batch_size'],
                            shuffle=dataloader_config['shuffle'],
                            num_workers=dataloader_config['num_workers'])
    return dataloader