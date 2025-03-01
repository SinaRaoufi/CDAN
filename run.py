import argparse

from utils.parser import create_model, define_dataloader, define_network, define_dataset, parse
from utils.reproducibility import set_seed_and_cudnn


def main(config):
    set_seed_and_cudnn()

    phase = config['phase']
    dataset = define_dataset(config[phase]['dataset'])
    dataloader = define_dataloader(dataset, config[phase]['dataloader']['args'])
    network = define_network(config['model']['networks'][0])

    model = create_model(config=config,
                         network=network,
                         dataloader=dataloader
                        )

    if phase == 'train':
        model.train()
    else:
        model.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/default.json', help='Path to the JSON configuration file')
    parser.add_argument('-p', '--phase', type=str, choices=['train', 'test'], help='Phase to run (train or test)', default='train')

    # parser configs
    args = parser.parse_args()
    config = parse(args)

    main(config)