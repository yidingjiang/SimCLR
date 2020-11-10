from simclr import SimCLR
from simclr import SimCLRAdv
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    if config["use_adv_aug"]:
        print("Use Adversarial Augmentation.")
        simclr = SimCLRAdv(dataset, config)
    else:
        simclr = SimCLR(dataset, config)
    simclr.train()


if __name__ == "__main__":
    main()
