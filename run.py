from simclr import SimCLR
from simclr import SimCLRAdv
from simclr_icm import IcmSimCLR
from simclr_icm import IcmSimCLRv2
import yaml
from data_aug.dataset_wrapper import DataSetWrapper


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])

    if config["exp_type"] == "adversarial":
        print("Use Adversarial Augmentation.")
        simclr = SimCLRAdv(dataset, config)
    elif config["exp_type"] == "icm_adversarial":
        simclr = IcmSimCLR(dataset, config)
    elif config["exp_type"] == "icm_adversarial_v2":
        simclr = IcmSimCLRv2(dataset, config)
    elif config["exp_type"] == "normal":
        simclr = SimCLR(dataset, config)
    else:
        raise ValueError("Unrecognized experiment type: {}".format(config["exp_type"]))
    simclr.train()


if __name__ == "__main__":
    main()
