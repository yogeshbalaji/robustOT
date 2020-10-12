# Main code for running fairness experiments

import argparse
from trainer import GAN
import utils
import json
import shutil
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch GAN hub')
    parser.add_argument('--base_cfg_path', default='configs/base_config.json', type=str,
                        help='path to config file')
    parser.add_argument('--cfg_path', default='', type=str,
                        help='path to config file')
    parser.add_argument('--anomaly_frac', default=-1, type=float,
                        help='Fraction of anomalies')
    parser.add_argument('--weight_update_iters', default=-1, type=int,
                        help='Number of iters to update weights')
    parser.add_argument('--rho', default=-1, type=float,
                        help='rho used')
    parser.add_argument('--weight_update', default=-1, type=int,
                        help='Whether to update weights or not')
    parser.add_argument('--weight_update_type', default=-1, type=int,
                        help='Weight update type')
    parser.add_argument('--run_id', default=-1, type=int,
                        help='Run id')
    args = parser.parse_args()

    with open(args.base_cfg_path, "r") as fp:
        configs = json.load(fp)

    # Overriding base configs
    if args.cfg_path != '':
        with open(args.cfg_path, "r") as fp:
            exp_configs = json.load(fp)
        for k in exp_configs.keys():
            configs[k] = exp_configs[k]

    # Overriding with parser args
    logname = ''
    args_to_override = ['anomaly_frac', 'weight_update_iters', 'rho', 'weight_update', 'weight_update_type', 'run_id']
    arg_dict = vars(args)
    for key in arg_dict:
        if key in args_to_override:
            if arg_dict[key] > -1:
                if key == 'weight_update':
                    if arg_dict[key] == 0:
                        configs[key] = False
                        logname += '_baseline'
                    else:
                        configs[key] = True
                        logname += '_robust'
                elif key == 'weight_update_type':
                    if arg_dict[key] == 0:
                        configs[key] = 'discrete'
                        logname += '_discrete'
                    else:
                        configs[key] = 'cont'
                        logname += '_cont'
                else:
                    configs[key] = arg_dict[key]
                    logname += '_{}_{}'.format(key, configs[key])
    configs['logdir'] = configs['logdir'] + configs['expname'] + '/'
    configs['logdir'] += logname

    Path(configs['logdir']).mkdir(parents=True, exist_ok=True)
    src_path = args.cfg_path
    dst_path = os.path.join(configs['logdir'], 'config.json')
    shutil.copy(src_path, dst_path)

    configs = utils.ConfigMapper(configs)
    return configs


def main(config):
    trainer = GAN(config)
    trainer.train()


if __name__ == '__main__':
    config = parse_args()
    main(config)

