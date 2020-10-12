from __future__ import print_function

import argparse
import os
from trainers import SourceonlyTrainer, AdversarialTrainer, RobustAdversarialTrainer
import json
import utils
from pathlib import Path


# Training settings
def parse_args():
    parser = argparse.ArgumentParser(description='Visda Classification')
    parser.add_argument('--cfg-path', type=str, required=True,
                        help='config path')
    parser.add_argument('--ano_type', type=int, default=-1)
    parser.add_argument('--rho', type=float, default=-1)
    parser.add_argument('--ent_weight', type=float, default=-1)
    parser.add_argument('--vat_weight', type=float, default=-1)
    parser.add_argument('--domain_src', type=str, default='')
    parser.add_argument('--domain_tgt', type=str, default='')
    parser.add_argument('--method', type=str, default='')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--logdir', type=str, default='')
    parser.add_argument('--run_id', type=int, default=-1)
    args = parser.parse_args()
    return args


def main(args):
    assert os.path.exists(args.cfg_path)

    # Forming config
    with open(args.cfg_path) as json_file:
        config = json.load(json_file)

    args_to_override = ['method', 'model', 'ano_type', 'logdir', 'rho', 'ent_weight',
                        'domain_src', 'domain_tgt']
    numeric_keys = ['ano_type', 'rho', 'ent_weight', 'vat_weight']
    arg_dict = vars(args)
    for key in args_to_override:
        if key in numeric_keys:
            if arg_dict[key] > -1:
                config[key] = arg_dict[key]
        else:
            if arg_dict[key] != '':
                config[key] = arg_dict[key]
    config = utils.ConfigMapper(config)

    # Initializing save_path
    if config.dataset == 'DomainNet':
        domain_list = '{}_{}'.format(config.domain_src, config.domain_tgt)
        config.logdir = os.path.join(config.logdir, config.exp, domain_list, config.method, config.model,
                                     'ent_{}_vat_{}_rho_{}'.format(config.ent_weight, config.vat_weight, config.rho))
    else:
        config.logdir = os.path.join(config.logdir, config.exp, config.method, config.model,
                                     'ent_{}_vat_{}_rho_{}'.format(config.ent_weight, config.vat_weight, config.rho))
    if args.run_id > -1:
        config.logdir += '_run_{}'.format(args.run_id)
    Path(config.logdir).mkdir(parents=True, exist_ok=True)

    # Creating trainer
    if config.method == 'sourceonly':
        trainer = SourceonlyTrainer(config)
    elif config.method == 'adversarial':
        trainer = AdversarialTrainer(config)
    elif config.method == 'robust_adversarial':
        trainer = RobustAdversarialTrainer(config)

    # Training !!
    trainer.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)
