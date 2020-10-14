# Robust Optimal Transport with Applications in Generative Modeling and Domain Adaptation

This is the official codebase of our NeurIPS 2020 paper "Robust Optimal Transport with Applications in
Generative Modeling and Domain Adaptation".


## GAN Experiments

Go to GAN folder. 

To train robust WGAN experiments on CIFAR-10 dataset corrupted with MNIST, run

`python main.py --base_cfg_path configs/experiments/CIFAR10_MNIST.json --cfg_path configs/unconditional/WGAN.json`

By changing the configs and experiment in base configs, different models and datasets can be run.

## Domain adaptation experiments

To train domain adaptation models, go to DA folder and run

`python main.py --cfg-path configs/robust_adversarial.json`

