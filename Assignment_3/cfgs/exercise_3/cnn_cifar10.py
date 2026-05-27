from functools import partial
from copy import deepcopy

import torch
import torch.nn as nn

from src.data_loaders.data_modules import CIFAR10DataModule
from src.trainers.cnn_trainer import CNNTrainer
from src.models.cnn.model import ConvNet
from src.models.cnn.metric import TopKAccuracy

q1_experiment = dict(
    name='CIFAR10_CNN',

    model_arch=ConvNet,
    model_args=dict(
        input_size=3,
        num_classes=10,
        hidden_layers=[128, 512, 512, 512, 512],
        activation=nn.ReLU,
        norm_layer=nn.Identity,
        drop_prob=0.4,
    ),

    datamodule=CIFAR10DataModule,
    data_args=dict(
        data_dir="data/exercise-2",  # You may need to change this for Colab.
        transform_preset='CIFAR10',
        batch_size=200,
        shuffle=True,
        heldout_split=0.1,
        num_workers=6,
    ),

    optimizer=partial(
        torch.optim.Adam,
        lr=0.002, weight_decay=0.001, amsgrad=True,
    ),
    lr_scheduler=partial(
        torch.optim.lr_scheduler.StepLR,
        step_size=5, gamma=0.8
    ),

    criterion=nn.CrossEntropyLoss,
    criterion_args=dict(),

    metrics=dict(
        top1=TopKAccuracy(k=1),
        top5=TopKAccuracy(k=5),
    ),

    trainer_module=CNNTrainer,
    trainer_config=dict(
        n_gpu=1,
        epochs=10,
        eval_period=1,
        save_dir="Saved",
        save_period=10,
        monitor="off",
        early_stop=0,

        log_step=100,
        tensorboard=True,
        wandb=False,
    ),

)


#########  TODO #####################################################
#  You would need to create the following config dictionaries       #
#  to use them for different parts of Q2 and Q3.                    #
#  Feel free to define more config files and dictionaries if needed.#
#  But make sure you have a separate config for every question so   #
#  that we can use them for grading the assignment.                 #
#####################################################################

q2a_normalization_experiment = deepcopy(q1_experiment)
q2a_normalization_experiment.update(dict(
    name='CIFAR10_CNN_BatchNorm',
))
# Swapped nn.Identity with nn.BatchNorm2d and increased epochs to 50
q2a_normalization_experiment['model_args']['norm_layer'] = nn.BatchNorm2d
q2a_normalization_experiment['trainer_config']['epochs'] = 50
q2a_normalization_experiment['trainer_config']['save_period'] = 100
q2a_normalization_experiment['trainer_config']['monitor'] = "max eval_top1"

# Instead of modifying Q1 directly, we create an extended version of it 
# to observe the overfitting behavior requested in the prompt.
q2b_baseline_extended_experiment = deepcopy(q1_experiment)
q2b_baseline_extended_experiment.update(dict(
    name='CIFAR10_CNN_Extended', # This saves checkpoints to a new folder
))
q2b_baseline_extended_experiment['trainer_config']['epochs'] = 50
q2b_baseline_extended_experiment['trainer_config']['save_period'] = 100
q2b_baseline_extended_experiment['trainer_config']['monitor'] = "max eval_top1"

q2c_earlystop_experiment = deepcopy(q2a_normalization_experiment)
q2c_earlystop_experiment.update(dict(
    name='CIFAR10_CNN_EarlyStop',
))
q2c_earlystop_experiment['trainer_config']['save_period'] = 100
q2c_earlystop_experiment['trainer_config']['early_stop'] = 4

# Augmentation 1: Spatial/Geometric Shifts
q3a_aug1_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug1_experiment.update(dict(
    name='CIFAR10_CNN_Aug1_Geometric',
))
q3a_aug1_experiment['data_args']['transform_preset'] = 'CIFAR10_WithFlip'
q3a_aug1_experiment['model_args']['drop_prob'] = 0.0
q3a_aug1_experiment['trainer_config']['epochs'] = 30

# Augmentation 2: Color Space Distortions
q3a_aug2_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug2_experiment.update(dict(
    name='CIFAR10_CNN_Aug2_Color',
))
q3a_aug2_experiment['data_args']['transform_preset'] = 'CIFAR10_ColorJitter'
q3a_aug2_experiment['model_args']['drop_prob'] = 0.0
q3a_aug2_experiment['trainer_config']['epochs'] = 30

# Augmentation 3: Heavy Combined Pipeline
q3a_aug3_experiment = deepcopy(q2a_normalization_experiment)
q3a_aug3_experiment.update(dict(
    name='CIFAR10_CNN_Aug3_Heavy',
))
q3a_aug3_experiment['data_args']['transform_preset'] = 'CIFAR10_HeavyAug'
q3a_aug3_experiment['model_args']['drop_prob'] = 0.0
q3a_aug3_experiment['trainer_config']['epochs'] = 30


# Dropout Sweep Experiment (Vanilla Data Distribution Only)
q3b_dropout_experiment = deepcopy(q2a_normalization_experiment)
q3b_dropout_experiment.update(dict(
    name='CIFAR10_CNN_Dropout_Sweep',
))
q3b_dropout_experiment['data_args']['transform_preset'] = 'CIFAR10'
q3b_dropout_experiment['model_args']['drop_prob'] = 0.3
q3b_dropout_experiment['trainer_config']['epochs'] = 30

# define more config dictionaries if needed...
