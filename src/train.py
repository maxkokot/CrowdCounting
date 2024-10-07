import click
import yaml
import logging

import os

import pytorch_lightning as pl

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import optuna
from copy import deepcopy
import scipy.io

from data.data import DataModule, prepare_transforms, \
    prepare_augmented_transforms
from models.model import Xception


def read_mall_data(mall_data_path, trainval_size, random_state):
    mall_df = pd.read_csv(os.path.join(mall_data_path, 'labels.csv').
                          replace("\\", "/"))
    mall_df['image_name'] = mall_df['id'].\
        apply(lambda x: '{}/frames/frames/seq_{:06d}.jpg'.
              format(mall_data_path, x))
    trainval_mall_df, \
        test_mall_df = train_test_split(mall_df,
                                        train_size=trainval_size,
                                        random_state=random_state)
    return trainval_mall_df, test_mall_df


def read_shan_data(shan_data_path, pretrain_test_size, random_state):

    '''explore nested organization of shanghai dataset
    and extract the data
    '''

    shan_info = _explore_lvl_1_shan(shan_data_path)
    shan_df = pd.DataFrame(shan_info)
    trainval_shan_df, \
        test_shan_df = train_test_split(shan_df,
                                        test_size=pretrain_test_size,
                                        random_state=random_state)
    return trainval_shan_df, test_shan_df


def _explore_lvl_1_shan(shan_data_path):

    shan_info = []
    for dir_1 in os.listdir(shan_data_path):
        shan_info = _explore_lvl_2_shan(shan_info,
                                        shan_data_path,
                                        dir_1)

    return shan_info


def _explore_lvl_2_shan(shan_info, shan_data_path, dir_1):

    for dir_2 in os.listdir(os.path.join(shan_data_path, dir_1).
                            replace("\\", "/")):
        files_lst = os.listdir(os.path.join(shan_data_path, dir_1,
                                            dir_2, 'images').
                               replace("\\", "/"))
        gt_list = os.listdir(os.path.join(shan_data_path, dir_1,
                                          dir_2, 'ground-truth').
                             replace("\\", "/"))
        shan_info = _explore_lvl_3_shan(shan_info, shan_data_path,
                                        dir_1, dir_2, gt_list,
                                        files_lst)

    return shan_info


def _explore_lvl_3_shan(shan_info, shan_data_path, dir_1,
                        dir_2, gt_list, files_lst):

    for i, gt in enumerate(gt_list):

        img_name = files_lst[i]
        mat = scipy.io.loadmat(os.path.join(shan_data_path, dir_1,
                                            dir_2, 'ground-truth',
                                            gt).replace("\\", "/"))
        count_ = mat['image_info'][0][0][0][0][1][0][0]
        shan_info.append({'id': i, 'count': count_,
                          'image_name': os.path.join(shan_data_path,
                                                     dir_1, dir_2,
                                                     'images',
                                                     img_name).
                         replace("\\", "/")})

    return shan_info


def train_model(model, pretrain_n_unfreeze, train_n_unfreeze,
                pretraining_num_epochs, training_num_epochs,
                shan_datamodule, mall_datamodule):
    '''
    Training procedure
    '''

    callbacks_st1 = [pl.callbacks.EarlyStopping(monitor='val_mse', mode='min',
                                                patience=5),
                     pl.callbacks.LearningRateMonitor(
                         logging_interval='epoch')]

    callbacks_st2 = [pl.callbacks.EarlyStopping(monitor='val_mae', mode='min',
                                                patience=5),
                     pl.callbacks.LearningRateMonitor(
                         logging_interval='epoch')]

    callbacks_st3 = [pl.callbacks.EarlyStopping(monitor='val_mse', mode='min',
                                                patience=5),
                     pl.callbacks.LearningRateMonitor(
                         logging_interval='epoch')]

    callbacks_st4 = [pl.callbacks.EarlyStopping(monitor='val_mae', mode='min',
                                                patience=5),
                     pl.callbacks.LearningRateMonitor(
                         logging_interval='epoch')]

    # stage 1: pretraining with MSE
    model.unfreeze_n_layers(pretrain_n_unfreeze)

    trainer = pl.Trainer(max_epochs=pretraining_num_epochs,
                         callbacks=callbacks_st1)
    trainer.fit(model, shan_datamodule)

    # stage 2: pretraining with MAPE
    model.define_loss('mape')
    model.define_scheduller(sch_factor=0.2,
                            sch_patience=3,
                            sch_min_lr=1e-6,
                            sch_monitor='val_mae')

    trainer = pl.Trainer(max_epochs=pretraining_num_epochs,
                         callbacks=callbacks_st2)
    trainer.fit(model, shan_datamodule)

    # stage 3: training with MSE
    model.define_loss('mse')
    model.define_scheduller(sch_factor=0.2,
                            sch_patience=3,
                            sch_min_lr=1e-6,
                            sch_monitor='val_mse')
    model.unfreeze_n_layers(train_n_unfreeze)

    trainer = pl.Trainer(max_epochs=training_num_epochs,
                         callbacks=callbacks_st3)
    trainer.fit(model, mall_datamodule)

    # stage 4: training with MAPE
    model.define_loss('mape')
    model.define_scheduller(sch_factor=0.2,
                            sch_patience=3,
                            sch_min_lr=1e-6,
                            sch_monitor='val_mae')

    trainer = pl.Trainer(max_epochs=training_num_epochs,
                         callbacks=callbacks_st4)
    trainer.fit(model, mall_datamodule)
    return trainer, model


def objective(trial):
    '''
    Function to optimize number
    of unfrozen layers
    '''

    global model_class, model_name, best_mae, best_model, \
        pretraining_num_epochs, training_num_epochs, \
        shan_datamodule, mall_datamodule, model_path

    pretrain_n_unfreeze = trial.suggest_int("pretrain_n_unfreeze", 10, 39)
    train_n_unfreeze = trial.suggest_int("train_n_unfreeze",
                                         pretrain_n_unfreeze, 40)

    curr_model = model_class(loss='mse')
    trainer, curr_model = train_model(curr_model, pretrain_n_unfreeze,
                                      train_n_unfreeze, pretraining_num_epochs,
                                      training_num_epochs, shan_datamodule,
                                      mall_datamodule)

    val_metrics = trainer.validate(curr_model, mall_datamodule)
    val_mae = val_metrics[0]['val_mae']

    if val_mae < best_mae:
        best_mae = val_mae
        best_model = deepcopy(curr_model)
        trainer.save_checkpoint(os.path.join(model_path,
                                             '{}.ckpt'.format(model_name)).
                                replace("\\", "/"))

    return val_mae


def xception(batch_size, pretraining_num_epochs,
             training_num_epochs, n_iter, pretrain_test_size,
             trainval_size, random_state, mall_data_path,
             shan_data_path, model_path, augmentation=False):

    logger = logging.getLogger(__name__)

    if augmentation:
        train_transform, val_transform, \
            test_transform = prepare_augmented_transforms()
        model_name = 'xception_aug'

    else:
        train_transform, val_transform, \
            test_transform = prepare_transforms()
        model_name = 'xception'

    trainval_mall_df, test_mall_df = read_mall_data(mall_data_path,
                                                    trainval_size,
                                                    random_state)
    trainval_shan_df, test_shan_df = read_shan_data(shan_data_path,
                                                    pretrain_test_size,
                                                    random_state)

    mall_datamodule = DataModule(trainval_mall_df, test_mall_df,
                                 mall_data_path, train_transform,
                                 val_transform, test_transform,
                                 trainval_size, batch_size,
                                 random_state)

    shan_datamodule = DataModule(trainval_shan_df,
                                 test_shan_df,
                                 shan_data_path, train_transform,
                                 val_transform, test_transform,
                                 trainval_size, batch_size,
                                 random_state)

    model_class = Xception
    best_mae = np.inf
    best_model = None

    logger.info(f"Tuning hyperparameters for {model_name}")
    current_study = optuna.create_study()
    current_study.optimize(objective, n_trials=n_iter)

    trainer = pl.Trainer()
    xception_best = deepcopy(best_model)
    logger.info(f"Validation scores of {model_name}")
    trainer.validate(xception_best, mall_datamodule)

    logger.info(f"Test scores of {model_name}")
    trainer.test(xception_best, mall_datamodule)


@click.command(name="train")
@click.option('--model_config_path', default='../config/model_config.yaml')
def train_command(model_config_path):
    logger = logging.getLogger(__name__)

    with open(model_config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    batch_size = config['batch_size']
    pretraining_num_epochs = config['pretraining_num_epochs']
    training_num_epochs = config['training_num_epochs']
    n_iter = config['n_iter']
    pretrain_test_size = config['pretrain_test_size']
    trainval_size = config['trainval_size']
    random_state = config['random_state']
    mall_data_path = config['mall_data_path']
    shan_data_path = config['shan_data_path']
    model_path = config['model_path']

    logger.info('Training Xception with no augmentations applied')
    xception(batch_size, pretraining_num_epochs,
             training_num_epochs, n_iter, pretrain_test_size,
             trainval_size, random_state, mall_data_path,
             shan_data_path, model_path, augmentation=False)

    logger.info('Training Xception with augmentations')
    xception(batch_size, pretraining_num_epochs,
             training_num_epochs, n_iter, pretrain_test_size,
             trainval_size, random_state, mall_data_path,
             shan_data_path, model_path, augmentation=True)

    logger.info('Models have been fitted and saved')


if __name__ == "__main__":
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    train_command()
