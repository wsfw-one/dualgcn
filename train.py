import torch
import argparse
import sys
sys.path.append('./options')
from trainer import Trainer
from Conv_TasNet import ConvTasNet
from models.dualgcn_q8 import DualGCN
from dataLoader.dataLoaders import make_dataloader
from options.option import parse
from util.utils import get_logger
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
import sys
import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from time import strftime, localtime
from torch.utils.data import DataLoader
import pdb


def main():
    # Reading option
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default=r'C:\Users\lyd\Desktop\bc\123\options\train\train.yml', help='Path to option YAML file.')
    args = parser.parse_args()

    opt = parse(args.opt, is_tain=True)
    logger = get_logger(__name__)

    logger.info('Building the model of GraphSep')

    net = DualGCN(num_features_global=64, num_features_local=64)
    net = net.to(torch.device("cuda:0"))
    total_params = sum(p.numel() for p in net.parameters())
    print(f"total para:{total_params} parameters")
    # net = ConvTasNet(**opt['net_conf'])
    #print(net)
    logger.info('Building the trainer of GraphSep')
    gpuid = tuple(opt['gpu_ids'])
    trainer = Trainer(
        net=net,
        checkpoint=opt['train']['checkpoint'],
        optimizer=opt['train']['optimizer'],
        gpuid=gpuid,
        optimizer_kwargs=opt['optimizer_kwargs'],
        clip_norm=opt['train']['clip_norm'],
        min_lr=opt['train']['min_lr'],
        patience=opt['train'].get('patience', 5),
        factor=opt['train']['factor'],
        logging_period=opt['train']['logging_period'],
        resume=opt['resume'],
        num_epochs=opt['train'].get('num_epochs', 200),
    # 新增配置传递
        lr_scheduler_config=opt['train'].get('lr_scheduler', {'type': 'cosine', 't0': 10, 't_mult': 2}),
        early_stopping_config=opt['train'].get('early_stopping', {'patience': 20, 'min_delta': 0.001}),
        )

    logger.info('Making the train and test data loader')
    train_loader = make_dataloader(is_train=True, data_kwargs=opt['datasets']['train'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])
    val_loader = make_dataloader(is_train=False, data_kwargs=opt['datasets']['val'], num_workers=opt['datasets']
                                   ['num_workers'], batch_size=opt['datasets']['batch_size'])
    #logger.info('Train data loader: {}, Test data loader: {}'.format(len(train_loader), len(val_loader)))
    trainer.run(train_loader,val_loader)


if __name__ == "__main__":
    main()
