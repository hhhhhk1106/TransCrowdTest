from __future__ import division
import warnings
from Networks.models import base_patch16_384_token, base_patch16_384_gap
import torch.nn as nn
from torchvision import transforms
import dataset
import math
from utils import save_checkpoint, setup_seed
import torch
import os
import logging
import nni
from nni.utils import merge_parameter
from config import return_args, args
import numpy as np
from image import load_data
import PIL.Image as Image

warnings.filterwarnings('ignore')
import time

setup_seed(args.seed)

logger = logging.getLogger('mnist_AutoML')


def main(args):
    if args['dataset'] == 'ShanghaiA':
        train_file = './npydata/ShanghaiA_train.npy'
        test_file = './npydata/ShanghaiA_test.npy'
    elif args['dataset'] == 'ShanghaiB':
        train_file = './npydata/ShanghaiB_train.npy'
        test_file = './npydata/ShanghaiB_test.npy'
    elif args['dataset'] == 'UCF_QNRF':
        train_file = './npydata/qnrf_train.npy'
        test_file = './npydata/qnrf_test.npy'
    elif args['dataset'] == 'JHU':
        train_file = './npydata/jhu_train.npy'
        test_file = './npydata/jhu_val.npy'
    elif args['dataset'] == 'NWPU':
        train_file = './npydata/nwpu_train.npy'
        test_file = './npydata/nwpu_val.npy'
    elif args['dataset'] == 'myData':
        train_file = './npydata/my_train.npy'
        test_file = './npydata/my_test.npy'

    with open(train_file, 'rb') as outfile:
        train_list = np.load(outfile).tolist()
    with open(test_file, 'rb') as outfile:
        val_list = np.load(outfile).tolist()

    print(len(train_list), len(val_list))

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_id']

    if args['model_type'] == 'token':
        model = base_patch16_384_token(pretrained=True)
    elif args['model_type'] == 'gap':
        model = base_patch16_384_gap(pretrained=True)

    model = nn.DataParallel(model, device_ids=[0])
    model = model.cuda()

    # criterion = nn.L1Loss(size_average=False).cuda()

    # optimizer = torch.optim.Adam(
    #     [  #
    #         {'params': model.parameters(), 'lr': args['lr']},
    #     ], lr=args['lr'], weight_decay=args['weight_decay'])

    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300], gamma=0.1, last_epoch=-1)
    # print(args['pre'])

    # args['save_path'] = args['save_path'] + str(args['rdt'])
    print(args['save_path'])
    if not os.path.exists(args['save_path']):
        os.makedirs(args['save_path'])

    if args['pre']:
        if os.path.isfile(args['pre']):
            print("=> loading checkpoint '{}'".format(args['pre']))
            checkpoint = torch.load(args['pre'])
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            args['start_epoch'] = checkpoint['epoch']
            args['best_pred'] = checkpoint['best_prec1']
        else:
            print("=> no checkpoint found at '{}'".format(args['pre']))

    torch.set_num_threads(args['workers'])

    # print(args['best_pred'], args['start_epoch'])


    '''inference'''
    # prec1 = validate(test_data, model, args)

    # print(' * best MAE {mae:.3f} '.format(mae=args['best_pred']))
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225]),
    ])

    test_path = "./dataset/test_data/images/"
    img_paths = [f"{test_path}{i}.jpg" for i in range(1, 1001)]


    for i in range(len(img_paths)):
        img = transform((Image.open(img_paths[i]).convert('RGB')))
        img = img.cuda()
        # img = Variable(img)
        output = model(img.unsqueeze(0))

        ans = output.detach().cpu().sum()
        ans = "{:.2f}".format(ans.item())
        print(f"{i+1},{ans}")





class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    # print(params)

    main(params)
