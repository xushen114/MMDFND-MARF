# -*-codeing = utf-8 -*-
# @Time : 2023-12-1215:43
# @Author : 童宇
# @File : main.py
# @software :
import os
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='MMDFND')
parser.add_argument('--dataset', default='weibo21')#weibo21 %% weibo
#parser.add_argument('--model_name', default='domain_ple6')
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--max_len', type=int, default=197)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--early_stop', type=int, default=3)
parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
#parser.add_argument('--root_path', default='./Weibo_21/')
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--bert', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
parser.add_argument('--batchsize', type=int, default=64)
parser.add_argument('--seed', type=int, default=3074)
parser.add_argument('--gpu', default='0')
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
#parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default= './param_model')
parser.add_argument('--tag', default=None)
parser.add_argument('--output_dir', default='./outputs')
parser.add_argument('--limit_batches', type=int, default=0)
parser.add_argument('--eval_only', action='store_true')
parser.add_argument('--eval_ckpt', default='param_model/MMDFND/parameter_mmdfnd.pkl')
parser.add_argument('--amp', action='store_true')
parser.add_argument('--optim', default='adam', choices=['adam', 'adafactor'])
parser.add_argument('--ada_lr', type=float, default=1e-3)
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.enabled = True


if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

tag = args.tag or time.strftime('run_%Y%m%d_%H%M%S')
config = {
        'use_cuda': True,
        'batchsize': args.batchsize,
        'max_len': args.max_len,
        'early_stop': args.early_stop,
        'num_workers': args.num_workers,
        'vocab_file': vocab_file,
        'emb_type': args.emb_type,
        'bert': args.bert,
        'root_path': args.root_path,
        'weight_decay': 5e-5,
        'model':
            {
            'mlp': {'dims': [384], 'dropout': 0.2}
            },
        'emb_dim': emb_dim,
        'lr': args.lr,
        'epoch': args.epoch,
        'model_name': args.model_name,
        'seed': args.seed,
        'save_param_dir': args.save_param_dir,
        'dataset':args.dataset,
        'tag': tag,
        'output_dir': args.output_dir,
        'limit_batches': args.limit_batches,
        'eval_only': args.eval_only,
        'eval_ckpt': args.eval_ckpt,
        'amp': args.amp,
        'optim': args.optim,
        'ada_lr': args.ada_lr
        }




if __name__ == '__main__':
    print('MMDFND | tag={} | seed={} | batch={} | epochs={} | lr={} | optim={}'.format(
        tag, args.seed, args.batchsize, args.epoch, args.lr, args.optim), flush=True)
    Run(config = config).main()
