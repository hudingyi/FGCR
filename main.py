#!/usr/bin/env python
# -*- coding:utf-8 -*-
# date: 2023/06
# author:Dingyi Hu
# emai:hudingyi@buaa.edu.cn

import argparse
import os
import time
import numpy as np
from yacs.config import CfgNode
import torch
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from itertools import chain
from model import FGCR, fgcr_inference
from loader import KernelWSILoader
from loader import DistributedWeightedSampler
from utils import *
import random
import builtins
import warnings

def arg_parse():
    parser = argparse.ArgumentParser(description='GCN-Hash arguments.')

    parser.add_argument('--cfg', type=str,
            default='configs/gastic_all_resent_prompt.yaml',
            help='The path of yaml config file')

    parser.add_argument('--fold', type=int, default=0, help='use all data for training if it is set -1')
    parser.add_argument('--batch-size', type=int, default=28,
                        help='Batch size.')
    parser.add_argument('--num-epochs', type=int, default=150,
                        help='Number of epochs to train.')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of workers to load data.')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate.')
    parser.add_argument('--shuffle-train', default=False, action='store_true',
                        help='Shuffle the train list')
    parser.add_argument('--weighted-sample', action='store_true',
                        help='Balance the sample number from different types\
                              in each mini-batch for training.')

    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true', default=False,
                        help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    
    parser.add_argument('--redo', default=False, action='store_true',
                        help='Ingore all the cache files and re-train the model.')
    parser.add_argument('--eval-freq', type=int, default=9,
                        help='The epoch frequency to evaluate on vlidation and test sets.')
    parser.add_argument('--print-freq', type=int, default=10,
                        help='The mini-batch frequency to print results.')
    parser.add_argument('--prefix-name', type=str, default='FGCR',
                        help='A prefix for the model name.')
    parser.add_argument('--kernel-num', default=25, type=int,
                        help='kernel number to use.')
    parser.add_argument('--prompt-num', default=25, type=int,
                        help='prompt number to use.')
    parser.add_argument('--node-aug', default=False, action='store_true',
                        help='Randomly reduce the nodes for data augmentation》')

    return parser.parse_args()


def main(args):
    if args.cfg:
        cfg = CfgNode(new_allowed=True)
        cfg.merge_from_file(args.cfg)
        merge_config_to_args(args, cfg)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node,
                 args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node, args):
    graph_model_path = os.path.join(args.kat_dir, args.prefix_name)

    checkpoint = []
    if not args.redo:
        checkpoint_path = os.path.join(
            graph_model_path, 'checkpoint.pth.tar')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device('cpu'))
            print("=> loading checkpoint")

    if checkpoint:
        args.start_epoch = checkpoint['epoch']
        if args.start_epoch >= args.num_epochs:
            print('model training is finished')
            return 0
        else:
            print('model train from epoch {}/{}'.format(args.start_epoch, args.num_epochs))
    else:
        args.start_epoch = 0

    args.gpu = gpu
    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None and not args.distributed:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.rank == -1:
            if args.dist_url == "env://":
                args.rank = int(os.environ["RANK"])
            elif 'SLURM_PROCID' in os.environ:
                args.rank = int(os.environ['SLURM_PROCID'])
                
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    graph_list_dir = args.graph_list_dir
    # train graph data

    train_set = KernelWSILoader(
            os.path.join(graph_list_dir, 'example_data'),
            max_node_number=args.max_nodes,
            max_kernel_num=args.kernel_num,
        )

    args.input_dim = train_set.get_feat_dim()
    # create model
    prompt_list = np.load("configs/prompt_index.npy")
    model = FGCR(
        patch_dim=args.input_dim,
        prompt_num=args.prompt_num, 
        dim=args.trfm_dim, 
        depth=args.trfm_depth, 
        heads=args.trfm_heads, 
        mlp_dim=args.trfm_mlp_dim, 
        dim_head=args.trfm_dim_head, 
        num_kernel=args.kernel_num,
        prompt_list=prompt_list
    )

    if args.gpu is not None:
        model = model.cuda(args.gpu)

    if os.path.isfile(args.resume):
        print("=> resume checkpoint '{}'".format(args.resume))
        resume_model_params = torch.load(
            args.resume, map_location=torch.device('cpu'))
        model.load_state_dict(resume_model_params['state_dict'])
    else:
        if checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.num_workers = int(args.num_workers / ngpus_per_node)

            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    if args.weighted_sample:
        print('activate weighted sampling')
        if args.distributed:
            train_sampler = DistributedWeightedSampler(
                train_set, train_set.get_weights(), args.world_size, args.rank)
        else:
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                train_set.get_weights(), len(train_set), replacement=True
            )
    else:
        if args.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_set)
        else:
            train_sampler = None

    train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=args.shuffle_train,
            num_workers=args.num_workers, sampler=train_sampler)

    # validation graph data
    val_path = os.path.join(graph_list_dir, 'example_data')
    if not os.path.exists(val_path):
        valid_loader = None
    else:
        valid_set = KernelWSILoader(val_path,
            max_node_number=args.max_nodes,
            max_kernel_num=args.kernel_num,
            )
        valid_loader = torch.utils.data.DataLoader(
            valid_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )
        
    # test graph data
    test_path = os.path.join(graph_list_dir, 'example_data')
    if not os.path.exists(test_path):
        test_loader = None
    else:
        test_set = KernelWSILoader(test_path,
            max_node_number=args.max_nodes,
            max_kernel_num=args.kernel_num,
            )
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, drop_last=False, sampler=None
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(0.3*args.num_epochs), eta_min=5e-8)
    if checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):    
        if not os.path.exists(graph_model_path):
            os.makedirs(graph_model_path)

    best_loss = 100
    for epoch in range(args.start_epoch, args.num_epochs):
        begin_time = time.time()
        train_loss = train(train_loader, model, optimizer, epoch, args)
        print('epoch time: ', time.time()-begin_time)
        scheduler.step()
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank == 0):
            if epoch % args.eval_freq == 0:
                if valid_loader is not None:
                    val_loss = evaluate(valid_loader, model, args, 'Valid')

                if test_loader is not None:
                    test_loss = evaluate(test_loader, model, args, 'Test')

                with open(graph_model_path + '/log_loss.csv', 'a') as f:
                    f.write('{},{:.3f},V,{:.3f}T,{:.3f}, SUB,'.format(
                        epoch, train_loss, val_loss, test_loss)
                            )
                    f.write('\n') 
                if val_loss<best_loss:
                    torch.save({
                            'epoch': epoch + 1,
                            'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                            'optimizer' : optimizer.state_dict(),
                        }, os.path.join(graph_model_path, 'checkpoint.pth.tar'))
                    best_loss = val_loss          

                torch.save({
                        'epoch': epoch + 1,
                        'state_dict': model.module.state_dict() if args.distributed else model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'args': args
                    }, os.path.join(graph_model_path, 'model_{}.pth.tar'.format(epoch + 1)))


def train(train_loader, model, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses,
                             prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    all_prompt = []
    all_pmask = []
    end = time.time()
    for i, (data, label, _) in enumerate(train_loader):
        # measure data loading time
        begin = time.time()
        data_time.update(time.time() - end)
        target = label.cuda(non_blocking=True)

        # compute output
        mlm_logits, tmask, text, cls_out, prompt, pmask, t_cls, img_token, text_token, kat_img_ebd, kmask, p, patch_atten_output, text_emd, text_atten_output = fgcr_inference(
            model, data)
        loss = calc_loss(mlm_logits, tmask, text, cls_out, prompt, pmask, t_cls, img_token,
                                     text_token, kat_img_ebd, kmask, p, patch_atten_output, text_emd, text_atten_output)

        all_prompt.append(data[3][:, :, 0])
        all_pmask.append(data[7][:, :, 0])

        # measure accuracy and record loss
        losses.update(loss.item(), target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # print(time.time()-begin)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    return losses.avg



def evaluate(val_loader, model, args, prefix='Valid'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(val_loader), batch_time, losses,
                             prefix=prefix)

    # switch to evaluate mode
    model.eval()
    y_labels = []
    prompt = []
    pmask = []
    end = time.time()
    
    processing_time = 0
    with torch.no_grad():
        for i, (data, label, _) in enumerate(val_loader):
            target = label.cuda(non_blocking=True)
            # compute output
            pro_start = time.time()
            mlm_logits, tmask, text, cls_out, prompt, pmask, t_cls, img_token, text_token, kat_img_ebd, kmask, p, patch_atten_output, text_emd, text_atten_output = fgcr_inference(
                model, data)
            loss = calc_loss(mlm_logits, tmask, text, cls_out, prompt, pmask, t_cls, img_token,
                                        text_token, kat_img_ebd, kmask, p, patch_atten_output, text_emd, text_atten_output)
            processing_time += (time.time() - pro_start)

            y_labels.append(label)
            prompt.append(data[3][:,:,0])
            pmask.append(data[7][:,:,0])
            # measure accuracy and record loss
            losses.update(loss.item(), target.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)
        
    return losses.avg

def save_feature(val_loader, model, args, graph_model_path, prefix='Valid'):
    batch_time = AverageMeter('Time', ':6.3f')
    progress = ProgressMeter(len(val_loader), batch_time, prefix=prefix)

    # switch to evaluate mode
    model.eval()
    y_labels = []
    all_prompt = []
    all_pmask = []
    slide_total = []
    img_feature = []
    text_feature = []
    end = time.time()
    
    processing_time = 0
    with torch.no_grad():
        for i, (data, label, slide) in enumerate(val_loader):
            target = label.cuda(non_blocking=True)
            # compute output
            pro_start = time.time()
            _, _, _, _, _, _, _, img_token, text_token, _, _, _, _, _, _ = fgcr_inference(
                model, data)
            processing_time += (time.time() - pro_start)

            y_labels.append(label)
            all_prompt.append(data[3][:,:,0])
            all_pmask.append(data[7][:,:,0])
            slide_total.append(slide)
            img_feature.append(img_token.cpu().data)
            text_feature.append(text_token.cpu().data)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

    y_labels = torch.cat(y_labels)
    obj = chain.from_iterable(slide_total)
    slide_total = list(obj)
    img_feature = torch.cat(img_feature)
    text_feature = torch.cat(text_feature)
    all_prompt = torch.cat(all_prompt)
    all_pmask = torch.cat(all_pmask)
    graph_save_path = os.path.jion(graph_model_path, prefix+'_token_feature.pkl')
    with open(graph_save_path, 'wb') as f:
        graph = {
            'prompt':all_prompt.numpy(),
            'pmask': all_pmask.numpy(),
            'img_feature':img_feature.numpy(),
            'text_feature':text_feature.numpy(),
            'labels':y_labels.numpy(),
            'slide_total':slide_total,
            }
        pickle.dump(graph, f)


def multi_cls_loss(im_out, prompt, pmask):
    target = torch.zeros(im_out.size()).cuda()
    ce_loss = torch.zeros(im_out.size()[0]).cuda()
    for i in range(im_out.size()[0]):
        target[i,prompt[i][pmask[i]>0]]=1 
        ce_loss[i] = F.binary_cross_entropy(torch.softmax(im_out[i].clone(),dim=0), target[i].clone())

    return ce_loss.mean()

def APA_Loss(img_feat, kmask, p_ebd, prompt, pmask, T=10):
    img_text_matrix =  einsum('b i j, b j d -> b i d', img_feat, p_ebd.permute(0, 2, 1))/img_feat.size()[2]/T
    target = torch.zeros(p_ebd.size()[0:2]).cuda()
    ce_loss = torch.zeros(p_ebd.size()[0]).cuda()
    soft_pred = torch.zeros(p_ebd.size()[0:2]).cuda()
    soft_pred_k = torch.zeros(img_text_matrix.size()).cuda()
    for i in range(img_text_matrix.size()[0]):
        target[i,prompt[i][pmask[i]>0]]=1 
        kmask_part = kmask[i].reshape(img_text_matrix.size()[1])>0
        img_text_matrix_part = img_text_matrix[i][kmask_part]
        pred_logit_part = torch.sigmoid(img_text_matrix_part)
        soft_pred_part = torch.softmax(img_text_matrix_part,dim=0) *pred_logit_part
        soft_pred_k[i][kmask_part] = soft_pred_part
        soft_pred[i] = soft_pred_part.sum(0)
        ce_loss[i] = F.binary_cross_entropy(soft_pred[i].clone(), target[i].clone())

    return ce_loss.mean(), soft_pred

def WRA_Loss(img_rep, text_rep):
    bz = img_rep.size(0)
    labels = torch.arange(bz).type_as(text_rep).long()
    scores = img_rep.mm(text_rep.t())/img_rep.size(1)
    scores1 = scores.transpose(0, 1)
    loss0 = F.cross_entropy(scores, labels)
    loss1 = F.cross_entropy(scores1, labels)
    loss_ita = (loss0 + loss1)*0.2

    return loss_ita

def cross_sim_loss(kat_img_ebd, data_atten_output, mask):
    data_sim = torch.bmm(kat_img_ebd, data_atten_output.permute(
        1, 2, 0)) / kat_img_ebd.size(2)
    data_num = data_sim.size(1)
    bz = data_sim.size(0)
    mask = rearrange(mask, "b n1 n2 -> (b n1 n2)")>0
    data_sim_1 = rearrange(data_sim, "b n1 n2 -> (b n1) n2")
    targets = torch.arange(data_num).type_as(
        data_sim).long().repeat(bz)
    loss_data_1 = torch.sum(F.cross_entropy(
        data_sim_1[mask], targets[mask], reduction="none") ) / mask.sum()

    data_sim_2 = rearrange(data_sim, "b n1 n2 -> (b n2) n1")
    loss_data_2 = torch.sum(F.cross_entropy(
        data_sim_2[mask], targets[mask], reduction="none") ) / mask.sum()
    loss = (loss_data_1 + loss_data_2) * 0.01

    return loss

def calc_loss(mlm_logits, tmask, text, cls_out, prompt, pmask, t_cls, img_token, 
            text_token, kat_img_ebd, kmask, p, patch_atten_output, text_emd, text_atten_output):
    mlm_loss = F.cross_entropy(mlm_logits.view(-1, mlm_logits.size(-1))[tmask.view(-1)>0], text.view(-1)[tmask.view(-1)>0])*0.1
    pc_loss =  multi_cls_loss(cls_out, prompt, pmask) + multi_cls_loss(t_cls, prompt, pmask)
    WRA_loss = WRA_Loss(img_token, text_token)
    APA_loss, soft_pred, = APA_Loss(kat_img_ebd, kmask, p, prompt, pmask)
    loss_patch = cross_sim_loss(kat_img_ebd, patch_atten_output, kmask)
    loss_text = cross_sim_loss(text_emd, text_atten_output, tmask)
    loss_cross_sim = loss_text+loss_patch
    loss = APA_loss+WRA_loss+mlm_loss+loss_cross_sim+pc_loss
    
    return loss


if __name__ == "__main__":
    args = arg_parse()
    main(args)


