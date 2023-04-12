"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

UNITER finetuning for VQA
"""
import argparse
import json
import os
import numpy as np
from os.path import abspath, dirname, exists, join
from time import time

import torch
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, Adamax

# from apex import amp
# from horovod import torch as hvd
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch_ema import ExponentialMovingAverage
import blip_utils

from tqdm import tqdm

from data import (TokenBucketSampler, PrefetchLoader,
                  TxtTokLmdb, ImageLmdbGroup, ConcatDatasetWithLens,
                  VqaDataset, VqaEvalDataset, HackyVqaOODEvalDataset,
                  vqa_collate, vqa_eval_collate)
from data.loader import move_to_cuda
from utils.distributed import DistributedTokenBucketSampler
from model.vqa import UniterForVisualQuestionAnswering
from optim import AdamW, build_scheduler, LinearDecayPlateau

from utils.logger import LOGGER, TB_LOGGER, RunningMeter, add_log_to_file, SetupWandb
from utils.distributed import (all_reduce_and_rescale_tensors, all_gather_list,
                               broadcast_tensors)
from utils.save import ModelSaver, save_training_meta
from utils.misc import NoOp, parse_with_config, set_dropout, set_random_seed
from utils.misc import create_kl_loss
from utils.const import BUCKET_SIZE, IMG_DIM
from utils.data import create_eval_datasets


def build_dataloader(dataset, collate_fn, is_train, opts):
    batch_size = (opts.train_batch_size if is_train
                  else opts.val_batch_size)
    # sampler = TokenBucketSampler(dataset.lens, bucket_size=BUCKET_SIZE,
    #                              batch_size=batch_size, droplast=is_train)
    sampler = DistributedTokenBucketSampler(dataset,
                                rank=blip_utils.get_rank(),
                                num_replicas=blip_utils.get_world_size(),
                                lens=dataset.lens,
                                bucket_size=BUCKET_SIZE,
                                batch_size=batch_size, drop_last=is_train)
    dataloader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=opts.n_workers,
                            pin_memory=opts.pin_mem, collate_fn=collate_fn)
    # FIXME: figure out if this was needed?
    # dataloader = PrefetchLoader(dataloader)
    return dataloader


def build_optimizer(model, opts):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'vqa_output' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'vqa_output' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': opts.learning_rate,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': opts.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    # currently Adam only
    if opts.optim == 'adam':
        OptimCls = Adam
    elif opts.optim == 'adamax':
        OptimCls = Adamax
    elif opts.optim == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=opts.learning_rate, betas=opts.betas)
    return optimizer


def main(opts):
    blip_utils.init_distributed_mode(opts)
    n_gpu = blip_utils.get_world_size()
    device = torch.device(opts.device) #"cuda", blip_utils.local_rank())
    # torch.cuda.set_device(blip_utils.get_local_rank(opts))
    rank = blip_utils.get_rank()
    opts.rank = rank

    # setup wandb
    wb_logger = SetupWandb(opts.exp_name, opts)
    wb_logger({
        'device': opts.device,
        'n_gpu': n_gpu,
        'rank': blip_utils.get_rank(),
        '16-bits training': opts.fp16
    })
    LOGGER.info("device: {} n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(
                    device, n_gpu, blip_utils.get_rank(), opts.fp16))

    if opts.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, "
                         "should be >= 1".format(
                            opts.gradient_accumulation_steps))

    set_random_seed(opts.seed)

    ans2label = json.load(open(f'{dirname(abspath(__file__))}'
                               f'/utils/ans2label.json'))
    label2ans = {label: ans for ans, label in ans2label.items()}

    # load DBs and image dirs
    all_img_dbs = ImageLmdbGroup(opts.conf_th, opts.max_bb, opts.min_bb,
                                 opts.num_bb, opts.compressed_db)
    # train
    LOGGER.info(f"Loading Train Dataset "
                f"{opts.train_txt_dbs}, {opts.train_img_dbs}")
    train_datasets = []
    for txt_path, img_path in zip(opts.train_txt_dbs, opts.train_img_dbs):
        img_db = all_img_dbs[img_path]
        txt_db = TxtTokLmdb(txt_path, opts.max_txt_len)
        train_datasets.append(VqaDataset(len(ans2label), txt_db, img_db))
    train_dataset = ConcatDatasetWithLens(train_datasets)
    train_dataloader = build_dataloader(train_dataset, vqa_collate, True, opts)

    # val
    LOGGER.info(f"Loading Val Datasets {opts.val_txt_db}, {opts.val_img_db}")
    val_dataloaders = create_eval_datasets(opts, ans2label, all_img_dbs, is_iid_set=True)
    ood_val_dataloaders = create_eval_datasets(opts, ans2label, all_img_dbs, is_iid_set=False)

    # val_img_db = all_img_dbs[opts.val_img_db]
    # val_txt_db = TxtTokLmdb(opts.val_txt_db, -1)
    # val_dataset = VqaEvalDataset(len(ans2label), val_txt_db, val_img_db)
    # val_dataloader = build_dataloader(val_dataset, vqa_eval_collate,
    #                                   False, opts)

    # ood_val_img_db = all_img_dbs[opts.ood_val_img_db]
    # ood_val_txt_db = TxtTokLmdb(opts.ood_val_txt_db, -1)
    # ood_val_dataset = HackyVqaOODEvalDataset(len(ans2label), ood_val_txt_db, ood_val_img_db)
    # ood_val_dataloader = build_dataloader(ood_val_dataset, vqa_eval_collate,
    #                                   False, opts)

    # Prepare model
    if opts.checkpoint:
        checkpoint = torch.load(opts.checkpoint)
    else:
        checkpoint = {}

    all_dbs = opts.train_txt_dbs + opts.val_txt_db + opts.ood_val_txt_db
    toker = json.load(open(f'{all_dbs[0]}/meta.json'))['bert']
    assert all(toker == json.load(open(f'{db}/meta.json'))['bert']
               for db in all_dbs)
    model = UniterForVisualQuestionAnswering.from_pretrained(
        opts.model_config, checkpoint,
        img_dim=IMG_DIM, num_answer=len(ans2label))
    model.to(device)

    # number of total params
    total_params = sum(p.numel() for p in model.parameters())
    LOGGER.info(f"Total number of parameters: {total_params}")
    # number of trainable params
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOGGER.info(f"Total number of trainable parameters: {trainable_params}")

    ema_model = ExponentialMovingAverage(model.parameters(), decay=opts.ema_decay)  # 0.995
    model = DDP(model, device_ids=[opts.gpu]) #, find_unused_parameters=True)

    # make sure every process has same model parameters in the beginning
    # broadcast_tensors([p.data for p in model.parameters()], 0)
    set_dropout(model, opts.dropout)

    # Prepare optimizer
    optimizer = build_optimizer(model, opts)
    scheduler = build_scheduler(optimizer, opts)
    scaler = torch.cuda.amp.GradScaler()
    # model, optimizer = amp.initialize(model, optimizer,
    #                                   enabled=opts.fp16, opt_level='O2')

    # for name, val_dataloader in val_dataloaders.items():
    #     val_log, results = validate(
    #                         model, val_dataloader, label2ans)

    #     print("Name: ", name)
    #     TB_LOGGER.log_scaler_dict(val_log)

    # print()
    # print("OOD")
    # for name, val_dataloader in ood_val_dataloaders.items():
    #     val_log, results = validate(
    #                         model, val_dataloader, label2ans)

    #     print("Name: ", name)
    #     TB_LOGGER.log_scaler_dict(val_log)

    global_step = 0
    if blip_utils.is_main_process():
        save_training_meta(opts)
        TB_LOGGER.create(join(opts.output_dir, 'log'))
        pbar = tqdm(total=opts.num_train_steps)
        model_saver = ModelSaver(join(opts.output_dir, 'ckpt'))
        json.dump(ans2label,
                  open(join(opts.output_dir, 'ckpt', 'ans2label.json'), 'w'))
        os.makedirs(join(opts.output_dir, 'results'))  # store VQA predictions
        add_log_to_file(join(opts.output_dir, 'log', 'log.txt'))
    else:
        LOGGER.disabled = True
        pbar = NoOp()
        model_saver = NoOp()

    LOGGER.info(f"***** Running training with {n_gpu} GPUs *****")
    LOGGER.info("  Num examples = %d", len(train_dataset))
    LOGGER.info("  Batch size = %d", opts.train_batch_size)
    LOGGER.info("  Accumulate steps = %d", opts.gradient_accumulation_steps)
    LOGGER.info("  Num steps = %d", opts.num_train_steps)
    for val_loader, set_name in zip(val_dataloaders.values(), opts.val_sets_name):
        LOGGER.info("  Num val examples in %s = %d", set_name, len(val_loader.dataset))

    for ood_val_loader, set_name in zip(ood_val_dataloaders.values(), opts.ood_sets_name):
        LOGGER.info("  Num val examples in %s = %d", set_name, len(ood_val_loader.dataset))

    wb_logger({'num_examples': len(train_dataset),
        'train_batch_size': opts.train_batch_size,
        'gradient_accumulation_steps': opts.gradient_accumulation_steps,
        'num_train_steps': opts.num_train_steps
    })

    kl_loss_computer = create_kl_loss()
    kl_loss_wt = opts.kl_loss_wt

    running_loss = RunningMeter('loss')
    running_bce_loss = RunningMeter('bin_ce_loss')
    running_kl_loss = RunningMeter('kl_loss')
    curr_losses = []
    curr_bce_loss = []
    curr_kl_loss = []
    # has_non_nan_loss = False
    model.train()
    n_examples = 0
    n_epoch = 0
    start = time()
    # quick hack for amp delay_unscale bug
    optimizer.zero_grad()
    optimizer.step()
    while True:
        for step, batch in enumerate(train_dataloader):
            batch = move_to_cuda(batch)
            n_examples += batch['input_ids'].size(0)

            enabled = True if opts.fp16 else False
            with torch.cuda.amp.autocast(enabled=enabled, dtype=torch.float16):
                loss, ans_scores = model(batch,
                                         compute_loss=True,
                                         return_ans_scores=True)

                with torch.no_grad():
                    with ema_model.average_parameters():
                        ema_ans_scores = model(batch,
                                                compute_loss=False,
                                                return_ans_scores=True)

                bin_ce_loss = loss.mean() * batch['targets'].size(1)  # instance-leval bce
                kl_loss = kl_loss_computer(ans_scores, ema_ans_scores)
                loss = (1 - kl_loss_wt) * bin_ce_loss + kl_loss_wt * kl_loss
                loss = loss / opts.gradient_accumulation_steps

            scaled_loss = scaler.scale(loss)
            scaled_loss.backward()
            if not torch.isnan(scaled_loss):
                curr_losses.append(loss.item())
                curr_bce_loss.append(bin_ce_loss.item())
                curr_kl_loss.append(kl_loss.item())
                # has_non_nan_loss = True

            if (step+1) % opts.gradient_accumulation_steps == 0 or (step+1) == len(train_dataloader):
                scaler.unscale_(optimizer)

                if opts.grad_norm != -1:
                    grad_norm = clip_grad_norm_(model.parameters(),
                                                opts.grad_norm)

                    TB_LOGGER.add_scalar('grad_norm', grad_norm, global_step)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema_model.update()

                # Native code
                # optimizer.step()
                # optimizer.zero_grad()
                pbar.update(1)

                if len(curr_losses) > 0:
                    running_loss(np.mean(curr_losses))
                    running_bce_loss(np.mean(curr_bce_loss))
                    running_kl_loss(np.mean(curr_kl_loss))
                    # has_non_nan_loss = False
                    curr_losses = []
                    curr_bce_loss = []
                    curr_kl_loss = []


            if (step + 1) % opts.gradient_accumulation_steps == 0:
                global_step += 1

                # learning rate scheduling
                lr_this_step = scheduler.step(global_step)
                TB_LOGGER.add_scalar('lr', lr_this_step, global_step)

                # log loss
                # NOTE: not gathered across GPUs for efficiency
                TB_LOGGER.add_scalar('loss', running_loss.val, global_step)
                TB_LOGGER.add_scalar('BCE_loss', running_bce_loss.val, global_step)
                TB_LOGGER.add_scalar('KL_loss', running_kl_loss.val, global_step)
                TB_LOGGER.step()

                if global_step % opts.log_steps == 0:
                    # monitor training throughput
                    LOGGER.info(f'============Step {global_step}=============')
                    tot_ex = sum(all_gather(n_examples))
                    # tot_ex = n_examples * blip_utils.get_world_size()
                    ex_per_sec = int(tot_ex / (time()-start))
                    LOGGER.info(f'{tot_ex} examples trained at '
                                f'{ex_per_sec} ex/s')
                    TB_LOGGER.add_scalar('perf/ex_per_s',
                                         ex_per_sec, global_step)
                    LOGGER.info('===========================================')
                    wb_logger({'train/loss': running_loss.val,
                               'train/bce_loss': running_bce_loss.val,
                               'train/kl_loss': running_kl_loss.val,
                               'train/lr': lr_this_step,
                               'train/ex_per_sec': ex_per_sec,
                               'train/step': global_step
                               })

                if global_step % opts.valid_steps == 0:
                    is_first_val_set = True
                    for val_dataloader, set_name in zip(val_dataloaders.values(), opts.val_sets_name):
                        print(set_name)
                        val_log, results = validate(
                            model, val_dataloader, label2ans, prefix=set_name)

                        if blip_utils.is_main_process():
                            with open(f'{opts.output_dir}/results/'
                                    f'results_{global_step}_'
                                    f'rank{rank}.json', 'w') as f:
                                json.dump(results, f)

                        TB_LOGGER.log_scaler_dict(val_log)
                        val_log['val/step'] = global_step
                        wb_logger(val_log)

                        if is_first_val_set and isinstance(scheduler, LinearDecayPlateau):
                            is_first_val_set = False
                            key = 'val/acc'
                            key = f'{key.split("/")[0]}/{set_name}_{key.split("/")[1]}'
                            scheduler.lower_lr_if_plateau(val_log[key], global_step)

                        with ema_model.average_parameters():
                            val_log, results = validate(model, val_dataloader, label2ans, prefix='ema_' + set_name)

                        if blip_utils.is_main_process():
                            with open(f'{opts.output_dir}/results/'
                                    f'ema_results_{global_step}_'
                                    f'rank{rank}.json', 'w') as f:
                                json.dump(results, f)
                        TB_LOGGER.log_scaler_dict(val_log)
                        val_log['val/step'] = global_step
                        wb_logger(val_log)

                    for ood_val_dataloader, set_name in zip(ood_val_dataloaders.values(), opts.ood_sets_name):
                        print(set_name)
                        ood_val_log, ood_results = validate(
                            model, ood_val_dataloader, label2ans, prefix='ood_' + set_name)
                        if blip_utils.is_main_process():
                            with open(f'{opts.output_dir}/results/'
                                    f'ood_results_{global_step}_'
                                    f'rank{rank}.json', 'w') as f:
                                json.dump(ood_results, f)

                        TB_LOGGER.log_scaler_dict(ood_val_log)
                        ood_val_log['val/step'] = global_step
                        wb_logger(ood_val_log)

                        with ema_model.average_parameters():
                            ood_val_log, ood_results = validate(
                                model, ood_val_dataloader, label2ans, prefix='ema_ood_' + set_name)
                        if blip_utils.is_main_process():
                            with open(f'{opts.output_dir}/results/'
                                    f'ema_ood_results_{global_step}_'
                                    f'rank{rank}.json', 'w') as f:
                                json.dump(ood_results, f)

                        TB_LOGGER.log_scaler_dict(ood_val_log)
                        ood_val_log['val/step'] = global_step
                        wb_logger(ood_val_log)

                    if blip_utils.is_main_process():
                        model_saver.save(model, global_step)

            if global_step >= opts.num_train_steps:
                break

        if global_step >= opts.num_train_steps:
            break
        n_epoch += 1
        LOGGER.info(f"finished {n_epoch} epochs")

    if opts.num_train_steps % opts.valid_steps != 0:
        val_log, results = validate(model, val_dataloader, label2ans)
        if blip_utils.is_main_process():
            with open(f'{opts.output_dir}/results/'
                    f'results_{global_step}_'
                    f'rank{rank}.json', 'w') as f:
                json.dump(results, f)
        TB_LOGGER.log_scaler_dict(val_log)
        val_log['val/step'] = global_step
        wb_logger(val_log)

        ood_val_log, ood_results = validate(model, ood_val_dataloader, label2ans, prefix='ood')
        if blip_utils.is_main_process():
            with open(f'{opts.output_dir}/results/'
                    f'ood_results_{global_step}_'
                    f'rank{rank}.json', 'w') as f:
                json.dump(ood_results, f)
        TB_LOGGER.log_scaler_dict(ood_val_log)
        val_log['val/step'] = global_step
        wb_logger(ood_val_log)

        # EMA validation
        with ema_model.average_parameters():
            val_log, results = validate(model, val_dataloader, label2ans, prefix='ema')
        if blip_utils.is_main_process():
            with open(f'{opts.output_dir}/results/'
                    f'ema_results_{global_step}_'
                    f'rank{rank}.json', 'w') as f:
                json.dump(results, f)
        TB_LOGGER.log_scaler_dict(val_log)
        val_log['val/step'] = global_step
        wb_logger(val_log)

        with ema_model.average_parameters():
            ood_val_log, ood_results = validate(model, ood_val_dataloader, label2ans, prefix='ema_ood')
        if blip_utils.is_main_process():
            with open(f'{opts.output_dir}/results/'
                    f'ema_ood_results_{global_step}_'
                    f'rank{rank}.json', 'w') as f:
                json.dump(ood_results, f)
        TB_LOGGER.log_scaler_dict(ood_val_log)
        val_log['val/step'] = global_step
        wb_logger(ood_val_log)

        if blip_utils.is_main_process():
            model_saver.save(model, global_step)


@torch.no_grad()
def validate(model, val_loader, label2ans, prefix=None):
    LOGGER.info("start running validation...")
    model.eval()
    val_loss = 0
    tot_score = 0
    n_ex = 0

    st = time()
    results = {}
    for i, batch in enumerate(val_loader):
        batch = move_to_cuda(batch)
        scores = model(batch, compute_loss=False)
        targets = batch['targets']
        loss = F.binary_cross_entropy_with_logits(
            scores, targets, reduction='sum')
        val_loss += loss.item()
        tot_score += compute_score_with_logits(scores, targets).sum().item()
        answers = [label2ans[i]
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        for qid, answer in zip(batch['qids'], answers):
            results[qid] = answer
        n_ex += len(batch['qids'])

    # FIXME: uncomment this
    # return n_ex, tot_score, val_loss, results

    # FIXME: remove this
    tot_time = time()-st
    val_loss = val_loss / n_ex
    val_acc = tot_score / n_ex
    val_log = {'val/loss': val_loss,
               'val/acc': val_acc*100,
               'val/ex_per_s': n_ex/tot_time}
    if prefix is not None:
        val_log = {f'{k.split("/")[0]}/{prefix}_{k.split("/")[1]}': v for k, v in val_log.items()}

    model.train()
    prefix = ""
    if prefix is not None:
        prefix = f"{prefix} "
    LOGGER.info(f"{prefix}validation finished in {int(tot_time)} seconds, "
            f"score: {val_acc*100:.2f}")
    # return val_log, results

    return val_log, results

    # val_loss = sum(all_gather_list(val_loss))
    # tot_score = sum(all_gather_list(tot_score))
    # n_ex = sum(all_gather_list(n_ex))

    # tot_time = time()-st
    # val_loss /= n_ex
    # val_acc = tot_score / n_ex
    # val_log = {'valid/loss': val_loss,
    #            'valid/acc': val_acc,
    #            'valid/ex_per_s': n_ex/tot_time}
    # model.train()
    # LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
    #             f"score: {val_acc*100:.2f}")
    # # return val_log, results

    # return


def all_gather(results):
    world_size = dist.get_world_size()
    all_results = [None for _ in range(world_size)]
    dist.all_gather_object(all_results, results)
    return all_results


# FIXME: test validate with and w/o dist
@torch.no_grad()
def dist_validate(model, val_loader, label2ans):
    st = time()
    n_ex, tot_score, val_loss, results = validate(model, val_loader, label2ans)

    all_results = all_gather(results)

    all_n_ex = sum(all_gather(n_ex))
    all_tot_score = sum(all_gather(tot_score))
    all_val_loss = sum(all_gather(val_loss))

    # val_loss = sum(all_gather_list(val_loss))
    # tot_score = sum(all_gather_list(tot_score))
    # n_ex = sum(all_gather_list(n_ex))

    tot_time = time()-st
    all_val_loss /= n_ex
    all_val_acc = all_tot_score / n_ex
    val_log = {'valid/loss': all_val_loss,
               'valid/acc': all_val_acc,
               'valid/ex_per_s': all_n_ex/tot_time}
    model.train()
    LOGGER.info(f"validation finished in {int(tot_time)} seconds, "
                f"score: {all_val_acc*100:.2f}")
    # return val_log, results

    return val_log, all_results


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1]  # argmax
    one_hots = torch.zeros(*labels.size(), device=labels.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--model_config",
                        default=None, type=str,
                        help="json file for model architecture")
    parser.add_argument("--checkpoint",
                        default=None, type=str,
                        help="pretrained model")

    parser.add_argument(
        "--output_dir", default=None, type=str,
        help="The output directory where the model checkpoints will be "
             "written.")
    parser.add_argument("--exp_name", default=None, type=str,
                        help="name of the experiment for WandB")

    # Prepro parameters
    parser.add_argument('--max_txt_len', type=int, default=60,
                        help='max number of tokens in text (BERT BPE)')
    parser.add_argument('--conf_th', type=float, default=0.2,
                        help='threshold for dynamic bounding boxes '
                             '(-1 for fixed)')
    parser.add_argument('--max_bb', type=int, default=100,
                        help='max number of bounding boxes')
    parser.add_argument('--min_bb', type=int, default=10,
                        help='min number of bounding boxes')
    parser.add_argument('--num_bb', type=int, default=36,
                        help='static number of bounding boxes')

    # training parameters
    parser.add_argument("--train_batch_size", default=4096, type=int,
                        help="Total batch size for training. "
                             "(batch by tokens)")
    parser.add_argument("--val_batch_size", default=4096, type=int,
                        help="Total batch size for validation. "
                             "(batch by tokens)")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16,
                        help="Number of updates steps to accumualte before "
                             "performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lowest_lr", default=3e-5, type=float,
                        help="The lowest learning rate for Adam. when decreasing lr")
    parser.add_argument("--lr_mul", default=10.0, type=float,
                        help="multiplier for top layer lr")
    parser.add_argument("--valid_steps", default=1000, type=int,
                        help="Run validation every X steps")
    parser.add_argument("--num_train_steps", default=100000, type=int,
                        help="Total number of training updates to perform.")
    parser.add_argument("--optim", default='adam',
                        choices=['adam', 'adamax', 'adamw'],
                        help="optimizer")
    parser.add_argument("--betas", default=[0.9, 0.98], nargs='+',
                        help="beta for adam optimizer")
    parser.add_argument('--lr_sched', default='uniter', type=str,
                        help='scheduler for learning rate')
    parser.add_argument("--dropout", default=0.1, type=float,
                        help="tune dropout regularization")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay (L2) regularization")
    parser.add_argument("--grad_norm", default=2.0, type=float,
                        help="gradient clipping (-1 for no clipping)")
    parser.add_argument("--warmup_steps", default=4000, type=int,
                        help="Number of training steps to perform linear "
                             "learning rate warmup for. (invsqrt decay)")
    parser.add_argument("--kl_loss_wt", type=float, help="KL loss weight")

    # device parameters
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead "
                             "of 32-bit")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true', help="pin memory")

    # can use config files
    parser.add_argument('--config', help='JSON config files')

    # DDP parameters
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--distributed', action='store_true', help='use distributed training')

    args = parse_with_config(parser)

    if exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not "
                         "empty.".format(args.output_dir))

    # options safe guard
    if args.conf_th == -1:
        assert args.max_bb + args.max_txt_len + 2 <= 512
    else:
        assert args.num_bb + args.max_txt_len + 2 <= 512

    main(args)
