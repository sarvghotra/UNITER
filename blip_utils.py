import math


def cosine_lr_schedule(optimizer, epoch, max_epoch, init_lr, min_lr):
    """Decay the learning rate"""
    lr = (init_lr - min_lr) * 0.5 * (1. + math.cos(math.pi * epoch / max_epoch)) + min_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class CosineLRScheduleIterLearn():
    def __init__(self, max_steps, init_lr, min_lr, nb_param_groups) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.param_groups_curr_step = [0.0] * nb_param_groups

    def step(self, optimizer):
        """Decay the learning rate"""
        for param_gp_idx, param_group in enumerate(optimizer.param_groups):
            curr_step = self.param_groups_curr_step[param_gp_idx]
            lr = (self.init_lr - self.min_lr) * 0.5 * (1. + math.cos(math.pi * curr_step / self.max_steps)) + self.min_lr
            param_group['lr'] = lr

    def step_counter(self):
        for i in range(len(self.param_groups_curr_step)):
            self.param_groups_curr_step[i] += 1.0

    def reset_scheduler_to_init(self, param_gp_idx):
        self.param_groups_curr_step[param_gp_idx] = 0.0

    def state_dict(self):
        return {'param_groups_curr_step': self.param_groups_curr_step}

    def load_state_dict(self, state_dict):
        self.param_groups_curr_step = state_dict['param_groups_curr_step']


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

import numpy as np
import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.4f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None, step=0, total_steps=0):
        i = step
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (total_steps - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, total_steps, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, total_steps, eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))

    def log_now(self, iterable, step, header=None):
        i = step
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0

        data_time.update(time.time() - end)
        iter_time.update(time.time() - end)

        eta_seconds = iter_time.global_avg * (len(iterable) - i)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if torch.cuda.is_available():
            print(log_msg.format(
                i, len(iterable), eta=eta_string,
                meters=str(self),
                time=str(iter_time), data=str(data_time),
                memory=torch.cuda.max_memory_allocated() / MB))
        else:
            print(log_msg.format(
                i, len(iterable), eta=eta_string,
                meters=str(self),
                time=str(iter_time), data=str(data_time)))

        end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def compute_acc(logits, label, reduction='mean'):
    ret = (torch.argmax(logits, dim=1) == label).float()
    if reduction == 'none':
        return ret.detach()
    elif reduction == 'mean':
        return ret.mean().item()

def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def get_local_rank(args):
    if not is_dist_avail_and_initialized():
        return 0
    return args.local_rank


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # FIXME
    # if args.non_distributed:
    #     print('Not using distributed mode')
    #     args.distributed = False
    #     return

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}, word {}): {}'.format(
        args.rank, args.world_size, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    print("process init")
    torch.distributed.barrier()
    print("barrier crossed")
    setup_for_distributed(args.rank == 0)
    print("setup done")


def get_filename_without_ext(filename):
    return os.path.splitext(os.path.basename(filename))[0]


def grad_norms_each_layer(model, avg_lyrs_grad_norm):
    for name, param in model.named_parameters():
        if param.requires_grad:
            avg_lyrs_grad_norm[name].append(param.grad.norm().item())



def get_reset_params_gps(model, reset_split_layer):
    pre_params = []
    post_params = []
    post_split = False

    for name, param in model.named_parameters():
        if param.requires_grad is False:
            print("Non trainable param: ", name)
            continue

        if reset_split_layer in name:
            post_split = True

        if not post_split:
            pre_params.append(param)
        else:
            post_params.append(param)

    if reset_split_layer is not None and not post_split:
        raise ValueError(
            f"reset_split_layer value {reset_split_layer} is not a valid "
            "parameter name in the model"
        )

    return pre_params, post_params

def get_modules_reset_params_gps(model, reset_split_layer, module):
    pre_params = []
    post_params = []
    post_split = False

    for name, param in model.named_parameters():
        if (module not in name):
            continue

        if param.requires_grad is False:
            print("Non trainable param: ", name)
            continue

        if reset_split_layer in name:
            post_split = True

        if not post_split:
            pre_params.append(param)
        else:
            post_params.append(param)

        # if isinstance(reset_factor, dict):
        #     # Note: zero wts are implicit in the following
        #     param.copy_(
        #         (reset_factor['init_wts'] * init_param.to(param)) + (reset_factor['updated_wts'] * param) + \
        #             self.add_noise(param)
        #     )
        # elif reset_factor < 1.0:
        #     param.copy_(
        #         ((1.0 - reset_factor) * init_param.to(param)) + (reset_factor * param)
        #     )

    if reset_split_layer is not None and not post_split:
        raise ValueError(
            f"reset_split_layer value {reset_split_layer} is not a valid "
            "parameter name in the model"
        )
    return pre_params, post_params


def create_optimizer(config, model):
    param_gps_shrink_perturb = []
    param_gp_llf = []
    param_gp_oth = []

    reset_split_layer = config['reset_split_layer']
    if isinstance(reset_split_layer, dict):

        for module in ['visual_encoder', 'text_encoder', 'text_decoder']:
            if reset_split_layer[module] is not None:
                shrink_perturb_params, llf_params = get_modules_reset_params_gps(model, reset_split_layer[module], module)
                param_gps_shrink_perturb.append(shrink_perturb_params)
                param_gp_llf.append(llf_params)
            else:
                for name, param in model.named_parameters():
                    if module not in name:
                        continue
                    param_gp_oth.append(param)
    else:
        param_gps_shrink_perturb, param_gp_llf = get_reset_params_gps(model, reset_split_layer)
        param_gps_shrink_perturb = [param_gps_shrink_perturb]
        param_gp_llf = [param_gp_llf]

    # create Adam optimizer with 2 param groups
    optimizer = torch.optim.AdamW([
            {'params': param_gps_shrink_perturb[0], 'lr': config['init_lr'], 'weight_decay': config['weight_decay']},
    ])
    optimizer.add_param_group({'params': param_gp_llf[0], 'lr': config['init_lr'], 'weight_decay': config['weight_decay']})

    for llf_gp, shrink_perturb_gp in zip(param_gp_llf[1:], param_gps_shrink_perturb[1:]):
        optimizer.add_param_group(
            {'params': shrink_perturb_gp, 'lr': config['init_lr'], 'weight_decay': config['weight_decay']}
        )
        optimizer.add_param_group(
            {'params': llf_gp, 'lr': config['init_lr'], 'weight_decay': config['weight_decay']},
        )

    if len(param_gp_oth) > 0:
        optimizer.add_param_group(
            {'params': param_gp_oth, 'lr': config['init_lr'], 'weight_decay': config['weight_decay']}
        )

    # optz_state = optimizer.state_dict()
    # nb_params_optz = 0
    # for gp in optz_state['param_groups']:
    #     gp = gp['params']
    #     print("boo")
    #     nb_params_optz += sum(p.numel() for p in gp if p.requires_grad)

    # nb_params_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # FIXME: figure it out
    # assert nb_params_optz == nb_params_model, f"No. of params in model {nb_params_model} is not equal to no. of params in optz {nb_params_optz}"

    return optimizer


def reset_adam_optz_state(optimizer, params, group):
    # reset Adam optimizer state
    for p in params:
        if p.grad is not None:
            state = optimizer.state[p]
            # Lazy state initialization

            state['step'] = (
                torch.zeros((1,), dtype=torch.float, device=p.device)
                if optimizer.defaults['capturable'] or optimizer.defaults['fused']
                else torch.tensor(0.)
            )
            # Exponential moving average of gradient values
            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            # Exponential moving average of squared gradient values
            state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            if group['amsgrad']:
                # Maintains max of all exp. moving avg. of sq. grad. values
                state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    return
