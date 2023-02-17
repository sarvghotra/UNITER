"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

sampler for length bucketing (batch by tokens)
"""
import math
import random

# import horovod.torch as hvd
import torch
import blip_utils
from torch.utils.data import Sampler
from cytoolz import partition_all


class TokenBucketSampler(Sampler):
    """
    This sampler does the following:
    1. Shuffle the dataset
    2. Group the dataset into buckets by length
    3. Shuffle the buckets
    4. Fill the buckets into batches until max_token (include padding)
    5. Shuffle the batches
    6. Repeat the above steps for each epoch
    """
    def __init__(self, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()
        random.shuffle(ids)

        #
        buckets = [
            # Sort each bucket by the sort function.
            sorted(ids[i:i+self._bucket_size], key=self._sort_fn, reverse=True)
            # Split the list of ids into buckets of size self._bucket_size.
            for i in range(0, len(ids), self._bucket_size)]

        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


def sort_fn(lens):
    def _sort_fn(i):
        return lens[i]

    return _sort_fn
    # return self._lens[i]


def sort_as_per_token_bucket(ids, _bucket_size, lens, _size_mul, _max_tok, _droplast):
    _lens = lens
    buckets = [
        # Sort each bucket by the sort function.
        sorted(ids[i:i+_bucket_size], key=sort_fn(lens), reverse=True)
        # Split the list of ids into buckets of size self._bucket_size.
        for i in range(0, len(ids), _bucket_size)]

    # fill batches until max_token (include padding)
    batches = []
    for bucket in buckets:
        max_len = 0
        batch_indices = []
        for indices in partition_all(_size_mul, bucket):
            max_len = max(max_len, max(_lens[i] for i in indices))
            if (max_len * (len(batch_indices) + _size_mul)
                    > _max_tok):
                if not batch_indices:
                    raise ValueError(
                        "max_tokens too small / max_seq_len too long")
                assert len(batch_indices) % _size_mul == 0
                batches.append(batch_indices)
                batch_indices = list(indices)
            else:
                batch_indices.extend(indices)
        if not _droplast and batch_indices:
            batches.append(batch_indices)
    random.shuffle(batches)

    return batches



class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = hvd.size()
        if rank is None:
            rank = hvd.rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset)
                                         * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        if self.shuffle:
            shufle_ind = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shufle_ind]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class DistributedTokenBucketSampler(Sampler):
    def __init__(self, dataset, lens, bucket_size, batch_size,
                 droplast=False, size_multiple=8,
                 num_replicas=None, rank=None, shuffle=True):
        self._lens = lens
        self._max_tok = batch_size
        self._bucket_size = bucket_size
        self._droplast = droplast
        self._size_mul = size_multiple

        if num_replicas is None:
            num_replicas = blip_utils.get_world_size()
        if rank is None:
            rank = blip_utils.get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset)
                                         * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def _create_ids(self):
        return list(range(len(self._lens)))

    def _sort_fn(self, i):
        return self._lens[i]

    def __iter__(self):
        ids = self._create_ids()

        random.shuffle(ids)

        buckets = [sorted(ids[i:i+self._bucket_size],
                          key=self._sort_fn, reverse=True)
                   for i in range(0, len(ids), self._bucket_size)]
        # fill batches until max_token (include padding)
        batches = []
        for bucket in buckets:
            max_len = 0
            batch_indices = []
            for indices in partition_all(self._size_mul, bucket):
                max_len = max(max_len, max(self._lens[i] for i in indices))
                if (max_len * (len(batch_indices) + self._size_mul)
                        > self._max_tok):
                    if not batch_indices:
                        raise ValueError(
                            "max_tokens too small / max_seq_len too long")
                    assert len(batch_indices) % self._size_mul == 0
                    batches.append(batch_indices)
                    batch_indices = list(indices)
                else:
                    batch_indices.extend(indices)
            if not self._droplast and batch_indices:
                batches.append(batch_indices)
        random.shuffle(batches)


        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]

        if self.shuffle:
            shufle_ind = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shufle_ind]
        assert len(indices) == self.num_samples

        return iter(indices)



        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")
