"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import json
import torch
import random
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from os.path import abspath, dirname

from .data import DetectFeatTxtTokDataset, pad_tensors, get_gather_index

from pytorch_pretrained_bert import BertTokenizer
from prepro import bert_tokenize


def _get_vqa_target(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


class VqaDataset(DetectFeatTxtTokDataset):
    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_answers = num_answers

    def __getitem__(self, i):
        example = super().__getitem__(i)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_vqa_target(example, self.num_answers)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target


def vqa_collate(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class VqaEvalDataset(VqaDataset):
    def __init__(self, num_answers, *args, **kwargs):
        # pop 'data_size' from kwargs
        data_size = kwargs.pop('data_size', None)
        super().__init__(num_answers, *args, **kwargs)

        # assert isinstance(txt_db, TxtTokLmdb)
        # assert isinstance(img_db, DetectFeatLmdb)
        # self.txt_db = txt_db
        # self.img_db = img_db
        # txt_lens, self.ids = get_ids_and_lens(txt_db)

        self.id_idx = list(range(len(self.ids)))

        if data_size is not None:
            # this is to "deterministically" shuffle the dataset
            # and yet not interfere with other random operations
            # which depend on the seed set by the user
            curr_rand_state = random.getstate()
            rand_no = random.randint(0, 1000000)
            random.seed(938427)
            random.shuffle(self.id_idx)
            random.setstate(curr_rand_state)
            post_rand_no = random.randint(0, 1000000)
            assert rand_no == post_rand_no, "Random state didn't resume as expected"

            self.id_idx = self.id_idx[:data_size]
            # self.ids = self.ids[:data_size]
            # self.txt_lens = self.txt_lens[:data_size]

        txt2img = self.txt_db.txt2img
        self.lens = [self.txt_lens[i] + self.img_db.name2nbb[txt2img[self.ids[i]]]
                     for i in self.id_idx]

        self.num_answers = num_answers

    def __len__(self):
        return len(self.id_idx)

    def _get_example(self, id_):
        example = self.txt_db[id_]
        return example

    def __getitem__(self, i):
        qid = self.ids[self.id_idx[i]]
        example = self._get_example(qid)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        if 'target' in example:
            target = _get_vqa_target(example, self.num_answers)
        else:
            target = None

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target


class GQVQAEvalDataset(VqaDataset):
    # □ 2405722
    # □ 2331963
    # □ 2400861
    # □ 1339
    # □ 2402467
    # □ 2352110

    def __init__(self, ip_file_txt, num_answers, toker, txt_db, img_db):
        # self.txt_db = txt_db
        # self.img_db = img_db
        self.num_answers = num_answers
        jsonl = json.load(open(ip_file_txt, 'r'))
        toker = BertTokenizer.from_pretrained(
            toker, do_lower_case='uncased' in toker)
        tokenizer = bert_tokenize(toker)
        self.examples, self.txt_lens, self.ids, self.orig_data_size, self.valid_ans_size, txt2img = process_gq_vqa(jsonl, tokenizer)

        meta = {}
        meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
        meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
        meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
        meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]


        self.cls_ = meta['CLS']
        self.sep = meta['SEP']
        self.mask = meta['MASK']

        # txt2img = txt_db.txt2img
        self.img_db = img_db
        self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                      for tl, id_ in zip(self.txt_lens, self.ids)]

    def get_orig_datasize(self):
        return self.orig_data_size

    def combine_inputs(self, *inputs):
        input_ids = [self.cls_]
        for ids in inputs:
            input_ids.extend(ids + [self.sep])
        return torch.tensor(input_ids)

    def __getitem__(self, i):
        qid = self.ids[i]
        example = self.examples[i]  # DetectFeatTxtTokDataset.__getitem__(self, i)
        imf_fname = 'nlvr2_' + example['img_fname'].split('.')[0] + '.npz'
        img_feat, img_pos_feat, num_bb = self._get_img_feat(imf_fname)

        # text input
        input_ids = example['input_ids']
        input_ids = self.combine_inputs(input_ids)

        if 'target' in example:
            target = _get_vqa_target(example, self.num_answers)
        else:
            target = None

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return qid, input_ids, img_feat, img_pos_feat, attn_masks, target

    def __len__(self):
        return len(self.ids)


class HackyVqaOODEvalDataset(VqaEvalDataset):
    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(num_answers, *args, **kwargs)

        # assert isinstance(txt_db, TxtTokLmdb)
        # assert isinstance(img_db, DetectFeatLmdb)
        # self.txt_db = txt_db
        # self.img_db = img_db
        # txt_lens, self.ids = get_ids_and_lens(txt_db)

        # FIXME:
        self.ids = self.ids[:15000]
        txt_lens = self.txt_lens[:15000]

        txt2img = self.txt_db.txt2img
        self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(txt_lens, self.ids)]

        self.num_answers = num_answers

        # Hack to make the dataset small for now

def vqa_eval_collate(inputs):
    (qids, input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)
    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    if targets[0] is None:
        targets = None
    else:
        targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'qids': qids,
             'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch
