# -*- ecoding: utf-8 -*-
# @ModuleName: model_method
# @Function: 
# @Author: long
# @Time: 2021/11/15 20:57
# *****************conding***************
import csv
import torch
from torch import nn
import pickle
from transformers import BertModel, BertTokenizer
import sys
import numpy as np
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args = {
    "max_object_len": 40,  # 平均长度 7.22, 最大长度93
    "max_subject_len": 40,  # 平均长度10.0, 最大长度138
    "max_time_len": 20,  # 平均长度6.03, 最大长度22
    "max_location_len": 25,  # 平均长度3.79,最大长度41
    "max_trigger_len": 6
}


class DomTrigger(torch.nn.Module):
    def __init__(self, pre_train_dir: str):
        """
        :param pre_train_dir: 预训练RoBERTa或者BERT文件夹
        """
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=768),
            torch.nn.Tanh()
        )
        self.start_layer = torch.nn.Linear(in_features=768, out_features=2)
        self.end_layer = torch.nn.Linear(in_features=768, out_features=2)
        self.span1_layer = torch.nn.Linear(in_features=768, out_features=1, bias=False)
        self.span2_layer = torch.nn.Linear(in_features=768, out_features=1,
                                           bias=False)  # span1和span2是span_layer的拆解, 减少计算时的显存占用
        self.epsilon = 1e-6
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, input_mask, input_seg, span_mask):
        bsz, seq_len = input_ids.size()[0], input_ids.size()[1]
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[
            0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep)  # (bsz, seq, 2)
        end_logits = self.end_layer(encoder_rep)  # (bsz, seq, 2)
        span1_logits = self.span1_layer(encoder_rep)  # (bsz, seq, 1)
        span2_logits = self.span2_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # 将两个span组合 => (bsz, seq, seq)
        span_logits = span1_logits.repeat(1, 1, seq_len) + span2_logits[:, None, :].repeat(1, seq_len, 1)
        start_prob_seq = self.softmax(start_logits)  # , dim=-1)  # (bsz, seq, 2)
        end_prob_seq = self.softmax(end_logits)  # , dim=-1)  # (bsz, seq, 2)
        # 使用span_mask
        span_logits.masked_fill_(span_mask == 0, -1e30)
        span_prob = self.softmax(span_logits)  # , dim=-1)  # (bsz, seq, seq)
        return start_prob_seq, end_prob_seq, span_prob


class AuxTrigger(torch.nn.Module):
    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=768),
            torch.nn.Tanh(),
        )
        self.start_layer = torch.nn.Linear(in_features=768, out_features=1)
        self.end_layer = torch.nn.Linear(in_features=768, out_features=1)
        self.epsilon = 1e-6
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, input_mask, input_seg):
        encoder_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=input_seg)[
            0]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        start_logits = self.start_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        mask = input_mask == 0.0
        start_logits.masked_fill_(mask, -1e30)
        end_logits.masked_fill_(mask, -1e30)
        start_prob_seq = self.softmax(start_logits)  # , dim=1)
        end_prob_seq = self.softmax(end_logits)  # , dim=1)
        return start_prob_seq, end_prob_seq


class Argument(torch.nn.Module):
    def __init__(self, pre_train_dir: str):
        super().__init__()
        self.roberta_encoder = BertModel.from_pretrained(pre_train_dir)
        self.encoder_linear = torch.nn.Sequential(
            torch.nn.Linear(in_features=768, out_features=768),
            torch.nn.Tanh(),
        )
        self.cls_layer = torch.nn.Linear(in_features=768, out_features=2)
        self.start_layer = torch.nn.Linear(in_features=768, out_features=1)
        self.end_layer = torch.nn.Linear(in_features=768, out_features=1)
        self.epsilon = 1e-6
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, input_mask, input_seg):
        encoder_rep, cls_rep = self.roberta_encoder(input_ids=input_ids, attention_mask=input_mask,
                                                    token_type_ids=input_seg)[:2]  # (bsz, seq, dim)
        encoder_rep = self.encoder_linear(encoder_rep)
        cls_logits = self.cls_layer(cls_rep)
        start_logits = self.start_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        end_logits = self.end_layer(encoder_rep).squeeze(dim=-1)  # (bsz, seq)
        # adopt softmax function across length dimension with masking mechanism
        mask = input_mask == 0.0
        start_logits.masked_fill_(mask, -1e30)
        end_logits.masked_fill_(mask, -1e30)
        start_prob_seq = self.softmax(start_logits)  # , dim=1)
        end_prob_seq = self.softmax(end_logits)  # , dim=1)
        return cls_logits, start_prob_seq, end_prob_seq


class InputEncoder(object):
    def __init__(self, max_len, tokenizer, special_query_token_map: dict):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.SEG_Q = 0
        self.SEG_P = 1
        self.ID_PAD = 0
        self.map = special_query_token_map
        self.arg_map = {
            "object": "主体", "subject": "客体", "time": "时间", "location": "地点"
        }

    def trigger_enc(self, context, is_dominant: bool):  # 适合dominant和auxiliary
        query = "找出事件中的触发词"
        query_tokens = [i for i in query]
        context_tokens = [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len - 1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_start = 2 + len(query_tokens)
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        span_mask = None
        if is_dominant:
            span_mask = np.zeros(shape=(self.max_len, self.max_len), dtype=np.float32)
            for i in range(context_start, context_end):
                for j in range(i, context_end):
                    span_mask[i, j] = 1.0
        return {
            "input_ids": torch.tensor(input_ids).long().unsqueeze(dim=0).to(device=device),
            "input_seg": torch.tensor(input_seg).long().unsqueeze(dim=0).to(device=device),
            "input_mask": torch.tensor(input_mask).float().unsqueeze(dim=0).to(device=device),
            "context": context,
            "span_mask": torch.from_numpy(span_mask).float().unsqueeze(dim=0).to(device=device),
            "context_range": "%d-%d" % (context_start, context_end)  # 防止被转化成tensor
        }

    def argument_enc(self, context, trigger, start, end, arg):
        query = "处于位置&%d&和位置-%d-之间的触发词*%s*的%s为?" % (start, end, trigger, self.arg_map[arg])
        query_tokens = []
        for i in query:
            if i in self.map.keys():
                query_tokens.append(self.map[i])
            else:
                query_tokens.append(i)
        context_tokens = [i for i in context]
        c = ["[CLS]"] + query_tokens + ["[SEP]"] + context_tokens
        if len(c) > self.max_len - 1:
            c = c[:self.max_len - 1]
        c += ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(c)
        input_mask = [1.0] * len(input_ids)
        input_seg = [self.SEG_Q] * (len(query_tokens) + 2) + [self.SEG_P] * (len(input_ids) - 2 - len(query_tokens))
        context_end = len(input_ids) - 1
        extra = self.max_len - len(input_ids)
        if extra > 0:
            input_ids += [self.ID_PAD] * extra
            input_mask += [0.0] * extra
            input_seg += [self.SEG_P] * extra
        return {
            "input_ids": torch.tensor(input_ids).long().unsqueeze(dim=0).to(device),
            "input_seg": torch.tensor(input_seg).long().unsqueeze(dim=0).to(device),
            "input_mask": torch.tensor(input_mask).float().unsqueeze(dim=0).to(device),
            "context": context, "context_range": "%d-%d" % (2 + len(query_tokens), context_end)
        }


class OutputDecoder(object):
    @staticmethod
    def dominant_dec(context, s_seq, e_seq, p_seq, context_range, n_triggers):
        ans_index = []
        t = context_range.split("-")
        c_start, c_end = int(t[0]), int(t[1])
        # 先找出所有被判别为开始和结束的位置索引
        i_start, i_end = [], []
        for i in range(c_start, c_end):
            if s_seq[i][1] > s_seq[i][0]:
                i_start.append(i)
            if e_seq[i][1] > e_seq[i][0]:
                i_end.append(i)
        # 然后遍历i_end
        cur_end = -1
        for e in i_end:
            s = []
            for i in i_start:
                if e >= i >= cur_end and (e - i) <= args["max_trigger_len"]:
                    s.append(i)
            max_s = 0.0
            t = None
            for i in s:
                if p_seq[i, e] > max_s:
                    t = (i, e)
                    max_s = p_seq[i, e]
            cur_end = e
            if t is not None:
                ans_index.append(t)
        out = []
        for item in ans_index:
            out.append({
                "answer": context[item[0] - c_start:item[1] - c_start + 1], "start": item[0] - c_start,
                "end": item[1] - c_start + 1,
                "score": ((s_seq[item[0]][1] + e_seq[item[1]][1]) / 2) * p_seq[item[0], item[1]]
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:n_triggers]

    @staticmethod
    def auxiliary_dec(context, s_seq, e_seq, context_range):
        max_score = 0.0
        dic = {"start": -1, "end": -1}
        t = context_range.split("-")
        start, end = int(t[0]), int(t[1])
        for i in range(start, end):
            for j in range(i, i + args["max_time_len"] if i + args["max_trigger_len"] <= end else end):
                if s_seq[i] + e_seq[j] > max_score:
                    max_score = s_seq[i] + e_seq[j]
                    dic["start"], dic["end"] = i, j
        return [{"answer": context[dic["start"] - start:dic["end"] - start + 1], "start": dic["start"] - start,
                 "end": dic["end"] - start + 1}]

    @staticmethod
    def argument_dec(context, cls, s_seq, e_seq, context_range, arg_type):
        if cls[1] > cls[0]:
            max_score = 0.0
            dic = {"start": -1, "end": -1}
            t = context_range.split("-")
            start, end = int(t[0]), int(t[1])
            for i in range(start, end):
                for j in range(i,
                               i + args["max_%s_len" % arg_type] if i + args["max_%s_len" % arg_type] <= end else end):
                    if s_seq[i] + e_seq[j] > max_score:
                        max_score = s_seq[i] + e_seq[j]
                        dic["start"], dic["end"] = i, j
            return context[dic["start"] - start:dic["end"] - start + 1]
        else:
            return ""