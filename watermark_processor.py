# coding=utf-8
# Copyright 2023 Authors of "A Watermark for Large Language Models"
# available at https://arxiv.org/abs/2301.10226
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import collections
from math import sqrt

import numpy as np
import scipy.stats

import torch
from torch import Tensor
from tokenizers import Tokenizer
from transformers import LogitsProcessor

from nltk.util import ngrams

from normalizers import normalization_strategy_lookup
import copy
import sys
import math
import torch.nn.functional as F
import ipdb
class WatermarkBase:
    def __init__(
            self,
            vocab: list[int] = None,
            gamma: float = 0.5,
            decrease_delta: bool = True,
            delta: float = 2.0,
            wm_mode = "combination",
            seeding_scheme: str = "simple_1",  # mostly unused/always default
            hash_key: int = 15485863,  # just a large prime number to create a rng seed with sufficient bit width
            select_green_tokens: bool = True,
            userid="10000100",
            args=None,
    ):

        # watermarking parameters
        self.wm_mode=wm_mode
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.gamma = gamma
        self.decrease_delta = decrease_delta
        self.delta = delta
        self.seeding_scheme = seeding_scheme
        self.rng = None
        self.hash_key = hash_key
        self.select_green_tokens = select_green_tokens
        self.idx_t = 0
        self.userid = userid
        self.hit = 0
        self.args=args
        # self.max_logit = np.zeros(200)
        # self.green_list=[]

    def _seed_rng(self, input_ids: torch.LongTensor, seeding_scheme: str = None) -> None:
        # can optionally override the seeding scheme,
        # but uses the instance attr by default
        if seeding_scheme is None:
            seeding_scheme = self.seeding_scheme

        if seeding_scheme == "simple_1":
            assert input_ids.shape[
                       -1] >= 1, f"seeding_scheme={seeding_scheme} requires at least a 1 token prefix sequence to seed rng"
            prev_token = input_ids[-1].item()
            # print("prev token: ",prev_token)
            self.rng.manual_seed(self.hash_key * prev_token)
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {seeding_scheme}")
        return
    def _seed_depth_rng(self) -> None:
        self.rng.manual_seed(self.hash_key * int(self.userid,2))
        return


    def _get_greenlist_ids(self, input_ids: torch.LongTensor) -> list[int]:
        # seed the rng using the previous tokens/prefix
        # according to the seeding_scheme
        self._seed_rng(input_ids)
        # print(input_ids.shape)
        # ipdb.set_trace()
        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)

        greenlist_ids = vocab_permutation[:greenlist_size]  # new
        redlist_ids = vocab_permutation[greenlist_size:]
        # ipdb.set_trace()
        self._seed_depth_rng()
        depth_permutation=torch.randperm(len(greenlist_ids), device=input_ids.device, generator=self.rng)
        depth_green_ids=greenlist_ids[depth_permutation]
        if len(redlist_ids)!=len(greenlist_ids):
            depth_red_ids=redlist_ids[:-1][depth_permutation]
            #append to tail
            depth_red_ids=torch.cat((depth_red_ids,torch.tensor([redlist_ids[-1]]).to(depth_red_ids.device)))
        else:
            depth_red_ids=redlist_ids[depth_permutation]

        if self.args.gen_mode=="depth_d":
            green_d_masks=[]
            red_d_masks=[]
            discrete_depth=self.args.depth
            g_discrete_length=greenlist_size//discrete_depth
            r_discrete_length=greenlist_size//discrete_depth
            for i in range(discrete_depth):
                if i == discrete_depth-1:
                    green_d_masks.append(depth_green_ids[i*g_discrete_length:])
                    red_d_masks.append(depth_red_ids[i*r_discrete_length:])
                else:
                    green_d_masks.append(depth_green_ids[i*g_discrete_length:(i+1)*g_discrete_length])
                    red_d_masks.append(depth_red_ids[i*r_discrete_length:(i+1)*r_discrete_length])
            return greenlist_ids,redlist_ids,green_d_masks,red_d_masks
        # ipdb.set_trace()
        return greenlist_ids, redlist_ids,[],[]
        # return greenlist_ids




class WatermarkLogitsProcessor_with_preferance(WatermarkBase, LogitsProcessor):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _bias_depth_d_logits(self,scores: torch.FloatTensor,greenlist_token_ids,d_masks,delta):
        for i in range(len(d_masks)):
            delta=delta*0.5**i
            for j in range(len(greenlist_token_ids)):
                scores[j][d_masks[i]]=scores[j][d_masks[i]]+delta
        return scores
    
    def _calc_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        # TODO lets see if we can lose this loop
        green_tokens_mask = torch.zeros_like(scores)
        for b_idx in range(len(greenlist_token_ids)):
            green_tokens_mask[b_idx][greenlist_token_ids[b_idx]] = 1
        final_mask = green_tokens_mask.bool()
        return final_mask

    def _bias_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor,
                               greenlist_bias: float, decrease_delta: bool) -> torch.Tensor:
        if decrease_delta:
            greenlist_bias=4.84*(math.e)**(-1*0.001*self.idx_t)
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        # print(greenlist_bias,self.idx_t)
        return scores

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        n = len(self.userid)

        # preferance = self.userid[(self.idx_t - n * (self.idx_t // n)) % n]  # 1->green ; 0-> red
        if self.wm_mode =='previous1':
            if len(self.userid[-1]) < 2:
                preferance = 1      # if the sentence is too short, we assign the pref to green by default
            else:
                preferance = self.userid[input_ids[-1][-1] % n]  # 1->green ; 0-> red
        else:
            if len(self.userid[-1]) < 2:
                preferance = 1      # if the sentence is too short, we assign the pref to green by default
            else:
                preferance = self.userid[(input_ids[-1][-1]*input_ids[-1][-2]) % n]  # 1->green ; 0-> red
        self.idx_t += 1
        # this is lazy to allow us to colocate on the watermarked model's device
        if self.rng is None:
            self.rng = torch.Generator(device=input_ids.device)

        # NOTE, it would be nice to get rid of this batch loop, but currently,
        # the seed and partition operations are not tensor/vectorized, thus
        # each sequence in the batch needs to be treated separately.
        batched_greenlist_ids = [None for _ in range(input_ids.shape[0])]
        if self.args.gen_mode=='depth_d':
            batched_d_masks = [None for _ in range(input_ids.shape[0])]

        for b_idx in range(input_ids.shape[0]):
            # d_masks only availiable in "depth_d"
            if preferance == '1':
                greenlist_ids, _ ,d_masks,_= self._get_greenlist_ids(input_ids[b_idx])
            else:
                _, greenlist_ids,_,d_masks = self._get_greenlist_ids(input_ids[b_idx])
            
            batched_greenlist_ids[b_idx] = greenlist_ids
            if self.args.gen_mode=='depth_d':
                batched_d_masks[b_idx] = d_masks

        if self.args.gen_mode !='depth_d':
            green_tokens_mask = self._calc_greenlist_mask(scores=scores, greenlist_token_ids=batched_greenlist_ids)
            scores_withnomask=copy.deepcopy(scores)
            scores = self._bias_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta,decrease_delta=self.decrease_delta)
            # ipdb.set_trace()
        else:
            scores_withnomask=copy.deepcopy(scores)
            scores=self._bias_depth_d_logits(scores=scores,greenlist_token_ids=batched_greenlist_ids,d_masks=d_masks,delta=self.delta)
            # ipdb.set_trace()
        return scores


class WatermarkDetector_with_preferance(WatermarkBase):
    def __init__(
            self,
            *args,
            device: torch.device = None,
            tokenizer: Tokenizer = None,
            z_threshold: float = 4.0,
            # normalizers: list[str] = ["unicode"],
            normalizers: list[str] = ["unicode"],  # or also: ["unicode", "homoglyphs", "truecase"]
            ignore_repeated_bigrams: bool = False,
            # userid,
            **kwargs,

    ):
        super().__init__(*args, **kwargs)
        # also configure the metrics returned/preprocessing options
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device

        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        if self.seeding_scheme == "simple_1" and self.wm_mode == 'previous1':
            self.min_prefix_len = 1
        elif self.wm_mode == 'combination':
            self.min_prefix_len = 2 
        else:
            raise NotImplementedError(f"Unexpected seeding_scheme: {self.seeding_scheme}")

        
        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(normalization_strategy_lookup(normalization_strategy))

        self.ignore_repeated_bigrams = ignore_repeated_bigrams
        if self.ignore_repeated_bigrams:
            assert self.seeding_scheme == "simple_1", "No repeated bigram credit variant assumes the single token seeding scheme."

    def _compute_z_score(self, observed_count, T):
        # count refers to number of green tokens, T is total number of tokens
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def _compute_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    def _score_sequence(
            self,
            input_ids: Tensor,
            return_num_tokens_scored: bool = True,
            return_num_green_tokens: bool = True,
            return_green_fraction: bool = True,
            return_green_token_mask: bool = False,
            return_z_score: bool = True,
            return_p_value: bool = True,
    ):
        mark = ""
        if self.ignore_repeated_bigrams:  # false
            # Method that only counts a green/red hit once per unique bigram.
            # New num total tokens scored (T) becomes the number unique bigrams.
            # We iterate over all unqiue token bigrams in the input, computing the greenlist
            # induced by the first token in each, and then checking whether the second
            # token falls in that greenlist.
            assert return_green_token_mask == False, "Can't return the green/red mask when ignoring repeats."
            bigram_table = {}
            token_bigram_generator = ngrams(input_ids.cpu().tolist(), 2)
            freq = collections.Counter(token_bigram_generator)
            num_tokens_scored = len(freq.keys())
            for idx, bigram in enumerate(freq.keys()):
                prefix = torch.tensor([bigram[0]],
                                      device=self.device)  # expects a 1-d prefix tensor on the randperm device

                greenlist_ids = self._get_greenlist_ids(prefix)
                bigram_table[bigram] = True if bigram[1] in greenlist_ids else False
            green_token_count = sum(bigram_table.values())

        else:
            num_tokens_scored = len(input_ids) - self.min_prefix_len  # 1
            if num_tokens_scored < 1:
                raise ValueError((f"Must have at least {1} token to score after "
                                  f"the first min_prefix_len={self.min_prefix_len} tokens required by the seeding scheme."))
            # Standard method.
            # Since we generally need at least 1 token (for the simplest scheme)
            # we start the iteration over the token sequence with a minimum
            # num tokens as the first prefix for the seeding scheme,
            # and at each step, compute the greenlist induced by the
            # current prefix and check if the current token falls in the greenlist.
            green_token_count, green_token_mask = 0, []
            depth_hit=torch.zeros(self.args.depth)
            # print("in detecter,userid-", self.userid)
            for idx in range(self.min_prefix_len, len(input_ids)):
                # if idx == self.min_prefix_len:
                    # print(input_ids[idx])
                curr_token = input_ids[idx]
                n = len(self.userid)
                if self.wm_mode == 'previous1':
                    preferance = self.userid[input_ids[idx-1] % n]  # 1->green ; 0-> red
                else:
                    preferance = self.userid[(input_ids[idx-1]*input_ids[idx-2]) % n]  # 1->green ; 0-> red
                # preferance = self.userid[(idx - n * (idx // n)) % n]  # 1->green ; 0-> red
                # print(preferance,end="")
                if preferance == '1':
                    greenlist_ids, _, d_masks,_ = self._get_greenlist_ids(input_ids[:idx])
                else:
                    _, greenlist_ids, _,d_masks = self._get_greenlist_ids(input_ids[:idx])

                if curr_token in greenlist_ids:
                    if preferance == '1':
                        mark += '1'
                    else:
                        mark += '0'
                    green_token_count += 1
                    green_token_mask.append(True)
                    if self.args.gen_mode=="depth_d":
                        for j in range(len(d_masks)):
                            if curr_token in d_masks[j]:
                                depth_hit[j]+=1
                else:
                    if preferance == '0':
                        mark += '1'
                    else:
                        mark += '0'
                    green_token_mask.append(False)

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        # print(green_token_count / num_tokens_scored)
        if return_z_score:
            score_dict.update(dict(z_score=self._compute_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self._compute_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask))
        sim_score=green_token_count / num_tokens_scored
        gr_sim_score=np.array(sim_score)
        depth_loss=0
        if self.args.gen_mode=="depth_d":
            depth_pd=depth_hit/green_token_count
            standard_depth_distribution=torch.tensor([0.6834,0.1800,0.1366])
            standard_gr=0.7959
            
            loss_func = torch.nn.CrossEntropyLoss()
            depth_loss = -loss_func(depth_pd, standard_depth_distribution)
            total_loss = -loss_func(depth_pd*sim_score, standard_depth_distribution*standard_gr)
            # kl_divergence = -F.kl_div(depth_pd.log(), depth_distribution_score, reduction='mean')
            # sim_score=kl_divergence
            
            gr_sim_score=np.array(sim_score)
            depth_loss=np.array(depth_loss)
            total_loss=np.array(total_loss)
            # print(green_token_count , num_tokens_scored,depth_hit,depth_pd,celoss)
            # sim_score+=depth_distribution_score
        return score_dict, gr_sim_score,total_loss, mark
        # return score_dict, gr_sim_score,depth_loss, mark

    def detect(
            self,
            text: str = None,
            tokenized_text: list[int] = None,
            return_prediction: bool = True,
            return_scores: bool = True,
            z_threshold: float = None,
            **kwargs,
    ) -> dict:

        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True  # to return the "confidence":=1-p of positive detections

        # run optional normalizers on text
        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(
                self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            # try to remove the bos_tok at beginning if it's there
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                # print("here")
                tokenized_text = tokenized_text[1:]
        # print("in detect, tokenized[3],[4]:", tokenized_text[3], tokenized_text[4])
        # call score method
        output_dict = {}
        # print("in _tokenized:", tokenized_text.shape)
        score_dict, gr_score,depth_score, mark = self._score_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        # if passed return_prediction then perform the hypothesis test and return the outcome
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        return output_dict, gr_score,depth_score, mark

