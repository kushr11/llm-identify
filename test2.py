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

import os
import random
import argparse
from argparse import Namespace
from pprint import pprint
from functools import partial
from pandas.core.frame import DataFrame 
import pandas as pd
import numpy  # for gradio hot reload
import gradio as gr
import time
import torch
import sys
import csv
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          LogitsProcessorList)

from watermark_processor import WatermarkLogitsProcessor, WatermarkDetector_with_preferance, \
    WatermarkLogitsProcessor_with_preferance
from utils import *


def str2bool(v):
    """Util function for user friendly boolean flag args"""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    """Command line argument specification"""

    parser = argparse.ArgumentParser(
        description="A minimum working example of applying the watermark to any LLM that supports the huggingface 🤗 `generate` API")

    parser.add_argument(
        "--user_dist",
        type=str,
        default="sparse",
        help="sparse or dense",
    )
    parser.add_argument(
        "--demo_public",
        type=str2bool,
        default=False,
        help="Whether to expose the gradio demo to the internet.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="facebook/opt-1.3b",
        help="Main model, path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--identify_mode",
        type=str,
        default="single",
        help="group or single.",
    )
    parser.add_argument(
        "--prompt_max_length",
        type=int,
        default=None,
        help="Truncation length for prompt, overrides model config's max length field.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,  # 200
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--generation_seed",
        type=int,
        default=123,
        help="Seed for setting the torch global rng prior to generation.",
    )
    parser.add_argument(
        "--use_sampling",
        type=str2bool,
        default=False, #True
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=1, #0.7
        help="Sampling temperature to use when generating using multinomial sampling.",
    )
    parser.add_argument(
        "--n_beams",
        type=int,
        default=1,
        help="Number of beams to use for beam search. 1 is normal greedy decoding",
    )
    parser.add_argument(
        "--use_gpu",
        type=str2bool,
        default=True,
        help="Whether to run inference and watermark hashing/seeding/permutation on gpu.",
    )
    parser.add_argument(
        "--seeding_scheme",
        type=str,
        default="simple_1",
        help="Seeding scheme to use to generate the greenlists at each generation and verification step.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="The fraction of the vocabulary to partition into the greenlist at each generation and verification step.",
    )
    parser.add_argument(  # change
        "--delta",
        type=float,
        default=6,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--normalizers",
        type=str,
        default="",
        help="Single or comma separated list of the preprocessors/normalizer names to use when performing watermark detection.",
    )
    parser.add_argument(
        "--ignore_repeated_bigrams",
        type=str2bool,
        default=False,
        help="Whether to use the detection method that only counts each unqiue bigram once as either a green or red hit.",
    )
    parser.add_argument(
        "--detection_z_threshold",
        type=float,
        default=4.0,
        help="The test statistic threshold for the detection hypothesis test.",
    )
    parser.add_argument(
        "--select_green_tokens",
        type=str2bool,
        default=True,
        help="How to treat the permuation when selecting the greenlist tokens at each step. Legacy is (False) to pick the complement/reds first.",
    )
    parser.add_argument(
        "--skip_model_load",
        type=str2bool,
        default=False,
        help="Skip the model loading to debug the interface.",
    )
    parser.add_argument(
        "--seed_separately",
        type=str2bool,
        default=True,
        help="Whether to call the torch seed function before both the unwatermarked and watermarked generate calls.",
    )
    parser.add_argument(
        "--load_fp16",
        type=str2bool,
        default=False,
        help="Whether to run model in float16 precsion.",
    )

    args = parser.parse_args()
    return args


def load_model(args):
    """Load and return the model and tokenizer"""

    args.is_seq2seq_model = any([(model_type in args.model_name_or_path) for model_type in ["t5", "T0"]])
    args.is_decoder_only_model = any(
        [(model_type in args.model_name_or_path) for model_type in ["gpt", "opt", "bloom"]])
    if args.is_seq2seq_model:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path)
    elif args.is_decoder_only_model:
        if args.load_fp16:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float16,
                                                         device_map='auto')
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    else:
        raise ValueError(f"Unknown model type: {args.model_name_or_path}")

    if args.use_gpu:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if args.load_fp16:
            pass
        else:
            model = model.to(device)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    return model, tokenizer, device


def generate(prompt, args, model=None, device=None, tokenizer=None, userid=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    # print(f"Generating with {args}")
    # user_id="1001001"

    watermark_processor = WatermarkLogitsProcessor_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                                   gamma=args.gamma,
                                                                   delta=args.delta,
                                                                   seeding_scheme=args.seeding_scheme,
                                                                   select_green_tokens=args.select_green_tokens,
                                                                   userid=userid
                                                                   )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens)

    if args.use_sampling:
        gen_kwargs.update(dict(
            do_sample=True,
            top_k=0,
            temperature=args.sampling_temp
        ))
    else:
        gen_kwargs.update(dict(
            num_beams=args.n_beams
        ))
    gen_kwargs.update(dict(
        # temperature=args.temperature, 
        repetition_penalty=1.0,
        # num_return_sequences=1
    ))
    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        # output_scores=True,
        return_dict_in_generate=True, 
        output_scores=True,
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config, "max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings - args.max_new_tokens
    else:
        args.prompt_max_length = 2048 - args.max_new_tokens


    prompt2=""
    for p in prompt:
        prompt2+=p
    tokd_input = tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)

    tokd_input2 = tokenizer.encode(prompt2, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=args.prompt_max_length).to(device)

    # truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    # redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    # print(redecoded_input)
    # sys.exit()

    torch.manual_seed(args.generation_seed)
    # output_without_watermark = generate_without_watermark(**tokd_input)
    # print(output_without_watermark.shape)
    # sys.exit()

    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    

    special_tokens = tokenizer.all_special_ids

    out=generate_with_watermark(tokd_input2)                                                                                               
    out_se = out[0][:, tokd_input["input_ids"].shape[-1]:]
    out_max_logit=np.zeros(200)
    out_max_idx=np.zeros(200) # out score.max() - logits.max()=0
    for k in range(len(out[1])):
        out_max_idx[k]=out[1][k].argmax().cpu()
        out_max_logit[k]=out[1][k].max().cpu()
    # print("gap between sequence and out.score.argmax()",(out_max_idx-out_se[0].cpu().numpy()).mean())
    # print("gap between out.score.max and logit.max()",(out_max_logit-watermark_processor.max_logit).mean())
    # logits = model(**tokd_input)[0]

    output_with_watermark = out[0]
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:] #取input_len后的
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:]
    
    
    redecoded_output = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    reencoded_output = tokenizer.encode(redecoded_output, return_tensors='pt',add_special_tokens=False)
    print(reencoded_output.shape,out_se.shape)
    # sys.exit()

    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    # print(tokd_input["input_ids"].shape[-1])
    return (tokd_input["input_ids"].shape[-1],
            output_with_watermark.shape[-1],
            redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            watermark_processor,
            args)
    # decoded_output_with_watermark)


def format_names(s):
    """Format names for the gradio demo interface"""
    s = s.replace("num_tokens_scored", "Tokens Counted (T)")
    s = s.replace("num_green_tokens", "# Tokens in Greenlist")
    s = s.replace("green_fraction", "Fraction of T in Greenlist")
    s = s.replace("z_score", "z-score")
    s = s.replace("p_value", "p value")
    s = s.replace("prediction", "Prediction")
    s = s.replace("confidence", "Confidence")
    return s


def list_format_scores(score_dict, detection_threshold):
    """Format the detection metrics into a gradio dataframe input format"""
    lst_2d = []
    # lst_2d.append(["z-score threshold", f"{detection_threshold}"])
    for k, v in score_dict.items():
        if k == 'green_fraction':
            lst_2d.append([format_names(k), f"{v:.1%}"])
        elif k == 'confidence':
            lst_2d.append([format_names(k), f"{v:.3%}"])
        elif isinstance(v, float):
            lst_2d.append([format_names(k), f"{v:.3g}"])
        elif isinstance(v, bool):
            lst_2d.append([format_names(k), ("Watermarked" if v else "Human/Unwatermarked")])
        else:
            lst_2d.append([format_names(k), f"{v}"])
    if "confidence" in score_dict:
        lst_2d.insert(-2, ["z-score Threshold", f"{detection_threshold}"])
    else:
        lst_2d.insert(-1, ["z-score Threshold", f"{detection_threshold}"])
    return lst_2d


def detect(input_text, args, device=None, tokenizer=None, userid=None):
    """Instantiate the WatermarkDetection object and call detect on
        the input text returning the scores and outcome of the test"""
    watermark_detector = WatermarkDetector_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                           gamma=args.gamma,
                                                           seeding_scheme=args.seeding_scheme,
                                                           device=device,
                                                           tokenizer=tokenizer,
                                                           z_threshold=args.detection_z_threshold,
                                                           normalizers=args.normalizers,
                                                           ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                                           select_green_tokens=args.select_green_tokens,
                                                           userid=userid)
    if len(input_text) - 1 > watermark_detector.min_prefix_len:
        score_dict, confidence, mark = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
    return output, confidence, mark,watermark_detector, args




def main(args):
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    # print(args)
    # gen_usr_list_sparse()
    usr_list = read_usr_list(args.user_dist)
    model, tokenizer, device = load_model(args)

    # Generate and detect, report to stdout
    if args.user_dist =='sparse':
        exp_num=32
    else:
        exp_num=50
    succ_num=0
    for i in range(exp_num):
        print(f"{i}th exp: ")
        if args.user_dist =='dense':
            gen_id=i
            # gen_id=random.randint(0, 127)
        else:
            # gen_id=random.randint(0, 31)
            gen_id=i
        # gen_id=127
        # gen_id=2
        userid = usr_list[gen_id]
        userid = usr_list[3]
        
        # sys.exit()
        # if not args.skip_model_load:
        input_text = (
            "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
            "species of turtle native to the brackish coastal tidal marshes of the "
            "Northeastern and southern United States, and in Bermuda.[6] It belongs "
            "to the monotypic genus Malaclemys. It has one of the largest ranges of "
            "all turtles in North America, stretching as far south as the Florida Keys "
            "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
            "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
            "British English and American English. The name originally was used by "
            "early European settlers in North America to describe these brackish-water "
            "turtles that inhabited neither freshwater habitats nor the sea. It retains "
            "this primary meaning in American English.[8] In British English, however, "
            "other semi-aquatic turtle species, such as the red-eared slider, might "
            "also be called terrapins. The common name refers to the diamond pattern "
            "on top of its shell (carapace), but the overall pattern and coloration "
            "vary greatly. The shell is usually wider at the back than in the front, "
            "and from above it appears wedge-shaped. The shell coloring can vary "
            "from brown to grey, and its body color can be grey, brown, yellow, "
            "or white. All have a unique pattern of wiggly, black markings or spots "
            "on their body and head. The diamondback terrapin has large webbed "
            "feet.[9] The species is"
            # "Hello! my name is Haggle."
            # "How's your day?"
        )

        args.default_prompt = input_text

        term_width = 80
        # print("#"*term_width)
        # print("Prompt:")
        # print(input_text)
        print()
        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid)



        # loop_usr_id = userid
        # with_watermark_detection_result, confidence, mark,watermark_detector, _ = detect(decoded_output_with_watermark,
        #                                                                 args,
        #                                                                 device=device,
        #                                                                 tokenizer=tokenizer,
        #                                                          userid=loop_usr_id)
        # # res=0
        # # for k in range(len(watermark_detector.green_list)):
        # #     wp=watermark_processor.green_list[k]
        # #     wd=watermark_detector.green_list[k]
        # #     res+=((wp.cpu().numpy()-wd.cpu().numpy())**2).mean()
        # # print("green list gap:",res)
        # print(confidence)
        # sys.exit()


        # start exp
        sim_list = []
        max_sim = 0
        max_sim_idx = -1
        for j in range(usr_list.shape[-1]):

            loop_usr_id = usr_list[j]
            with_watermark_detection_result, confidence, mark,watermark_detector, _ = detect(decoded_output_with_watermark,
                                                                            args,
                                                                            device=device,
                                                                            tokenizer=tokenizer,
                                                                            userid=loop_usr_id)
            # sim = compute_similarity(mark, loop_usr_id)
            sim_list.append((confidence))
            if confidence> max_sim:
                max_sim = confidence
                max_sim_idx = j
        max_number = []
        max_index = []
        mapped_sim=sim_list[gen_id]
        if args.identify_mode == "group":
            detect_range=3
        else:
            detect_range=10
        
        # sim_result=np.zeros([2,detect_range])
        sim_result=[]
        id_result=[]
        for r in range(detect_range):
            sim = max(sim_list)
            index = sim_list.index(sim)
            sim_list[index] = -1
            sim_result.append(sim)
            id_result.append(usr_list[index])
            # max_number.append(number)
            # max_index.append(usr_list[index])
        result_dic={"sim_score":sim_result, "id":id_result}
        if_succ=0
        if args.identify_mode == "group":
            if gen_id in max_index:
                succ_num += 1
                if_succ=1
        else:
            if max_sim_idx == gen_id:
                succ_num+=1
                if_succ=1

        pd.get_option('display.width')
        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', None)
        print(f"exp  {i}, if succ: {if_succ} ,time used: {time.time() - start_time}")
        print(f"gen id {userid}, mapped sim {mapped_sim}")
        print(DataFrame(result_dic).T)
        print(succ_num,exp_num)
        with open(f"data_{args.identify_mode}_{args.user_dist}_result_m{args.model_name_or_path.split('/')[1]}_d{args.delta}_g{args.max_new_tokens}.txt", "a+") as f:
            print(f"exp  {i}, if succ: {if_succ} ,time used: {time.time() - start_time}",file=f)
            print(f"gen id {userid}, mapped sim {mapped_sim}",file=f)
            print(DataFrame(result_dic).T,file=f)
            print(f"succ rate: {succ_num}/{exp_num}\n",file=f)
            # f.write(f"{time.time() - start_time}   exp  {i}: {if_succ}\n")
            # f.write(f"max sim: {max_sim}, mapped sim: {mapped_sim}")
            # f.write(str(max_number) + '\n')
            # f.write(str(max_index) + '\n\n')
            print(f"data saved in {args.identify_mode}_{args.user_dist}_result_m{args.model_name_or_path.split('/')[1]}_d{args.delta}_g{args.max_new_tokens}.txt")
            f.close()
        


            # comment ppl here
            # sys.path.append("/ssddata1/user03/identification/lm-watermarking/experiments")
            # import watermark
            # from watermark import evaluate_generation_fluency_from_output
            # from datasets import load_dataset, Dataset
            # from io_utils import write_jsonlines, write_json, read_jsonlines, read_json

            # if model is not None:
            #     model = model.to(torch.device("cpu"))
            # del model

            # oracle_model_name = 'facebook/opt-1.3b'
            # print(f"Loading oracle model: {oracle_model_name}")

            # oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
            # oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to(device)
            # oracle_model.eval()

            # input_p_output = f"{input_text}{decoded_output_without_watermark}"
            # baseline_output = decoded_output_without_watermark

            # loss_no_wm, ppl_no_wm = evaluate_generation_fluency_from_output(input_p_output,
            #                                                                 baseline_output,
            #                                                                 # idx: int,
            #                                                                 oracle_model_name,
            #                                                                 oracle_model,
            #                                                                 oracle_tokenizer)

            # input_p_output = f"{input_text}{decoded_output_with_watermark}"
            # baseline_output = decoded_output_with_watermark

            # # construct fluency/ppl partial
            # loss_wm, ppl_wm = evaluate_generation_fluency_from_output(input_p_output,
            #                                                           baseline_output,
            #                                                           # idx: int,
            #                                                           oracle_model_name,
            #                                                           oracle_model,
            #                                                           oracle_tokenizer)

            # with_watermark_detection_result, confidence, _ = detect(decoded_output_with_watermark,
            #                                                         args,
            #                                                         device=device,
            #                                                         tokenizer=tokenizer)
            # with open("result.csv", "a+") as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow([input_token_num, output_token_num, loss_no_wm, ppl_no_wm, loss_wm, ppl_wm, confidence])

    return

def testppl(args):
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    usr_list = read_usr_list(args.user_dist)


    model, tokenizer, device = load_model(args)

    # Generate and detect, report to stdout
    exp_num=10
    succ_num=0
    result_wm=np.zeros(10)
    result_bl=np.zeros(10)
    for i in range(exp_num):
        print(f"{i+1}th exp: ")
        if args.user_dist =='dense':
            gen_id=i*10
            # gen_id=random.randint(0, 127)
        else:
            # gen_id=random.randint(0, 31)
            gen_id=i
        # gen_id=127
        # gen_id=2
        userid = usr_list[gen_id]
        
        # sys.exit()
        # if not args.skip_model_load:
        input_text = (
            "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
            "species of turtle native to the brackish coastal tidal marshes of the "
            "Northeastern and southern United States, and in Bermuda.[6] It belongs "
            "to the monotypic genus Malaclemys. It has one of the largest ranges of "
            "all turtles in North America, stretching as far south as the Florida Keys "
            "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
            "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
            "British English and American English. The name originally was used by "
            "early European settlers in North America to describe these brackish-water "
            "turtles that inhabited neither freshwater habitats nor the sea. It retains "
            "this primary meaning in American English.[8] In British English, however, "
            "other semi-aquatic turtle species, such as the red-eared slider, might "
            "also be called terrapins. The common name refers to the diamond pattern "
            "on top of its shell (carapace), but the overall pattern and coloration "
            "vary greatly. The shell is usually wider at the back than in the front, "
            "and from above it appears wedge-shaped. The shell coloring can vary "
            "from brown to grey, and its body color can be grey, brown, yellow, "
            "or white. All have a unique pattern of wiggly, black markings or spots "
            "on their body and head. The diamondback terrapin has large webbed "
            "feet.[9] The species is"
            # "Hello! my name is Haggle."
            # "How's your day?"
        )

        args.default_prompt = input_text

        term_width = 80

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid)
        
        # ppl evaluation
        sys.path.append("/ssddata1/user03/identification/lm-watermarking/experiments")
        import watermark
        from watermark import evaluate_generation_fluency_from_output
        # from datasets import load_dataset, Dataset
        # from io_utils import write_jsonlines, write_json, read_jsonlines, read_json


        oracle_model_name = 'facebook/opt-1.3b'
        print(f"Loading oracle model: {oracle_model_name}")

        oracle_tokenizer = AutoTokenizer.from_pretrained(oracle_model_name)
        oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to(device)
        oracle_model.eval()

        input_p_output_wm = f"{input_text}{decoded_output_with_watermark}"
        baseline_output_wm = decoded_output_with_watermark

        input_p_output_bl = f"{input_text}{decoded_output_without_watermark}"
        baseline_output_bl = decoded_output_without_watermark

        loss_wm, ppl_wm = evaluate_generation_fluency_from_output(input_p_output_wm,
                                                                        baseline_output_wm,
                                                                        # idx: int,
                                                                        oracle_model_name,
                                                                        oracle_model,
                                                                        oracle_tokenizer)

        loss_bl, ppl_bl = evaluate_generation_fluency_from_output(input_p_output_bl,
                                                                        baseline_output_bl,
                                                                        # idx: int,
                                                                        oracle_model_name,
                                                                        oracle_model,
                                                                        oracle_tokenizer)
        
        
        result_wm[i]=ppl_wm
        result_bl[i]=ppl_bl
        print(ppl_bl,ppl_wm ,args.delta,args.max_new_tokens)
    
        # sys.exit()
    print("wm: ", result_wm.mean(),"baseline: ",result_bl.mean(),args.delta)
if __name__ == "__main__":
    args = parse_args()
    # print(args)

    main(args)
    # testppl(args)