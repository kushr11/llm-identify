# -*- coding: utf-8 -*-
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
                          OPTForCausalLM,
                          LogitsProcessorList)

from watermark_processor import  WatermarkDetector_with_preferance, \
    WatermarkLogitsProcessor_with_preferance
from utils import *
# from datasets import load_dataset, Dataset
from datasets import load_dataset
# from torch.nn.parallel import DataParallel
import kaggle_ds
from pubmedqa import Dataset


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
        "--ppl",
        type=str2bool,
        default=False,
        help="To evaluate ppl instead of run generating and detecting",
    )
    parser.add_argument(
        "--wm_mode",
        type=str,
        default="combination",
        help="previous1 or combination",
    )
    parser.add_argument(
        "--detect_mode",
        type=str,
        default="accumulate",
        help="normal or iterative or accumulate",
    )
    parser.add_argument(
        "--user_dist",
        type=str,
        default="dense",
        help="sparse or dense",
    )
    # parser.add_argument(
    #     "--identify_mode",
    #     type=str,
    #     default="single",
    #     help="group or single.",
    # )
    parser.add_argument(  
        "--delta",
        type=float,
        default=3,
        help="The amount/bias to add to each of the greenlist token logits before each token sampling step.",
    )
    parser.add_argument(
        "--decrease_delta",
        type=str2bool,
        default=False,
        help="Modify delta according to output length.",
        
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,  # 200
        help="Maximmum number of new tokens to generate.",
    )
    parser.add_argument(
        "--min_new_tokens",
        type=int,
        default=100,  
        help="Minimum number of new tokens to generate.",
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
        "--prompt_max_length",
        type=int,
        default=None, #None
        help="Truncation length for prompt, overrides model config's max length field.",
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
        default=True, 
        help="Whether to generate using multinomial sampling.",
    )
    parser.add_argument(
        "--sampling_temp",
        type=float,
        default=0.7, #0.7
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
        default=True,
        help="Whether to run model in float16 precsion.",
    )
    
    # Modification
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'c4',
        help = "Select the dataset we want to use to run tests on",
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
        # if any([(model_type in args.model_name_or_path) for model_type in ["125m", "1.3b","2.7b"]]):
        #     if args.load_fp16:
        #         pass
        #     else:
        #         model = model.to(device)
        # else:
        #     model = OPTForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto",  torch_dtype=torch.float16)
        #     # print(model.hf_device_map)
    else:
        device = "cpu"
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    return model, tokenizer, device


def generate(prompt, args, model=None, device=None, tokenizer=None, userid=None, index=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """
       
    # Gordon: check the tokenized vocab
    # print ("tokenized vocab is: ",tokenizer.get_vocab())
    watermark_processor = WatermarkLogitsProcessor_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                                   gamma=args.gamma,
                                                                   decrease_delta=args.decrease_delta,
                                                                   delta=args.delta,
                                                                   wm_mode=args.wm_mode,
                                                                   seeding_scheme=args.seeding_scheme,
                                                                   select_green_tokens=args.select_green_tokens,
                                                                   userid=userid
                                                                   )

    gen_kwargs = dict(max_new_tokens=args.max_new_tokens,min_new_tokens=args.min_new_tokens)

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

    generate_without_watermark = partial(
        model.generate,
        **gen_kwargs
    )
    generate_with_watermark = partial(
        model.generate,
        logits_processor=LogitsProcessorList([watermark_processor]),
        return_dict_in_generate=True, 
        output_scores=True,
        **gen_kwargs
    )
    if args.prompt_max_length:
        pass
    elif hasattr(model.config,"max_position_embedding"):
        args.prompt_max_length = model.config.max_position_embeddings-args.max_new_tokens
    else:
        args.prompt_max_length = 2048-args.max_new_tokens

    tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, 
                           max_length=args.prompt_max_length).to(device)
    

    truncation_warning = True if tokd_input["input_ids"].shape[-1] == args.prompt_max_length else False
    redecoded_input = tokenizer.batch_decode(tokd_input["input_ids"], skip_special_tokens=True)[0]
    # print(redecoded_input)
    # sys.exit()

    torch.manual_seed(args.generation_seed)
    output_without_watermark = generate_without_watermark(**tokd_input)


    # optional to seed before second generation, but will not be the same again generally, unless delta==0.0, no-op watermark
    if args.seed_separately:
        torch.manual_seed(args.generation_seed)
    
    # generate_with_watermark = partial(
    #     model.generate,
    #     logits_processor=LogitsProcessorList([watermark_processor]),
    #     # output_scores=True,
    #     **gen_kwargs

    #just for test
    # generate_logit_with_watermark = partial(
    #     model.generate,
    #     logits_processor=LogitsProcessorList([watermark_processor]),
    #     return_dict_in_generate=True, 
    #     output_scores=True,
    #     **gen_kwargs
    # )

    out=generate_with_watermark(**tokd_input)                                                                                               
    out_se = out[0][:, tokd_input["input_ids"].shape[-1]:]
    logit=out[1]
    logit=torch.stack(logit)
    torch.save(logit,f"./assest/clean_z_200/clean_z_{index}.pt")
    
    # out_max_logit=np.zeros(200)
    # out_max_idx=np.zeros(200) # out score.max() - logits.max()=0
    # for k in range(len(out[1])):
    #     out_max_idx[k]=out[1][k].argmax().cpu()
    #     out_max_logit[k]=out[1][k].max().cpu()
    # print("gap between sequence and out.score.argmax()",(out_max_idx-out_se[0].cpu().numpy()).mean())
    # print("gap between out.score.max and logit.max()",(out_max_logit-watermark_processor.max_logit).mean())
    # logits = model(**tokd_input)[0]

    output_with_watermark = out[0]
    # print("tokdid:",tokd_input["input_ids"].shape)
    # print("before minus:",output_with_watermark.shape)
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:] #取input_len后的
        output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:]
    
    
    # redecoded_output = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    # reencoded_output = tokenizer.encode(redecoded_output, return_tensors='pt',add_special_tokens=False)
    # print(reencoded_output.shape,out_se.shape)
    # sys.exit()
    # print("len in gen:",(output_with_watermark.shape))
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
                                                           wm_mode=args.wm_mode,
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
        # print("debug in else")
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
    return output, confidence, mark,watermark_detector, args




def main(args):
    #load dataset
    
    # Modification
    dataset_name = args.dataset
    # dataset_config_name = 'ro-en'
    
    
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    usr_list = read_usr_list(args.user_dist)
    model, tokenizer, device = load_model(args)
    print(device)

    # Generate and detect, report to stdout
    # if args.user_dist =='sparse':
    #     exp_num=32
    # else:
    #     exp_num=50
    
    exp_num=1000
    
    total_detect_len=0 #for detect mode: iterative
    succ_num_top1=0
    succ_num_top3=0
    # for t in range(17):
    #         input_text=next(ds_iterator)
    for i in range(exp_num):
        print(f"{i}th exp: ")
        if args.user_dist =='dense':
            # gen_id=i
            gen_id=random.randint(0, 127)
        else:
            gen_id=random.randint(0, 31)
            # gen_id=i
        # gen_id=127
        # gen_id=2
        userid = usr_list[gen_id]
        # userid= usr_list[89]
        
        # input_text = (
        #     "The diamondback terrapin or simply terrapin (Malaclemys terrapin) is a "
        #     "species of turtle native to the brackish coastal tidal marshes of the "
        #     "Northeastern and southern United States, and in Bermuda.[6] It belongs "
        #     "to the monotypic genus Malaclemys. It has one of the largest ranges of "
        #     "all turtles in North America, stretching as far south as the Florida Keys "
        #     "and as far north as Cape Cod.[7] The name 'terrapin' is derived from the "
        #     "Algonquian word torope.[8] It applies to Malaclemys terrapin in both "
        #     "British English and American English. The name originally was used by "
        #     "early European settlers in North America to describe these brackish-water "
        #     "turtles that inhabited neither freshwater habitats nor the sea. It retains "
        #     "this primary meaning in American English.[8] In British English, however, "
        #     "other semi-aquatic turtle species, such as the red-eared slider, might "
        #     "also be called terrapins. The common name refers to the diamond pattern "
        #     "on top of its shell (carapace), but the overall pattern and coloration "
        #     "vary greatly. The shell is usually wider at the back than in the front, "
        #     "and from above it appears wedge-shaped. The shell coloring can vary "
        #     "from brown to grey, and its body color can be grey, brown, yellow, "
        #     "or white. All have a unique pattern of wiggly, black markings or spots "
        #     "on their body and head. The diamondback terrapin has large webbed "
        #     "feet.[9] The species is"
        #     # "Hello! my name is Haggle."
        #     # "How's your day?"
        # )
        # print("args.prompt max l",args.prompt_max_length)
        
        # two datasets. Whether the length of sentences input is larger than prompt_max_length
        if dataset_name == 'c4':
            dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
            ds_iterator = iter(dataset)
            if args.prompt_max_length is not None and len(next(ds_iterator)['text']) >= args.prompt_max_length:
                input_text = next(ds_iterator)['text'][:args.prompt_max_length]
            else:
                input_text = next(ds_iterator)['text'][:]
                
                
        # TODO: there will be 6 datasets that needs to be implemented from "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
        # dataset: xsum
        if dataset_name == 'xsum':
            dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
            ds_iterator = iter(dataset)
            if args.prompt_max_length is not None and len(next(ds_iterator)['document']) >= args.prompt_max_length:
                input_text = next(ds_iterator)['document'][:args.prompt_max_length]
            else:
                input_text = next(ds_iterator)['document'][:]
        # dataset: squad
        # dataset_config_name = plain_text
        if dataset_name == 'squad':
            dataset_config_name = plain_text
            dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
            ds_iterator = iter(dataset)
            if args.prompt_max_length is not None and len(next(ds_iterator)['context']) >= args.prompt_max_length:
                input_text = next(ds_iterator)['context'][:args.prompt_max_length]
            else:
                input_text = next(ds_iterator)['context'][:]
                
        # dataset: WMT16
        # dataset_config_name = 'ro-en'
        if dataset_name == 'wmt16':
            dataset_config_name = 'ro-en'
            dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
            ds_iterator = iter(dataset)
            if args.prompt_max_length is not None and len(next(ds_iterator)['translation']) >= args.prompt_max_length:
                input_text = tuple(next(ds_iterator)['translation'])[:args.prompt_max_length]
            else:
                input_text = tuple(next(ds_iterator)['translation'])[:]
                
        # dataset: PubMedQA
        # dataset_config_name = 'pubmed_qa_artificial_source'
        if dataset_name == 'bigbio/pubmed_qa':
            file_path = 'ori_pqal.json'
            dataset = Dataset(file_path)
            ds_iterator = iter(dataset.get_items())
            _, item_data = next(ds_iterator)
            if args.prompt_max_length is not None and len(item_data['LONG_ANSWER']) >= args.prompt_max_length:
                input_text = item_data['LONG_ANSWER'][:args.prompt_max_length]
            else:
                input_text = item_data['LONG_ANSWER'][:]
                
        # dataset: Reddit WritingPrompts dataset
        # we load from local source
        if dataset_name == 'writingprompts':
            dataset = kaggle_ds.load_data('/data1/ybaiaj/llm-identify-main-new-6.7b-25/writing-prompts')
            ds_iterator = dataset.iterrows()
            _, row_data = next(ds_iterator)
            if args.prompt_max_length is not None and len(row_data['story']) >= args.prompt_max_length:
                input_text = row_data['story'][:args.prompt_max_length]
            else:
                input_text = row_data['story']
        #
        args.default_prompt = input_text
        #print("input:",input_text[:10])

        term_width = 80
        # print("#"*term_width)
        # print("Prompt:")
        # print(input_text)

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid,
            index=i)
        
        
        continue


        # loop_usr_id = userid
        # with_watermark_detection_result, confidence, mark,watermark_detector, _ = detect(decoded_output_with_watermark,
        #                                                                 args,
        #                                                                 device=device,
        #                                                                 tokenizer=tokenizer,
        #                                                          userid=loop_usr_id)
        # res=0
        # for k in range(len(watermark_detector.green_list)):
        #     wp=watermark_processor.green_list[k]
        #     wd=watermark_detector.green_list[k]
        #     res+=((wp.cpu().numpy()-wd.cpu().numpy())**2).mean()
        # print("green list gap:",res)
        # print(confidence)
        # sys.exit()

        del(watermark_processor)
        #start detect
        
        if args.detect_mode == 'normal':
        
            sim_list = []
            max_sim = 0
            max_sim_idx = -1
            
            for j in range(usr_list.shape[-1]):

                loop_usr_id = usr_list[j]

                with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                # sim = compute_similarity(mark, loop_usr_id)
                sim_list.append((confidence))
                if confidence> max_sim:
                    max_sim = confidence
                    max_sim_idx = j
            
            # if args.identify_mode == "group":
            #     detect_range=3
            # else:
            #     detect_range=10
            detect_range=10
            mapped_sim=sim_list[gen_id]
            # sim_result=np.zeros([2,detect_range])
            sim_result=[]
            id_result=[]
            id_index=[]
            for r in range(detect_range):
                sim = max(sim_list)
                index = sim_list.index(sim)
                sim_list[index] = -1
                sim_result.append(sim)
                id_result.append(usr_list[index])
                id_index.append(index)
            result_dic={"sim_score":sim_result, "id":id_result}
            if_succ_top1=0
            if_succ_top3=0

            if gen_id in id_index[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if max_sim_idx == gen_id:
                succ_num_top1+=1
                if_succ_top1=1
        if args.detect_mode == 'iterative':
            
            max_sim=-1
            init_num=3
            sim_list=[]
            code_list=[]
            ## append initial code
            code_list.append(usr_list[0])
            for j in range (init_num-1):
                code_list.append(usr_list[(usr_list.shape[-1]//init_num)*(j+1)])
            code_list.append(usr_list[-1])
            for loop_usr_id in code_list:
                with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=loop_usr_id)
                sim_list.append((confidence))
            max_sim=max(sim_list)
            best_code = code_list[sim_list.index(max_sim)]
            # change loop_usr_id 1 by 1
            
            stop=False
            while (not stop):
                previous_max_sim=max_sim
                for j in range(len(best_code)):
                    if j==0:
                        loop_usr_id=str(1-int(best_code[j]))+best_code[j+1:]
                    elif j==len(best_code)-1:
                        loop_usr_id=best_code[:-1]+str(1-int(best_code[j]))
                    else:
                        loop_usr_id=best_code[:j]+str(1-int(best_code[j]))+best_code[j+1:]
                    if loop_usr_id in code_list:
                        continue
                    else:
                        code_list.append(loop_usr_id)
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=loop_usr_id)
                    sim_list.append((confidence))
                    if confidence>max_sim:
                        max_sim=confidence
                        best_code = code_list[sim_list.index(max_sim)]
                if previous_max_sim==max_sim:
                    stop=True
                previous_max_sim=max_sim
            
            #calculate mapped sim
            if userid in code_list:
                mapped_sim=sim_list[code_list.index(userid)]
            else:
                with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=userid)
                mapped_sim=confidence
            
            #calculate top10
            detect_range=10
            sim_result=[]
            id_result=[]
            id_index=[]
            for r in range(detect_range):
                sim = max(sim_list)
                index = sim_list.index(sim)
                sim_list[index] = -1
                sim_result.append(sim)
                id_result.append(code_list[index])
            result_dic={"sim_score":sim_result, "id":id_result}
            if_succ_top1=0
            if_succ_top3=0

            if userid in id_result[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if userid == id_result[0]:
                succ_num_top1+=1
                if_succ_top1=1

            total_detect_len+=len(sim_list)
        

        pd.get_option('display.width')
        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', None)
        print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3}, time used: {time.time() - start_time}")
        print(f"gen id {userid}, mapped sim {mapped_sim}")
        print(DataFrame(result_dic).T)
        print(f"top1 succ rate: {succ_num_top1}/{exp_num},top3 succ rate: {succ_num_top3}/{exp_num} \n")
        if args.detect_mode=='iterative':
            print("userlist len:",usr_list.shape[-1],"detect list len:",len(sim_list),'average detect list len:',total_detect_len/(i+1))
        
        save_file_name=f"data_wm{args.wm_mode}_d{args.user_dist}_result_m{args.model_name_or_path.split('/')[1]}_d{args.delta}_g{args.max_new_tokens}_t{args.sampling_temp}_d{args.detect_mode}.txt"
        with open(save_file_name, "a+") as f:
            print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3} ,time used: {time.time() - start_time}",file=f)
            print(f"gen id {userid}, mapped sim {mapped_sim}",file=f)
            print(DataFrame(result_dic).T,file=f)
            print(f"top1 succ rate: {succ_num_top1}/{exp_num},top3 succ rate: {succ_num_top3}/{exp_num} \n",file=f)
            if args.detect_mode=='iterative':
                print(f"userlist len:{usr_list.shape[-1]}, detect list len:,{len(sim_list)}, average detect list len: {total_detect_len/(i+1)}\n",file=f)
            print(f"data saved in {save_file_name}")
            f.close()
        


            

    return

def testppl(args):
    dataset_name, dataset_config_name = "c4", "realnewslike"
    dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
    ds_iterator = iter(dataset)
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    usr_list = read_usr_list(args.user_dist)


    model, tokenizer, device = load_model(args)

    # Generate and detect, report to stdout
    exp_num=10
    result_wm=np.zeros(exp_num)
    result_bl=np.zeros(exp_num)
    for i in range(exp_num):
        print(f"{i+1}th exp: ")
        if args.user_dist =='dense':
            # gen_id=i*10
            gen_id=random.randint(0, 127)
        else:
            gen_id=random.randint(0, 31)
            # gen_id=i
        # gen_id=127
        # gen_id=2
        userid = usr_list[gen_id]
        input_text=next(ds_iterator)['text'][:args.prompt_max_length]


        args.default_prompt = input_text

        term_width = 80

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark,_, _ = generate(
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
        oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name, torch_dtype=torch.float16,
                                                         device_map='auto')
        # oracle_model = AutoModelForCausalLM.from_pretrained(oracle_model_name).to(device)
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
        print(baseline_output_wm)
        print(ppl_bl,ppl_wm ,args.delta,args.max_new_tokens)
        
    
        # sys.exit()
    print("wm: ", result_wm.mean(),"baseline: ",result_bl.mean(),args.delta)
    # print(baseline_output_wm)
if __name__ == "__main__":
    args = parse_args()
    # print(args)
    if args.ppl:
        testppl(args)
    else:
        main(args)
    # 
