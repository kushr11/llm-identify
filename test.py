
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
import copy
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
import itertools
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
        default="normal",
        help="normal",
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="depth_d",
        help="depth_d, normal",
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=3,
        help="sub list number",
    )
    parser.add_argument(
        "--dataset",
        type = str,
        default = 'c4',
        help = "c4, squad, xsum, PubMedQA, writingprompts",
    )
    parser.add_argument(
        "--user_dist",
        type=str,
        default="dense",
        help="sparse or dense",
    )
    parser.add_argument(
        "--user_magnitude",
        type=int,
        default=7,
        help="user number = 2**magnitude",
    )
    parser.add_argument(  
        "--delta",
        type=float,
        default=5,
        help="The (max) amount/bias to add to each of the greenlist token logits before each token sampling step.",
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
        default=25,  # 200
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

    watermark_processor = WatermarkLogitsProcessor_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                                   gamma=args.gamma,
                                                                   decrease_delta=args.decrease_delta,
                                                                   delta=args.delta,
                                                                   wm_mode=args.wm_mode,
                                                                   seeding_scheme=args.seeding_scheme,
                                                                   select_green_tokens=args.select_green_tokens,
                                                                   userid=userid,
                                                                   args=args
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
    # print(len(tokd_input[0]))

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
    # torch.save(logit,f"./assest/clean_z_200/clean_z_{index}.pt")
    
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
                                                           userid=userid,
                                                           args=args)
    if len(input_text) - 1 > watermark_detector.min_prefix_len:
        score_dict, gr_score,depth_score, mark = watermark_detector.detect(input_text)
        # output = str_format_scores(score_dict, watermark_detector.z_threshold)
        output = list_format_scores(score_dict, watermark_detector.z_threshold)
    else:
        # print("debug in else")
        # output = (f"Error: string not long enough to compute watermark presence.")
        output = [["Error", "string too short to compute metrics"]]
        output += [["", ""] for _ in range(6)]
    return output, gr_score,depth_score, mark,watermark_detector, args

def load_dataiter_by_name(dataset_name):
    if dataset_name == 'c4':
        dataset_config_name= "realnewslike"
        dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
        ds_iterator = iter(dataset)

    # TODO: there will be 6 datasets that needs to be implemented from "DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature"
    # dataset: xsum
    if dataset_name == 'xsum':
        dataset_config_name= "realnewslike"
        dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
        ds_iterator = iter(dataset)
        
    # dataset: squad
    # dataset_config_name = plain_text
    if dataset_name == 'squad':
        dataset_config_name = "plain_text"
        dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
        ds_iterator = iter(dataset)
        

    # dataset: WMT16
    # dataset_config_name = 'ro-en'
    if dataset_name == 'wmt16':
        dataset_config_name = 'ro-en'
        dataset = load_dataset(dataset_name, split="validation", streaming=True)
        ds_iterator = iter(dataset)

    # dataset: PubMedQA
    # dataset_config_name = 'pubmed_qa_artificial_source'
    if dataset_name == 'PubMedQA':
        file_path = 'ori_pqal.json'
        dataset = Dataset(file_path)
        ds_iterator = iter(dataset.get_items())
        _, item_data = next(ds_iterator)

    # dataset: Reddit WritingPrompts dataset
    # we load from local source
    if dataset_name == 'writingprompts':
        basepath='/data1/ybaiaj/llm-identify-main-new-6.7b-25/writing-prompts'
        dataset = kaggle_ds.load_data(basepath)
        ds_iterator = dataset.iterrows()
        _, row_data = next(ds_iterator)
    return  ds_iterator

def load_text_by_iter(dataset_name,ds_iterator):
    # dataset: xsum
        if dataset_name == 'xsum':
            if args.prompt_max_length is not None and len(next(ds_iterator)['document']) >= args.prompt_max_length:
                input_text = next(ds_iterator)['document'][:args.prompt_max_length]
            else:
                input_text = next(ds_iterator)['document'][:]

        if dataset_name == 'squad':
            if args.prompt_max_length is not None and len(next(ds_iterator)['context']) >= args.prompt_max_length:
                input_text = next(ds_iterator)['context'][:args.prompt_max_length]
            else:
                input_text = next(ds_iterator)['context'][:]
            

        # dataset: WMT16
        # dataset_config_name = 'ro-en'
        if dataset_name == 'wmt16':
            if args.prompt_max_length is not None and len(next(ds_iterator)['translation']) >= args.prompt_max_length:
                input_text = tuple(next(ds_iterator)['translation'])[:args.prompt_max_length]
            else:
                input_text = tuple(next(ds_iterator)['translation'])[:]

        # dataset: PubMedQA
        # dataset_config_name = 'pubmed_qa_artificial_source'
        if dataset_name == 'PubMedQA':
            _, item_data = next(ds_iterator)
            if args.prompt_max_length is not None and len(item_data['LONG_ANSWER']) >= args.prompt_max_length:
                input_text = item_data['LONG_ANSWER'][:args.prompt_max_length]
            else:
                input_text = item_data['LONG_ANSWER'][:]

        # dataset: Reddit WritingPrompts dataset
        # we load from local source
        if dataset_name == 'writingprompts':
            _, row_data = next(ds_iterator)
            if args.prompt_max_length is not None and len(row_data['story']) >= args.prompt_max_length:
                input_text = row_data['story'][:args.prompt_max_length]
            else:
                input_text = row_data['story']
        return input_text
def main(args):
    # Load datasets
    dataset_name = args.dataset
    ds_iterator=load_dataiter_by_name(dataset_name)

    
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    usr_list = read_usr_list(args.user_dist,args.user_magnitude)

    model, tokenizer, device = load_model(args)
    
    exp_num=200
    
    total_detect_len=0 #for detect mode: iterative
    succ_num_top1=0
    succ_num_top3=0
    succ_num_top10=0
    # for t in range(17):
    #         input_text=next(ds_iterator)
    for i in range(exp_num):
        print(f"{i}th exp: ")
        if args.user_dist =='dense':
            gen_id=random.randint(0, 2**args.user_magnitude-1)
        else:
            gen_id=random.randint(0, 31)

        # gen_id=5
        userid = usr_list[gen_id]
        
        input_text=load_text_by_iter(dataset_name,ds_iterator)
        
        
        args.default_prompt = input_text

        term_width = 80


        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid,
            index=i)
        
        
        # continue


        # loop_usr_id = userid
        # with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector, _ = detect(decoded_output_with_watermark,
        #                                                                 args,
        #                                                                 device=device,
        #                                                                 tokenizer=tokenizer,
        #                                                          userid=loop_usr_id)


        # print(gr_score,depth_score)
        # sys.exit()

        # del(watermark_processor)
        #start detect
        
        if args.detect_mode == 'normal':
        
            gr_score_list = []
            depth_score_list=[]
            max_sim = 0
            max_sim_idx = -1
            
            for j in range(usr_list.shape[-1]):

                loop_usr_id = usr_list[j]

                with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                # sim = compute_similarity(mark, loop_usr_id)
                gr_score_list.append(gr_score)
                depth_score_list.append(depth_score)
                if gr_score> max_sim:
                    max_sim = gr_score
                    max_sim_idx = j
            
            detect_range=len(usr_list)//10
            mapped_gr_score=gr_score_list[gen_id]
            mapped_depth_score=depth_score_list[gen_id]
            # sim_result=np.zeros([2,detect_range])
            gr_result=[]
            depth_result=[]
            id_result=[]
            id_index=[]
            # for r in range(detect_range): # sort with gr
            #     sim = max(gr_score_list)
            #     index = gr_score_list.index(sim)
            #     gr_score_list[index] = -1
            #     gr_result.append(sim)
            #     depth_result.append(depth_score_list[index])
            #     id_result.append(usr_list[index])
            #     id_index.append(index)
            # c_depth_result=copy.deepcopy(depth_result)
            # final_index=[]
            # # ipdb.set_trace()
            # for r in range(detect_range): # sort with depth
            #     sim = max(c_depth_result)
            #     index = c_depth_result.index(sim)
            #     c_depth_result[index] = -100
            #     final_index.append(index)
            
            if args.gen_mode=='depth_d':

                #sort with comb depth score
                for r in range(detect_range): # sort with depth
                    sim = max(depth_score_list)
                    index = depth_score_list.index(sim)
                    depth_score_list[index] = -100
                    depth_result.append(sim)
                    id_result.append(usr_list[index])
                    gr_result.append(gr_score_list[index])
                    id_index.append(index)

                result_dic={"gr_score":gr_result[:10], "depth_score":depth_result[:10],"id":id_result[:10]}
                if_succ_top1=0
                if_succ_top3=0
                if_succ_top10=0
                
                
                
                
                
                
            elif args.gen_mode=='normal':
                for r in range(detect_range): # sort with depth
                    sim = max(gr_score_list)
                    index = gr_score_list.index(sim)
                    gr_score_list[index] = -100
                    gr_result.append(sim)
                    id_result.append(usr_list[index])
                    id_index.append(index)

                result_dic={"gr_score":gr_result[:10], "id":id_result[:10]}
                if_succ_top1=0
                if_succ_top3=0
                if_succ_top10=0

            if gen_id in id_index[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if gen_id in id_index[:1]:
                succ_num_top1+=1
                if_succ_top1=1
                
            if gen_id in id_index[:10]:
                succ_num_top10+=1
                if_succ_top10=1
        
        
        if args.detect_mode == 'iterative':
            detect_range=len(usr_list)//10
            max_sim=-1
            init_num=3
            gr_score_list=[]
            depth_score_list=[]
            code_list=[]
            ## append initial code
            code_list.append(usr_list[0])
            for j in range (init_num-1):
                code_list.append(usr_list[(usr_list.shape[-1]//init_num)*(j+1)])
            code_list.append(usr_list[-1])
            for loop_usr_id in code_list:
                with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=loop_usr_id)
                gr_score_list.append((gr_score))
                depth_score_list.append(depth_score)
            max_sim=max(gr_score_list)
            best_code = code_list[gr_score_list.index(max_sim)]
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
                    with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=loop_usr_id)
                    gr_score_list.append((gr_score))
                    depth_score_list.append(depth_score)
                    if gr_score>max_sim:
                        max_sim=gr_score
                        best_code = code_list[gr_score_list.index(max_sim)]
                if previous_max_sim==max_sim:
                    stop=True
                previous_max_sim=max_sim
            
            #calculate mapped sim
            if userid in code_list:
                mapped_gr_score=gr_score_list[code_list.index(userid)]
                mapped_depth_score=depth_score_list[code_list.index(userid)]
            else:
                with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                    args,
                                                                                    device=device,
                                                                                    tokenizer=tokenizer,
                                                                                    userid=userid)
                mapped_gr_score=gr_score
                mapped_depth_score=depth_score
            
            #diverge best code
            top10pids=[best_code]
            top10p_depth_score=[]
            top10p_gr_score=[]
            indicator=1
            item=[i for i in range(len(best_code))]
            while len(top10pids)<detect_range:
                comb=[p for p in itertools.combinations(item,indicator)]
                for c in comb:
                    modi_code=best_code
                    for j in c:
                        if j==0:
                            modi_code=str(1-int(modi_code[j]))+modi_code[j+1:]
                        elif j==len(best_code)-1:
                            modi_code=modi_code[:-1]+str(1-int(modi_code[j]))
                        else:
                            modi_code=modi_code[:j]+str(1-int(modi_code[j]))+modi_code[j+1:]
                    top10pids.append(modi_code)
                    if len(top10pids)>=detect_range:
                        break
                indicator+=1

            # cal top10 percentage depth score
            for loop_id in top10pids:
                with_watermark_detection_result, gr_score,depth_score, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                        args,
                                                                                        device=device,
                                                                                        tokenizer=tokenizer,
                                                                                        userid=loop_id)
                top10p_gr_score.append(gr_score)
                top10p_depth_score.append(depth_score)
                # ipdb.set_trace()
                        
            # calculate top10
            detect_range=10
            gr_result=[]
            depth_result=[]
            id_result=[]
            id_index=[]
            for r in range(detect_range):
                sim = max(top10p_depth_score)
                index = top10p_depth_score.index(sim)
                gr_result.append(top10p_gr_score[index])
                depth_result.append(top10p_depth_score[index])
                top10p_depth_score[index] = -999999
                id_result.append(top10pids[index])

            
            result_dic={"gr_score":gr_result, "depth_score":depth_result,"id":id_result[:10]}
            if_succ_top1=0
            if_succ_top3=0
            if_succ_top10=0

            if userid in id_result[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if userid == id_result[0]:
                succ_num_top1+=1
                if_succ_top1=1
            
            if userid in id_result[:10]:
                succ_num_top10+=1
                if_succ_top10=1

            total_detect_len+=len(gr_score_list)
        

        pd.get_option('display.width')
        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', None)
        print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3}, if top10 succ: {if_succ_top10},time used: {time.time() - start_time}")
        if args.gen_mode=="depth_d":
            print(f"gen id {userid}, mapped depth {mapped_depth_score}, mapped gr {mapped_gr_score}")
        else:
            print(f"gen id {userid}, mapped gr {mapped_gr_score}")
        print(DataFrame(result_dic).T)
        print(f"top1 succ rate: {succ_num_top1}/{i+1},top3 succ rate: {succ_num_top3}/{i+1}, top10 succ rate: {succ_num_top10}/{i+1} \n")
        if args.detect_mode=='iterative':
            print("userlist len:",usr_list.shape[-1],"detect list len:",len(gr_score_list),'average detect list len:',total_detect_len/(i+1))

        save_file_name=f"./results/comb_score_d{args.user_dist}_m{args.model_name_or_path.split('/')[1]}_d{args.delta}_l{args.max_new_tokens}_d{args.detect_mode}_g{args.gen_mode}_mag{args.user_magnitude}.txt"
        with open(save_file_name, "a+") as f:
            print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3} , if top10 succ: {if_succ_top10},time used: {time.time() - start_time}",file=f)
            if args.gen_mode=="depth_d":
                print(f"gen id {userid}, mapped depth {mapped_depth_score}, mapped gr {mapped_gr_score}",file=f)
            else:
                print(f"gen id {userid}, mapped gr {mapped_gr_score}",file=f)
            print(DataFrame(result_dic).T,file=f)
            print(f"top1 succ rate: {succ_num_top1}/{i+1},top3 succ rate: {succ_num_top3}/{i+1} ,top10 succ rate: {succ_num_top10}/{i+1} \n",file=f)
            if args.detect_mode=='iterative':
                print(f"userlist len:{usr_list.shape[-1]}, detect list len:,{len(gr_score_list)}, average detect list len: {total_detect_len/(i+1)}\n",file=f)
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
    usr_list = read_usr_list(args.user_dist,args.user_magnitude)


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

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark, watermark_processor,_ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid,
            index=i)
        
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
