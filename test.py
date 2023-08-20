
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
import ipdb
from transformers import (AutoTokenizer,
                          AutoModelForSeq2SeqLM,
                          AutoModelForCausalLM,
                          OPTForCausalLM,
                          LogitsProcessorList)

from watermark_processor import  WatermarkDetector_with_preferance, \
    WatermarkLogitsProcessor_with_preferance, UnbiasedWatermarkGenerator,UnbiasedWatermarkDetector
from utils import *
# from datasets import load_dataset, Dataset
from datasets import load_dataset
# from torch.nn.parallel import DataParallel
import torch.nn as nn

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
        "--unbiased_detect_factor",
        type=int,
        default=16,
        help="x for detect",
    )
    parser.add_argument(
        "--wm_mode",
        type=str,
        default="combination",
        help="previous1 or combination",
    )
    parser.add_argument(
        "--gen_mode",
        type=str,
        default="biased",
        help="biased or unbiased",
    )
    parser.add_argument(
        "--detect_mode",
        type=str,
        default="iterative",
        help="normal or iterative or accumulate",
    )
    parser.add_argument(
        "--user_mag",
        type=int,
        default=7,
        help="7 or 10",
    )
    parser.add_argument(
        "--user_dist",
        type=str,
        default="dense",
        help="sparse or dense",
    )
    parser.add_argument(  
        "--delta",
        type=float,
        default=4,
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
        default=70,  # 200
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


def generate(prompt, args, model=None, device=None, tokenizer=None, userid=None):
    """Instatiate the WatermarkLogitsProcessor according to the watermark parameters
       and generate watermarked text by passing it to the generate method of the model
       as a logits processor. """

    watermark_processor = WatermarkLogitsProcessor_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                                   gamma=args.gamma,
                                                                   decrease_delta=args.decrease_delta,
                                                                   delta=args.delta,
                                                                   wm_mode=args.wm_mode,
                                                                   detect_mode=args.detect_mode,
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
    if args.gen_mode =='unbiased':
        generate_without_watermark = partial(
            model.generate,
            return_dict_in_generate=True, # To show score
            output_scores=True,  # To show score
            **gen_kwargs
        )
    else:
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
    b_out='' # for biased return
    if args.gen_mode =='unbiased':
        out_nowm=generate_without_watermark(**tokd_input)
        output_without_watermark = out_nowm[0].int() #torch.Size([1, xx])
        logit = torch.stack(list(out_nowm[1]), dim=0)
        logit=logit.reshape([logit.shape[0],logit.shape[-1]])
        sm=nn.Softmax(dim=1)
        pd=sm(logit) #([xx,50272])
        unb_generator=UnbiasedWatermarkGenerator(pd,userid)
        b_out,output_with_watermark = unb_generator.gen_samp_binary_alphabet(tokd_input["input_ids"][0],args)
        output_with_watermark=torch.reshape(output_with_watermark,(1,-1)).int()
        print(output_with_watermark)
        print(output_without_watermark)
        print(b_out)
        # ipdb.set_trace()
        torch.save(output_with_watermark,'./woutput.pt')
        torch.save(output_without_watermark,'./unwoutput.pt')
        # with open("b_out.txt", "a+") as f:
        #     print(f"{b_out}",file=f)
        

    else:
        output_without_watermark = generate_without_watermark(**tokd_input)
        out=generate_with_watermark(**tokd_input)                                                                                               
        out_se = out[0][:, tokd_input["input_ids"].shape[-1]:]
        output_with_watermark = out[0]
        # out_max_logit=np.zeros(200)
        # out_max_idx=np.zeros(200) # out score.max() - logits.max()=0
        # for k in range(len(out[1])):
        #     out_max_idx[k]=out[1][k].argmax().cpu()
        #     out_max_logit[k]=out[1][k].max().cpu()
        # print("gap between sequence and out.score.argmax()",(out_max_idx-out_se[0].cpu().numpy()).mean())
        # print("gap between out.score.max and logit.max()",(out_max_logit-watermark_processor.max_logit).mean())
        # logits = model(**tokd_input)[0]

    

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

    
    # print("tokdid:",tokd_input["input_ids"].shape)
    # print("before minus:",output_with_watermark.shape)
    if args.is_decoder_only_model:
        # need to isolate the newly generated tokens
        
        output_without_watermark = output_without_watermark[:, tokd_input["input_ids"].shape[-1]:] 
        if args.gen_mode !="unbiased":
            output_with_watermark = output_with_watermark[:, tokd_input["input_ids"].shape[-1]:] #取input_len后的
    
    
    # redecoded_output = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]
    # reencoded_output = tokenizer.encode(redecoded_output, return_tensors='pt',add_special_tokens=False)
    # print(reencoded_output.shape,out_se.shape)
    # sys.exit()
    # print("len in gen:",(output_with_watermark.shape))
    decoded_output_without_watermark = tokenizer.batch_decode(output_without_watermark, skip_special_tokens=True)[0]
    # print(output_without_watermark)
    decoded_output_with_watermark = tokenizer.batch_decode(output_with_watermark, skip_special_tokens=True)[0]

    
    # print(tokd_input["input_ids"].shape[-1])

    return (tokd_input["input_ids"].shape[-1],
            output_with_watermark.shape[-1],
            redecoded_input,
            int(truncation_warning),
            decoded_output_without_watermark,
            decoded_output_with_watermark,
            b_out,
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
    if args.gen_mode=="biased":
        watermark_detector = WatermarkDetector_with_preferance(vocab=list(tokenizer.get_vocab().values()),
                                                            gamma=args.gamma,
                                                            seeding_scheme=args.seeding_scheme,
                                                            device=device,
                                                            wm_mode=args.wm_mode,
                                                            detect_mode=args.detect_mode,
                                                            tokenizer=tokenizer,
                                                            z_threshold=args.detection_z_threshold,
                                                            normalizers=args.normalizers,
                                                            ignore_repeated_bigrams=args.ignore_repeated_bigrams,
                                                            select_green_tokens=args.select_green_tokens,
                                                            userid=userid)
    
        if len(input_text) - 1 > watermark_detector.min_prefix_len:
            score_dict, confidence, mark = watermark_detector.detect(input_text)
            # output = str_format_scores(score_dict, watermark_detector.z_threshold)
            if args.detect_mode == 'accumulate' and confidence == -1:
                return -1,-1,-1,-1,args
            output = list_format_scores(score_dict, watermark_detector.z_threshold)
        else:
            # print("debug in else")
            # output = (f"Error: string not long enough to compute watermark presence.")
            output = [["Error", "string too short to compute metrics"]]
            output += [["", ""] for _ in range(6)]
        return output, confidence, mark,watermark_detector, args
    elif args.gen_mode=="unbiased":
        watermark_detector=UnbiasedWatermarkDetector(input_text,device,userid,args.unbiased_detect_factor)
        score=watermark_detector.detect(args)
        return -1, score, -1, -1,args



def main(args):
    # b_out='11000011110001100010001011011111000001111111111101011111101101010000000000111101000000000000101000000000000000000111111111111111000000011001111100000000000000000000000000000001000000000000000000001100100101100101111110110101000000000001111100001111110111110000000001011010000111000101011100010110000010010000000000000110000000100001010000001011101001101000000100111101000000000011101000100001110000100000000000001111000001001100110100000100100001110000000000001001000010111001110100000000000001100000000011011001000000000000001100000011100101110000000000000100001010000001011101000000001111000000001010000100010101000100000100000000000011000000001110111110010100110101000000000000000001101100001110000110001001101000110101011000100010010000000000001001000000000000010100000011001101110000000000100001000000000000111000000000000101000010010111000110000101010001011000000100001001010000001000101011000000011010011100000000000101110000000001001101000010000011011100000000010001100000000001010111000000000000010100001001110001000101010011100011000000000011110000000000000110100000101000101001000001001010111000001010000100010000010101111111000000000000011000000000000001100000001001100001001001110011100000000000000011000000111001110111000000000000100100001000000001010000100111111110000011011011010100000000000001110000000000000100000000101100011000001101100110010110110010011001010101000111110100000000001000110000000001110100000000000111111100000000101101110001111111010100000000000000011100001010000010010010101000001111000010010000010100000000010011010000000100000101000000100001111000001111111100000000000001001100110000111100011000000001110001010000010000001001000000101001000100000000001100000001001100001110000101111111110000000000000111100000000000000100000000101111000100110000011000100000011001011000010111011001110100001100001110110000000000100010100000011100011000000000000100000000011010001010000000000001001100001010100001110000110110110101000000000000011100000000000001010000001111110111000000000000010000000000001010000000001000000111000000000010001100000000000001010000010001010010010001001111001000000000010011000000000000100111000000000000010011000011100001100010011010001111000000101111101000000000011101000000000000011100000000000000100000000000010011100000000000010100000011010110101100101110111011000000010010010110010111111011010100000000000001110000000011101110000000001110010100000010100011000000000000000110000000100110000100000010100100010010010000010100000000010000111100000100110101010000011001011000010000000001001010000001000101100010000011100011000000010100101000001000000111000000000000001000010100001101011000000110111111100000000000001100001001111111111100000001100010000000000000000001100001001001000000000101101001110101111110110101000000000001110100000000000010101100001111000110000000011100010100001000111011100000000001110100000000000011000000000000000101110000100001101001000000000000010100001001011100100000000000001001000010011111011000001000000001100000011011000000001001001010001000000110010110100000000000000110000000000011110100100000000100000000000111111100000000000000111100000000101000010000000010000111000000000001100101101110001100100010100111011111'
    # userid='1111100'
    
    
    #fold
    #load dataset
    dataset_name, dataset_config_name = "c4", "realnewslike"
    dataset = load_dataset(dataset_name, dataset_config_name, split="validation", streaming=True)
    ds_iterator = iter(dataset)
    
    start_time = time.time()
    """Run a command line version of the generation and detection operations
        and optionally launch and serve the gradio demo"""
    # Initial arg processing and log
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    usr_list = read_usr_list(args.user_dist,args.user_mag)
    model, tokenizer, device = load_model(args)

    
    exp_num=200
    
    total_detect_len=0 #for detect mode: iterative
    succ_num_top1=0
    succ_num_top3=0
    # for t in range(17):
    #         input_text=next(ds_iterator)
    for i in range(exp_num):
        print(f"{i}th exp: ")
        if args.user_dist =='dense':
            # gen_id=i
            gen_id=random.randint(0, 2**args.user_mag-1)
        else:
            gen_id=random.randint(0, 31)
            # gen_id=i
        # gen_id=127
        # gen_id=2
        userid = usr_list[gen_id]

        input_text=next(ds_iterator)['text'][:args.prompt_max_length]
        args.default_prompt = input_text

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark,b_out, watermark_processor,_ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid)





        # del(watermark_processor)
        #fold
        #start detect
        
        if args.detect_mode == 'normal':
        
            sim_list = []
            max_sim = 0
            max_sim_idx = -1
            
            for j in range(usr_list.shape[-1]):

                loop_usr_id = usr_list[j]
                if args.gen_mode=='biased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                elif args.gen_mode=='unbiased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(b_out,
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
            result_dic={"sim_score":torch.tensor(sim_result).cpu(), "id":id_result}
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

                if args.gen_mode=='biased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                elif args.gen_mode=='unbiased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(b_out,
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
                    

                    if args.gen_mode=='biased':
                        with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                    elif args.gen_mode=='unbiased':
                        with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(b_out,
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
                # if args.gen_mode=='biased':
                if args.gen_mode=='biased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                elif args.gen_mode=='unbiased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(b_out,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)

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
            result_dic={"sim_score":torch.tensor(sim_result).cpu(), "id":id_result}
            if_succ_top1=0
            if_succ_top3=0

            if userid in id_result[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if userid == id_result[0]:
                succ_num_top1+=1
                if_succ_top1=1

            total_detect_len+=len(sim_list)
        if args.detect_mode == 'accumulate':
        
            sim_list = []
            max_sim = 0
            max_sim_idx = -1
            
            for j in range(usr_list.shape[-1]):

                loop_usr_id = usr_list[j]

                if args.gen_mode=='biased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(decoded_output_with_watermark,
                                                                                args,
                                                                                device=device,
                                                                                tokenizer=tokenizer,
                                                                                userid=loop_usr_id)
                elif args.gen_mode=='unbiased':
                    with_watermark_detection_result, confidence, mark,watermark_detector,_ = detect(b_out,
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
            result_dic={"sim_score":torch.tensor(sim_result).cpu(), "id":id_result}
            if_succ_top1=0
            if_succ_top3=0

            if gen_id in id_index[:3]:
                succ_num_top3 += 1
                if_succ_top3=1

            if max_sim_idx == gen_id:
                succ_num_top1+=1
                if_succ_top1=1

        pd.get_option('display.width')
        pd.set_option('display.width', 500)
        pd.set_option('display.max_columns', None)
        print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3}, time used: {time.time() - start_time}")
        print(f"gen id {userid}, mapped sim {mapped_sim}")
        print(DataFrame(result_dic).T)
        print(f"top1 succ rate: {succ_num_top1}/{exp_num},top3 succ rate: {succ_num_top3}/{exp_num} \n")
        if args.detect_mode=='iterative':
            print("userlist len:",usr_list.shape[-1],"detect list len:",len(sim_list),'average detect list len:',total_detect_len/(i+1))
        
        save_file_name=f"data_wm{args.wm_mode}_d{args.user_dist}_mag{args.user_mag}_m{args.model_name_or_path.split('/')[1]}_d{args.delta}_g{args.max_new_tokens}_t{args.sampling_temp}_gen{args.gen_mode}_d{args.detect_mode}.txt"
        with open(save_file_name, "a+") as f:
            print(f"exp  {i}, if top1 succ: {if_succ_top1} ,if top3 succ: {if_succ_top3} ,time used: {time.time() - start_time}",file=f)
            print(f"gen id {userid}, mapped sim {mapped_sim}",file=f)
            print(DataFrame(result_dic).T,file=f)
            print(f"top1 succ rate: {succ_num_top1}/{exp_num},top3 succ rate: {succ_num_top3}/{exp_num} ",file=f)
            if args.detect_mode=='iterative':
                print(f"userlist len:{usr_list.shape[-1]}, detect list len:,{len(sim_list)}, average detect list len: {total_detect_len/(i+1)}",file=f)
            print(f"\n data saved in {save_file_name}\n")
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
    usr_list = read_usr_list(args.user_dist,args.user_mag)


    model, tokenizer, device = load_model(args)

    # Generate and detect, report to stdout
    exp_num=1
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
        
        #fold
        userid = usr_list[gen_id]
        input_text=next(ds_iterator)['text'][:args.prompt_max_length]


        args.default_prompt = input_text

        term_width = 80

        input_token_num, output_token_num, _, _, decoded_output_without_watermark, decoded_output_with_watermark,b_out,_, _ = generate(
            input_text,
            args,
            model=model,
            device=device,
            tokenizer=tokenizer,
            userid=userid)
        #fold
        # userid='0110000'
        # decoded_output_without_watermark = tokenizer.batch_decode(torch.load("./unwoutput.pt"), skip_special_tokens=True)[0]

        # decoded_output_with_watermark = tokenizer.batch_decode(woutput, skip_special_tokens=True)[0]

        
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
    # wn=torch.load('./unwoutput.pt')
    # w=torch.load('./woutput.pt')
    # # ipdb.set_trace()
    # print(torch.load('./unwoutput.pt'))
    # print(torch.load('./woutput.pt'))
    # print(args)
    if args.ppl:
        testppl(args)
    else:
        main(args)
    # 
