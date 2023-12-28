# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
import torch.nn.functional as F
import random
import time
import torch
from functools import partial
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
import ipdb

device="cuda"
# tokenizer = AutoTokenizer.from_pretrained("t5-large")
# model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
model.to(device)
# prompt="The systems interconnect is expected to cost more than $5 million and to begin to be released by 2010. Its information technology and telecommunications team, under Andy Hill, will write and deploy the operating system for the proposed supercomputer infrastructure."
prompt="The systems interconnect is expected to cost"
tokd_input = tokenizer(prompt, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=20)["input_ids"].to(device)
# ipdb.set_trace()
# print(tokd_input)
# out=model.generate(tokd_input,max_new_tokens=1)
# print(out[0][-1])
# decoded_out= tokenizer.decode(out[0], skip_special_tokens=True,max_new_tokens=50)
# print(decoded_out)

#cut prompt->1 token
def replace_1_token(cut_prompt):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cut_prompt=cut_prompt.unsqueeze(0)
    gen_kwargs = dict(max_new_tokens=1)
    generate_with_score = partial(
        model.generate,
        output_scores=True,
        return_dict_in_generate=True, 
    )
    # tokd_input = tokenizer(cut_prompt, return_tensors="pt", add_special_tokens=True, truncation=True)["input_ids"].to(device)
    out=generate_with_score(cut_prompt,pad_token_id=tokenizer.pad_token_id,max_new_tokens=1)
    # decoded_out= tokenizer.decode(out[0], skip_special_tokens=True,max_new_tokens=50)
    res=out[0][0][-1]
    logit=torch.stack(out[1])
    sm=nn.Softmax(dim=1)
    pd=sm(logit.squeeze(0))
    top_values, top_indices = torch.topk(pd, 2)
    
    top1ind=top_indices[0][0].item()
    top2ind=top_indices[0][1].item()
    return res,top1ind,top2ind
def attack_process(decoded_output,epsilon):
    epsilon=epsilon
    skip=False
    succ=0
    
    tokd_input = tokenizer(decoded_output, return_tensors="pt", add_special_tokens=True, truncation=True,
                           max_length=300)["input_ids"].to(device)
    attacked_output=tokd_input.detach()
    # print(attacked_output)
    attack_num=epsilon*tokd_input.shape[-1]
    # select_list=[]
    # ipdb.set_trace()
    
    if(int(attack_num)>=tokd_input.shape[-1]-31):
        skip=True
        return decoded_output, skip
    random_numbers = random.sample(range(31, tokd_input.shape[-1]-1), int(attack_num))

    for rand_idx in random_numbers:

        # random.seed(time.time())
        # print(time.time())
        # rand_idx = random.randint(31,tokd_input.shape[-1]-1)
        # print(attack_num,succ,c,rand_idx,len(select_list))
        cut_prompt=tokd_input[0][rand_idx-30:rand_idx]
        standard_token=tokd_input[0][rand_idx]
        select_token,top1ind,top2ind=replace_1_token(cut_prompt)
        if(standard_token!=select_token):
            attacked_output[0][rand_idx]=select_token
        else:
            if (standard_token!=top1ind):
                attacked_output[0][rand_idx]=top1ind
            else:
                attacked_output[0][rand_idx]=top2ind
        succ+=1
                
    # print(attacked_output)
    decoded_output= tokenizer.decode(attacked_output[0], skip_special_tokens=True)
    return decoded_output,skip


# prompt1="The systems interconnect is expected to cost more than $5 million and to begin to be released by 2010. Its information technology and telecommunications team, under Andy Hill, will write and deploy the operating system for the proposed supercomputer infrastructure."

# print(attack_process(prompt1))
# print(prompt1)
                