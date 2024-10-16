# User guide for llm identification
Official Implementation of 'Where Am I From? Identifying Origin of LLM-generated Content'
---



## Installation:

Setting up the environment : python 3.9 is recommended

```sh
pip install -r requirments.txt
```

## Useage

To generate userlist, please run `gen_usr_list_dense()` in utils.py, modify 'magnitude' to control user pool size.

To generate and detect watermark:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0 --max_new_tokens 200 --delta 2 --user_magnitude 10 
```

Note: the above settings depends on RTX3090, to run opt-13b model, at least 2 gpu devices should be availiable
```
CUDA_VISIBLE_DEVICES=0,1 python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0 --max_new_tokens 200 --delta 2 --user_magnitude 10 

```

To evaluate perplexity:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b --user_dist dense --wm_mode combination --max_new_tokens 200 --delta 2 --ppl 1
```

To detect watermark under attack:
```sh
CUDA_VISIBLE_DEVICES=0 python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0.1 --max_new_tokens 200 --delta 2 --user_magnitude 10 
```



note: the above settings depends on RTX3090, if you use other smaller devices, please modify `load_model()` in `test.py`.




