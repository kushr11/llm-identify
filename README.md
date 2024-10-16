# Where Am I From? Identifying Origin of LLM-generated Content
---
Official Implementation of 'Where Am I From? Identifying Origin of LLM-generated Content'



## Installation:
Setting up the environment : python 3.9 is recommended
```sh
pip install -r requirments.txt
```
## Usage:
To generate userlist, please run `gen_usr_list_dense()` in utils.py, modify 'magnitude' to control user pool size.

To generate and detect watermark:
```sh
python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0 --max_new_tokens 200 --delta 2 --user_magnitude 10 
```

To evaluate perplexity:
```sh
python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0 --max_new_tokens 200 --delta 2 --user_magnitude 10 --ppl 1
```

To detect watermark under attack:
```sh
python test.py --model_name_or_path facebook/opt-1.3b ----attack_ep 0.1 --max_new_tokens 200 --delta 2 --user_magnitude 10 
```



