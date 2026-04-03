"""
Created on Tue Oct 8 15:38:42 2025

@author: zhouy
    
"""

import pandas as pd
import json
from openai import OpenAI
from tqdm import tqdm
import os
import argparse
import time
import csv
import torch
import logging
from utils import *
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from transformers import pipeline

os.environ['DEEPSEEK_API_KEY'] = 'sk-xxxxxxxxxxxxxxxxxx'
os.environ['OPEN_ROUTER_KEY'] = 'sk-xxxxxxxxxxxxxxxxxxxxxx'

def add_parser():
    parser = argparse.ArgumentParser(description='data processing')
    parser.add_argument('-model', type=str, default='txgemma-9b-chat', help='LLM model')
    parser.add_argument('-device', type=str, default='auto', help='inference device')
    parser.add_argument('-ckpt', type=str, default='./ckpt', help='model checkpoint path')
    parser.add_argument('-data', type=str, default='./dataset', help='benchmark data path')
    parser.add_argument('-simplify', action="store_true", help='use simplified prompt or not')
    parser.add_argument('-api', action="store_true", help='use local model or not')
    parser.add_argument('-platform', type=str, default='deepseek', help='API model platform')
    parser.add_argument('-result_dir', type=str, default='demo_result', help='saving path for all results')
    parser.add_argument('-token', type=int, default=1024, help='number of maximum output tokens, 4096 for LLM and 2048 for domain')
    parser.add_argument('-system', action="store_true", help='add system role and content')
    parser.add_argument('-think', action="store_true", help='use thinking mode')
    parser.add_argument('-tool', action="store_true", help='use tools for prediction')

    args = parser.parse_args()
    return args

def init_env_ds():

    api_key = os.getenv('DEEPSEEK_API_KEY')
    if api_key:
        print("API Key is set successfully.")
    else:
        print("Failed to set API Key.")
    
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    return client

def init_env_vllm():

    client = OpenAI(
            api_key="EMPTY",
            base_url="http://0.0.0.0:8887/v1",
        )
    
    return client   


def init_open_router():
    
    api_key = os.getenv('OPEN_ROUTER_KEY')
    
    if api_key:
        print("Open router API Key is set successfully.")
    else:
        print("Failed to set open router API Key.")

    client = OpenAI(api_key=api_key,
                    base_url="https://openrouter.ai/api/v1")
    
    return client

def load_models(args):
    model_path = os.path.join(args.ckpt, args.model)
    if args.model.startswith('biot5-plus'):
        tokenizer = T5Tokenizer.from_pretrained(model_path, model_max_length=4096)
        model = T5ForConditionalGeneration.from_pretrained(model_path, device_map=args.device)
    elif args.model.startswith('gemma-3'):
        model = Gemma3ForConditionalGeneration.from_pretrained(model_path, device_map=args.device).eval()
        tokenizer = AutoProcessor.from_pretrained(model_path)
        model = pipeline(
                "image-text-to-text",
                model=model_path,
                device=args.device,
            )
        tokenizer = None
    
    elif args.model == 'Llama-3.1-8B-base':
        tokenizer = None
        model = pipeline('text-generation', model=model_path, 
                         model_kwargs={"torch_dtype": torch.bfloat16},
                         device_map=args.device)
        
    else:
        if args.model.startswith('ChemLLM') or args.model.startswith('NatureLM-'):
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
        if args.model.startswith('NVIDIA-') or args.model.startswith('ChemLLM') or args.model.startswith('NatureLM-'):
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=args.device,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map=args.device,
            )
        
    args.tokenizer = tokenizer
    # print(model)
        
    return model
          
        
        
def load_data(data_path):
    if data_path.endswith('.json'):
        try:
            input_df = pd.DataFrame(pd.read_json(data_path, orient='table')['data'][0])
        except:
            input_df = pd.DataFrame(pd.read_json(data_path, orient='table'))
    elif data_path.endswith('.parquet'):
        input_df = pd.read_parquet(data_path, engine="pyarrow")
    else:
        return None
    
    print(len(input_df))
    
    return input_df[:10]


def process(args, model, task_name, all_prompts, keys_all_prompts, path_all_prompts, par_dir, tool_all_prompts):
    data_path = os.path.join(args.data, path_all_prompts[task_name])
    
    output_folder = os.path.join(par_dir, os.path.dirname(data_path))
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    input_df = load_data(data_path)

    filename = 'result_' + os.path.basename(data_path)
    output_file = os.path.join(output_folder, filename)
    
    ptemp = all_prompts[task_name]
    pkeys = keys_all_prompts[task_name]
    sysp = all_prompts['SYS_Prompt']
    if args.tool:
        tool_func = tool_all_prompts[task_name]
    else:
        tool_func = None
    
    results = []
    count = 0
    for _, row in tqdm(input_df.iterrows(), desc="Processing samples", disable=True):
        if count % 10 == 0 or count + 1 == len(input_df):
            logging.info(f'{count}/{len(input_df)}: start at time {time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())}.')

        if 'biot5' in args.model:
            pre_text = ''
            pre_mol_id = '<bom>'
            post_mol_id = '<eom>'
            pre_prot_id = '<bop>'
            post_prot_id = '<eop>'
            aa_id = '<p>'
            domain_flag = True
            selfies_flag = True

        elif 'NatureLM' in args.model:
            pre_text = ''
            pre_mol_id = '<mol>'
            post_mol_id = '</mol>'
            pre_prot_id = '<protein>'
            post_prot_id = '</protein>'
            aa_id = '<a>'
            domain_flag = True
            selfies_flag = False

        elif 'BioMol' in args.model:
            if task_name == 'General_Text':
                pre_text = ''
            else:
                pre_text = '<BOS>'
            pre_mol_id = '<SELFIES>'
            post_mol_id = '</SELFIES>'
            pre_prot_id = '<FASTA>'
            post_prot_id = '</FASTA>'
            aa_id = '<p>'
            domain_flag = True
            selfies_flag = True

        else:
            pre_text = ''
            pre_mol_id = ''
            post_mol_id = ''
            pre_prot_id = ''
            post_prot_id = ''
            aa_id = ''
            domain_flag = False
            selfies_flag = False
                
        content = tuple([prefix_input(row, key, pre_text, pre_mol_id, post_mol_id, pre_prot_id, post_prot_id, aa_id, domain_flag, selfies_flag) for key in pkeys])
        prompt = ptemp.format(*content)
        
        if args.simplify:
            prompt = '\n'.join([line for line in prompt.split('\n') 
                   if not (line.startswith('Instruction') or line.startswith('Confidence') or line.startswith('Explanation'))])

        if args.api:
            response, message = query_api(args, model, prompt, sysp, tool_func)
        else:
            response, message = query_model(args, model, prompt, sysp, tool_func)
        results.append(response)
                
        count += 1
                
    input_df['Model_Response'] = results
    logging.info(message)
    logging.info(response)
    if output_file.endswith('.parquet'):
        input_df.to_parquet(output_file, engine='pyarrow')
    elif output_file.endswith('.json'):
        if 'general_text' in output_file:
            input_df.to_json(output_file, orient='records', lines=True)
        else:
            input_df.to_json(output_file, orient='table', indent=4)
    return #results

def load_tasks():
    if 'NatureLM' in args.model:
        with open(os.path.join(args.data, 'all_prompts_naturelm.json'), 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)    
    elif 'txgemma' in args.model:
        with open(os.path.join(args.data, 'all_prompts_txgemma.json'), 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)    
    else:
        with open(os.path.join(args.data, 'all_prompts.json'), 'r', encoding='utf-8') as f:
            all_prompts = json.load(f)
            
    with open(os.path.join(args.data, 'keys_all_prompts.json'), 'r', encoding='utf-8') as f:
        keys_all_prompts = json.load(f)
        
    with open(os.path.join(args.data, 'path_all_prompts.json'), 'r', encoding='utf-8') as f:
        path_all_prompts = json.load(f)

    with open(os.path.join(args.data, 'tool_all_prompts.json'), 'r', encoding='utf-8') as f:
        tool_all_prompts = json.load(f)
        
    return all_prompts, keys_all_prompts, path_all_prompts, tool_all_prompts

args = add_parser()
START_TIME = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())

def main():
    if args.api:
        if args.platform == 'deepseek':
            model = init_env_ds()
        elif args.platform == 'openrouter':
            model = init_open_router()
        else:
            model = init_env_vllm()
    else:
        model = load_models(args)

    all_prompts, keys_all_prompts, path_all_prompts, tool_all_prompts = load_tasks()
    
    base_dir = f'./{args.result_dir}/{args.model.split('/')[-1]}'
    par_dir = os.path.join(base_dir, START_TIME)
    
    if not os.path.exists(par_dir):
        os.makedirs(par_dir)
    
    logging.basicConfig(
        filename=os.path.join(par_dir, 'evaluation.log'),
        filemode='w',
        level=logging.INFO,   # logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logging.info(args)
    logging.info(all_prompts)
    print(START_TIME)
    
    for task_name in path_all_prompts.keys():

        if args.tool and task_name not in ["MOL_Solubility", "MOL_Freesolv", "MOL_BBBP", "MOL_TOX"]:
            continue
            
        process(args, model, task_name, all_prompts, keys_all_prompts, path_all_prompts, par_dir, tool_all_prompts)

    
if __name__ == "__main__":
    main()
    print('Finished')
