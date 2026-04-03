# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 10:46:56 2025

@author: zhouy

refined code from 2025-10-17

usage: 
    conda activate torch0808
    python ./run_judgement.py -model InstructBioMol-base -time 2025-08-08_17_37_33

"""

import pandas as pd
import json
from tqdm import tqdm
import os
import argparse
import time
import csv
import numpy as np
import logging
import selfies as sf
from metrics import *
from utils import *

def add_parser():
    parser = argparse.ArgumentParser(description='data processing')
    parser.add_argument('-root_path', type=str, default='./demo_result', help='result path')
    parser.add_argument('-model', type=str, default='txgemma-9b-chat', help='LLM model')
    parser.add_argument('-time', type=str, default='2025-08-28_12_01_53', help='experiment time')
    parser.add_argument('-data', type=str, default='./dataset', help='benchmark data path')
    parser.add_argument('-log_path', type=str, default='./demo_parse', help='metric result')
    parser.add_argument('-tool', action="store_true", help='preditions with tools')

    args = parser.parse_args()
    return args

def load_tasks():
    with open(os.path.join(args.data, 'ans_all_prompts.json'), 'r', encoding='utf-8') as f:
        ans_all_prompts = json.load(f)
    with open(os.path.join(args.data, 'path_all_prompts.json'), 'r', encoding='utf-8') as f:
        path_all_prompts = json.load(f)
        
    return ans_all_prompts, path_all_prompts

args = add_parser()
# print(args)
REGS = ["MOL_HLGap", "MOL_Thermo", "MOL_Excited", "CI_AbAg", "PLI_BA", "PROT_Mutation", "PROT_Fitness", "PROT_Energy", "PROT_Melt", "MOL_Freesolv", "MOL_Solubility"]
CLAS_B = ["MOL_TOX", "MOL_HIV", "MOL_BBBP", "PPI_Binary"]
CLAS_M = ["PROT_EC", "PROT_Location", "General_Text", "PPI_Type"]
TEXT_A = ["PROT_Conserve", "PROT_Invfold", "MM_TI", "PROT_SSC","DDI_Interact", ]  
MOL_React = ["MOL_Resyn", "MOL_Syn"] 
GO_Sim = ["PROT_GO"]   
PROTD = ["PROT_Fold"] 


FASTA = ["PROT_Invfold"]
SELFIES = ["MOL_Resyn", "MOL_Syn"]

def main(): 
    ans_all_prompts, path_all_prompts = load_tasks()
    
    base_folder = os.path.join(args.root_path, f'{args.model}/{args.time}')
    
    log_folder = os.path.join(args.log_path, f'{args.model}/{args.time}')

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    
    logging.basicConfig(
            filename=os.path.join(log_folder, 'results.log'),
            filemode='w',
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    for root, dirs, files in os.walk(os.path.join(base_folder, args.data)):
        for file in files:
            name_index = root.split('/')[-1] + '/' + file.split('result_')[-1]
            s_key = [key for key, value in path_all_prompts.items() if name_index in str(value)]
            print(name_index, s_key, len(s_key))
            if 'PROT_Fold' in s_key:
                continue
            assert len(s_key) == 1
            ans_col = ans_all_prompts[s_key[0]][0]
            data_path = os.path.join(root, file)
            print(f'------------------ analyzing {data_path} ---------------------')
            logging.info(f'==== results of {data_path} ====')
            
            if 'General_Text' in s_key:
                input_df = pd.read_json(data_path, orient='records', lines=True)
            else:
                try:
                    input_df = pd.DataFrame(pd.read_json(data_path, orient='table')['data'][0])
                except:
                    input_df = pd.DataFrame(pd.read_json(data_path, orient='table'))

            if args.tool:
                efas, confs = extract_tool_response(input_df)
            elif args.model.startswith('InstructBioMol') or args.model.startswith('biot5'):
                efas, confs = extract_first_line(input_df)
            elif args.model.startswith('txgemma'):
                efas, confs = extract_txgemma(input_df)
            elif args.model.startswith('NatureLM'):
                efas, confs = extract_naturelm(input_df)
            else:
                efas, confs = extract_answer(input_df)
            # print(type(input_df[ans_col][0]), type(efas[0]))  
            
            try:
                if s_key[0] in FASTA and (args.model.startswith('InstructBioMol') or args.model.startswith('biot5')):
                    label = ''.join([f"<p>{aa}" for aa in input_df[ans_col]])
                elif s_key[0] in SELFIES and (args.model.startswith('InstructBioMol') or args.model.startswith('biot5')):
                    label = sf.encoder(input_df[ans_col])
                else:
                    label = input_df[ans_col]                        
                    
                if s_key[0] in CLAS_B:
                    if args.tool:
                        efas = ["1" if _x and float(_x) > 0.5 else "0" for _x in efas]  # ADMET AI prediction result format float value between [0, 1] or blank
                    metrics = eval_classify_binary(label, efas)
                elif s_key[0] in CLAS_M:
                    metrics = eval_classify_multiple(label, efas, confs)
                elif s_key[0] in REGS:
                    metrics = calculate_regression_metrics(label, efas)
                elif s_key[0] in TEXT_A:
                    metrics = eval_text(label, efas)
                elif s_key[0] in MOL_React:
                    metrics = eval_MOL_reaction(label, efas)
                    # metrics = eval_text(label, efas)
                elif s_key[0] in GO_Sim:
                    metrics = eval_GO_Sim(label, efas)
                else:
                    logging.info('unknown task.')
                    continue

                try:
                    for metric in metrics.keys():
                        score = metrics[metric]
                        logging.info(f'{metric}: {score:.4f}')     
                except:
                    for metric, score in metrics['average_scores'].items():
                        logging.info(f"{metric}: {score:.8f}")  
            
            except Exception as e:
                logging.info(f'error: {e}')

    
if __name__ == "__main__":
    main()
    print('Finished')