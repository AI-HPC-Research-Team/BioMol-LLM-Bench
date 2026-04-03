# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 11:15:43 2025

@author: zhouy

"""

from tqdm import tqdm
import os
import json
import numpy as np
from templates import *
import selfies as sf
from tooluniverse import ToolUniverse

# Initialize components
tu = ToolUniverse()
tu.load_tools()


def parse_judge_resp(input_string):
    # Initialize variables
    answer = ""
    confidence = "100"
    # Split the string into lines
    
    try:
        lines = input_string.strip().split('\n')
        # Process each line
        for line in lines:
            line = line.strip()

            if line.startswith('Answer:') or line.startswith('**Answer') or line.startswith('### Answer'):
                # print('-------------\n', line)
                answer = line.split(':')[1].strip('*').strip().split('.')[0].split('(')[0].strip('*').strip().strip('*').strip()
            elif line.startswith('Confidence:') or line.startswith('**Confidence') or line.startswith('### Confidence'):
                # print('-------------\n', line)
                confidence = line.split(':')[-1].strip('*').strip().split('(')[0].strip().strip('*')
        return answer, confidence
    except:
        # print('error parse')
        return None, None

def extract_first_line(input_df):
    efas = []
    confs = []

    for idx, row in tqdm(input_df.iterrows(), desc="Processing samples"):
        res = row['Model_Response']     
        lines = res.strip().split('\n')
        efa_ = lines[0]
        efas.append(efa_)
        confs.append(100.)
        
    return efas, confs

def extract_txgemma(input_df):
    efas = []
    confs = []

    for idx, row in tqdm(input_df.iterrows(), desc="Processing samples"):
        res = row['Model_Response']     
        efa_ = res.strip('\n').strip()
        efas.append(efa_)
        confs.append(100.)
        
    return efas, confs

def extract_tool_response(input_df):
    efas = []
    confs = []

    for idx, row in tqdm(input_df.iterrows(), desc="Processing samples"):
        res = row['Model_Response']     
        # efa_ = res.strip('\n').strip()
        efas.append(str(res))
        confs.append(100.)
        
    return efas, confs
    

def extract_naturelm(input_df):
    efas = []
    confs = []

    for idx, row in tqdm(input_df.iterrows(), desc="Processing samples"):
        res = row['Model_Response']     
        efa_ = res.split('Response:')[-1].strip().split(' ')[0]
        efas.append(efa_)
        confs.append(100.)
        
    return efas, confs
    
def extract_answer(input_df):
    efas = []
    confs = []

    for idx, row in tqdm(input_df.iterrows(), desc="Processing samples"):
        res = row['Model_Response']     
        efa_, conf_ = parse_judge_resp(res)
        efas.append(efa_)
        try:
            confs.append(float(conf_.strip('%')))
        except:
            confs.append(100.)
    return efas, confs


def gen_biot5_plus_prompt(lines):
    ptemp = BIOT5_PRE.format(lines[0])
    ptemp += lines[1].split(':')[-1]
    ptemp = ptemp + BIOT5_POST
    return ptemp

MOL_KEYS = ['smiles', 'Reactant1', 'Reactant2', 'Production', 'Drug1', 'Drug2']
PROT_KEYS = ['sequence', 'mutated_sequence', 'Seq1', 'Seq2','mut_heavy_chain_seq', 'light_chain_seq', 'X2']

def prefix_input(row, key, pre_text, pre_mol_id, post_mol_id, pre_prot_id, post_prot_id, aa_id, domain_flag, selfies_flag):
    if key in MOL_KEYS and domain_flag:
        if selfies_flag:
            prefixed = pre_mol_id + sf.encoder(row[key]) + post_mol_id
        else:
            prefixed = pre_mol_id + row[key] + post_mol_id
    elif key in PROT_KEYS and domain_flag:
        prefixed = pre_prot_id + ''.join([f"{aa_id}{aa}" for aa in row[key]]) + post_prot_id
    elif key == 'X1' and domain_flag:
        if row['TYPE'] == 'prot_prot':
            prefixed = pre_prot_id + ''.join([f"{aa_id}{aa}" for aa in row[key]]) + post_prot_id
        else:
            prefixed = pre_mol_id + sf.encoder(row[key]) + post_mol_id
    elif key == 'antigen_chains_seq' and domain_flag:
        prefixed = pre_prot_id
        for ag in row[key].keys():
            prefixed = prefixed + ''.join([f"{aa_id}{aa}" for aa in row[key][ag]])
            prefixed += ','
        prefixed = prefixed[:-1] + post_prot_id
    elif domain_flag:
        # print(type(pre_text), type(row[key]))
        prefixed = pre_text + row[key]
    else:
        prefixed = row[key]
    return prefixed

def query_model(args, model, prompt, sysp, tool_func): 
    if args.model in ['InstructBioMol-base', 'txgemma-9b-predict', 'Mistral-Nemo-Base-2407',]:
        input_ids = args.tokenizer(prompt, return_tensors="pt").to(model.device)
        if args.model== 'Mistral-Nemo-Base-2407' and 'token_type_ids' in input_ids:
            input_ids.pop('token_type_ids')
        # Generate response
        outputs = model.generate(**input_ids, max_new_tokens=args.token)
        response = args.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response[len(prompt):], prompt    
    
    elif args.model == 'Llama-3.1-8B-base':
        outputs = model(text_inputs=prompt) 
        response = outputs[0]['generated_text'][len(prompt):]
        return response, prompt
    elif args.model.startswith('NatureLM-'):
        input_ids = args.tokenizer(prompt, return_tensors="pt").to(model.device)

        ret = model.generate(
            input_ids=input_ids.input_ids,
            attention_mask=input_ids.attention_mask,
            max_new_tokens=args.token,
            do_sample=True,
            temperature=0.7,
        )
        response = args.tokenizer.decode(ret[0], skip_special_tokens=True).replace("<m>", "")

        return response, prompt 


    elif args.model.startswith('biot5-plus') or args.model.startswith('ChemLLM'):
        
        input_ids = args.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generation_config = model.generation_config
        generation_config.max_length = args.token
        generation_config.num_beams = 1

        outputs = model.generate(input_ids, generation_config=generation_config)
        
        response = args.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return response, prompt
    elif args.model.startswith('ChemLLM'):
        generation_config = GenerationConfig(
            do_sample=True,
            top_k=1,
            temperature=0.9,
            max_new_tokens=args.token,
            repetition_penalty=1.5,
            pad_token_id=args.tokenizer.eos_token_id
        )
        inputs = args.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        outputs = model.generate(inputs, generation_config=generation_config)
        response = args.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(response)
        return response, prompt
    
    elif args.model.startswith('gemma-3'): 
        if args.system:
            messages = [
                            {
                                "role": "system", "content": [{"type": "text", "text": sysp}]
                            },
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": prompt}
                                ]
                            }
                        ]
        else:
            messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
        output = model(text=messages, max_new_tokens=args.token)

        response = output[0]["generated_text"][-1]["content"]
        return response, messages
        
    else:
        if args.system:
            if 'gemma-3' in args.model:
                messages = [
                        {
                            "role": "system", "content": [{"type": "text", "text": sysp}]
                        },
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
            else:   # phi-4
                messages = [
                        {"role": "system", "content": sysp},
                        {"role": "user", "content": prompt},
                    ]
        else:
            if args.model.startswith('gemma-3'):
                messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
            elif args.model.startswith('NVIDIA-'):
                if args.think:
                    messages = [
                        {"role": "system", "content": "/think"},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": "/no_think"},
                        {"role": "user", "content": prompt},
                    ]
            else:
                messages= [
                    { "role": "user", "content": prompt},
                ]

        if args.tool:  
            [tool_name, key_name] = tool_func.split('/')
            functions = tu.get_tool_specification_by_names([tool_name], format='openai')
            tools = [{
                        "type": "function",
                        "function": functions[0]
                    }]
            response = model.chat.completions.create(
                        model=args.model,
                        messages=messages,
                        tools=tools,
                        tool_choice={"type": "function", "function": {"name": tool_name}}
                    )
            try:
                resp = response.choices[0].message
                tu_name = resp.tool_calls[0].function.name
                tool_id = resp.tool_calls[0].id
                tu_args = json.loads(resp.tool_calls[0].function.arguments or "{}")
                # print(resp)
                result_tmp = tu.run({"name": tu_name, "arguments": tu_args})
                # print(result_tmp)
                result = list(result_tmp.values())[0][key_name]
                return result, messages
            except:
                return response.choices[0].message.content, messages

        else:
            if 'Qwen' in args.model:
                chat_prompt = args.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True , enable_thinking=args.think)
            else:
                chat_prompt = args.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            
            inputs = args.tokenizer.encode(chat_prompt, add_special_tokens=False, return_tensors="pt")
            outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=args.token)
            response = args.tokenizer.decode(outputs[0, len(inputs[0]):], skip_special_tokens=True)
            return response, messages


def query_api(args, client, prompt, sysp, tool_func):
    try:           
        if args.system:
            messages=[
                    {"role": "system", "content": sysp},
                    {"role": "user", "content": prompt},
                ]
        else:
            messages=[
                    {"role": "user", "content": prompt},
                ]
            
        if args.tool:
            [tool_name, key_name] = tool_func.split('/')
            # print(tool_name, key_name)
            # print(tool_func.split('/'))
            functions = tu.get_tool_specification_by_names([tool_name], format='openai')
            tools = [{
                        "type": "function",
                        "function": functions[0]
                    }]
            model_name = client.models.list().data[0].id   # args.model
            if '/' in args.model: 
                response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                            tool_choice={"type": "function", "function": {"name": tool_name}},
                            extra_body={}
                        )
            else:
                response = client.chat.completions.create(
                            model=model_name,
                            messages=messages,
                            tools=tools,
                            tool_choice={"type": "function", "function": {"name": tool_name}}
                        )
            try:
                resp = response.choices[0].message
                tu_name = resp.tool_calls[0].function.name
                tool_id = resp.tool_calls[0].id
                tu_args = json.loads(resp.tool_calls[0].function.arguments or "{}")
                # print(resp)
                result_tmp = tu.run({"name": tu_name, "arguments": tu_args})
                # print(result_tmp)
                result = list(result_tmp.values())[0][key_name]
                return result, messages
            except:
                return response.choices[0].message.content, messages
            
        else: 
            if '/' in args.model:       # openrouter model
                if args.system:
                    messages=[
                            {
                            "role": "user",
                            "content": sysp + '\n' + prompt
                            }]
                else:
                    if 'gpt' in args.model:
                        messages=[
                                {
                                  "role": "user",
                                  "content": [
                                    {
                                      "type": "text",
                                      "text": prompt
                                    },
                                  ]
                                }
                              ]
                    else:
                        messages=[
                                {
                                "role": "user",
                                "content": prompt
                                }]

                response = client.chat.completions.create(
                    extra_body={},
                    model=args.model,
                    messages = messages,
                    max_tokens=args.token,
                )
                
            else:
                if args.system:
                    messages=[
                            {"role": "system", "content": sysp},
                            {"role": "user", "content": prompt},
                        ]
                else:
                    messages=[
                            {"role": "user", "content": prompt}
                        ]
                response = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    stream=False
                )
            return response.choices[0].message.content, messages

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
