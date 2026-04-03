import os
import pandas as pd
import re
import time
import argparse

def add_parser():
    parser = argparse.ArgumentParser(description='data integration')
    parser.add_argument('-log_path', type=str, default='demo_parse', help='metric result')
    parser.add_argument('-final_path', type=str, default='final_results', help='metric result')

    args = parser.parse_args()
    return args

def main(args):
    all_results = []
    for root, dirs, files in os.walk(args.log_path):    
        for file in files:
            fp = os.path.join(root,file)
            result = {}
            result['time'] = fp.split('/')[-2]
            result['model'] = fp.split('/')[-3]
            lines = []
            with open(fp, 'r', encoding='utf-8') as data:
                # lines = [line.strip().sp for line in data]
                for line in data:
                    content = line.split('INFO - ')[-1]
                    # print(content)
                    if content.startswith('===='):
                        task = content.split('result_')[-1].split('.')[0]
                        continue
                    
                    metric_match = re.match(r'.* - (.*): (.*)', line)
                    if metric_match and task:
                        metric_name = task + '_and_' + metric_match.group(1).strip()
                        metric_value = metric_match.group(2).strip()
                        result[metric_name] = metric_value
                    else:
                        print('here')
                        continue
                # print(result)
            
            all_results.append(result)
    
    print(all_results)
    
    df = pd.DataFrame(all_results)
    
    df = df.sort_values(by='model', ascending=False)
    
    front_columns = ['model', 'time']
    new_columns = front_columns + sorted([col for col in df.columns if col not in front_columns])
    df = df[new_columns].reset_index(drop=True)
    
    split_columns = [col.split('_and_') for col in df.columns]
    level0 = [parts[0] for parts in split_columns]
    level1 = [parts[1] if len(parts) > 1 else '' for parts in split_columns] 
    
    df.columns = pd.MultiIndex.from_arrays([level0, level1])
    
    START_TIME = time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())
    
    if not os.path.exists(args.final_path):
        os.makedirs(args.final_path)

    df.to_csv(f'./{args.final_path}/{args.log_path}_all_model_results_{START_TIME}.csv', index=False, encoding='utf-8')

if __name__ == "__main__":
    args = add_parser()
    main(args)
    print('Finished')
