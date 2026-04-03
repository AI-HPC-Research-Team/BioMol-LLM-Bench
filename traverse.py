import os
import argparse

def add_parser():
    parser = argparse.ArgumentParser(description='data parsing')
    parser.add_argument('-root_path', type=str, default='demo_result', help='result path')
    parser.add_argument('-log_path', type=str, default='demo_parse', help='metric result')
    parser.add_argument('-tool', action="store_true", help='preditions with tools')

    args = parser.parse_args()
    return args

def traverse_levels(root_path, log_path, tool=False):

    items = os.listdir(root_path)
    
    for item in items:
        item_path = os.path.join(root_path, item)

        exps = os.listdir(item_path)
        for exp in exps:
            model = item
            logtime = exp
            if tool:
                run_command = f'python run_judgement.py -root_path {root_path} -model {model} -time {logtime} -log_path {log_path} -tool'
            else:
                run_command = f'python run_judgement.py -root_path {root_path} -model {model} -time {logtime} -log_path {log_path}'
            result = os.system(run_command)
            print(result)

if __name__ == "__main__":
    args = add_parser()
    traverse_levels(args.root_path, args.log_path, args.tool)
