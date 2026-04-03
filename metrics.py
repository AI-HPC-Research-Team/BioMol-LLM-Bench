# -*- coding: utf-8 -*-
"""
Created on Tue Nov 25 11:22:09 2025

@author: zhouy
"""

import pandas as pd
import json
from tqdm import tqdm
import numpy as np
import os
import re
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
from rdkit import Chem, rdBase
rdBase.DisableLog('rdApp.warning')

from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_recall_curve, roc_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from typing import List, Dict, Any, Tuple, Optional

from difflib import SequenceMatcher
import subprocess
from Bio import pairwise2
from Bio.Align import substitution_matrices
import multiprocessing as mp
from itertools import product
import concurrent.futures
import warnings
warnings.filterwarnings('ignore')

from collections import Counter
import math

os.environ['NLTK_DATA'] = './nltk_data'
nltk.data.path = [os.environ['NLTK_DATA']]

try:
    nltk.data.find('tokenizers/punkt')
    print('punkt exists')
except LookupError:
    nltk.download('punkt', download_dir=os.environ['NLTK_DATA'])

try:
    nltk.data.find('corpora/wordnet')
    print('wordnet exists')
except LookupError:
    nltk.download('wordnet', download_dir=os.environ['NLTK_DATA'])
    
try:
    nltk.data.find('tokenizers/punkt_tab')
    print('punkt_tab exists')
except LookupError:    
    nltk.download('punkt_tab', download_dir=os.environ['NLTK_DATA'])

def advance_GO_sim(pred_gos, label_gos):
    pred_list = [p.strip() for p in pred_gos.split(',')]
    label_list = [l.strip() for l in label_gos.split(',')]
    
    correct_count = 0
    for pred in pred_list:
        if pred in label_list:
            correct_count += 1
    
    if correct_count > 0:
        return correct_count, len(label_list), 1.0
    else:    # coumpute similarity
        similarities = []
        for pred in pred_list:
            for label in label_list:
                distance = lev(label, pred)
                similarity = 1 - (distance / max(len(label), len(pred)))
                similarities.append(similarity)
        
        return 0, len(label_list), max(similarities)

def eval_GO_Sim(y_true, y_pred):

    count = 0
    acc_nums = []
    label_nums = []
    max_simis = []

    for ii, (pred_gos, label_gos) in enumerate(zip(y_pred, y_true)):
        if pred_gos is not None and pred_gos != '':
            count += 1
            acc_tmp, label_tmp, simi_tmp = advance_GO_sim(pred_gos, label_gos)
            acc_nums.append(acc_tmp)
            label_nums.append(label_tmp)
            max_simis.append(simi_tmp)
            count += 1
        else:
            continue

    # print(max_simis)
    return {
        # 'Validity': count * 1.0 / len(y_true),
        'Total': np.sum(acc_nums),
        'Avgsim': np.mean(max_simis)
    }


def get_all_permutations(lst):
    if len(lst) <= 1:
        return [lst]
    
    result = []
    for i in range(len(lst)):
        rest = lst[:i] + lst[i+1:]
        for p in get_all_permutations(rest):
            result.append([lst[i]] + p)
    return result

def calculate_similarities(list1, list2):
    maccs = []
    morgans = []
    rdks = []
    for i, (mol1, mol2) in enumerate(zip(list1, list2)):
        fp1_maccs = MACCSkeys.GenMACCSKeys(mol1)
        fp2_maccs = MACCSkeys.GenMACCSKeys(mol2)
        maccs_similarity = DataStructs.TanimotoSimilarity(fp1_maccs, fp2_maccs)
        maccs.append(round(maccs_similarity, 4))
        
        fp1_morgan = AllChem.GetMorganFingerprintAsBitVect(mol1, radius=2, nBits=2048)
        fp2_morgan = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=2048)
        morgan_similarity = DataStructs.TanimotoSimilarity(fp1_morgan, fp2_morgan)
        morgans.append(round(morgan_similarity, 4))
        
        fp1_rdkit = Chem.RDKFingerprint(mol1)
        fp2_rdkit = Chem.RDKFingerprint(mol2)
        rdk_similarity = DataStructs.TanimotoSimilarity(fp1_rdkit, fp2_rdkit)
        rdks.append(round(rdk_similarity, 4))
    
    return np.mean(maccs), np.mean(morgans), np.mean(rdks)
    
def mol_syn_valid_simi(pred_smiles, label_smiles):
    pred_list = [s.strip() for s in pred_smiles.split(',')]
    label_list = [s.strip() for s in label_smiles.split(',')]
    
    pred_mols = []
    for i, smi in enumerate(pred_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, 0, 0, 0
        else:
            pred_mols.append(mol)

    label_mols = []
    for i, smi in enumerate(label_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False, 0, 0, 0
        else:
            label_mols.append(mol)
            
    if len(pred_mols) != len(label_mols):
        return False, 0, 0, 0

    perm_pred_mols = get_all_permutations(pred_mols)

    macc = 0
    morgan = 0
    rdk = 0
    for tmp_list in perm_pred_mols:
        macc_tmp, morgan_tmp, rdk_tmp = calculate_similarities(label_mols, tmp_list)
        if macc_tmp > macc:
            macc = macc_tmp
            morgan = morgan_tmp
            rdk = rdk_tmp

    return True, macc, morgan, rdk

def eval_MOL_reaction(y_true, y_pred):

    count = 0
    all_maccs = []
    all_morgans = []
    all_rdks = []
    for ii, (pred_smiles, label_smiles) in enumerate(zip(y_pred, y_true)):
        if pred_smiles is not None and pred_smiles != '':
            is_valid, macc, morgan, rdk = mol_syn_valid_simi(pred_smiles, label_smiles)
            if is_valid:
                all_maccs.append(macc)
                all_morgans.append(morgan)
                all_rdks.append(rdk)
                count += 1

    if count == 0:
        return {
            'Validity': 0,
            'Maccs': 0,
            'RDK': 0,
            'Morgan': 0
        }
    else:
        return {
            'Validity': count * 1.0/len(y_true),
            'Maccs': np.mean(all_maccs),
            'RDK': np.mean(all_rdks),
            'Morgan': np.mean(all_morgans)
        }
    

def calib_err(confidence, correct, p='2', beta=10): 
    # beta is target bin size
    idxs = np.argsort(confidence)
    confidence = confidence[idxs]
    correct = correct[idxs]
    bins = [[i * beta, (i + 1) * beta] for i in range(len(confidence) // beta)]
    bins[-1] = [bins[-1][0], len(confidence)]

    cerr = 0
    total_examples = len(confidence)
    for i in range(len(bins) - 1):
        bin_confidence = confidence[bins[i][0]:bins[i][1]]
        bin_correct = correct[bins[i][0]:bins[i][1]]
        num_examples_in_bin = len(bin_confidence)

        if num_examples_in_bin > 0:
            difference = np.abs(np.nanmean(bin_confidence) - np.nanmean(bin_correct))

            if p == '2':
                cerr += num_examples_in_bin / total_examples * np.square(difference)
            elif p == '1':
                cerr += num_examples_in_bin / total_examples * difference
            elif p == 'infty' or p == 'infinity' or p == 'max':
                cerr = np.maximum(cerr, difference)
            else:
                assert False, "p must be '1', '2', or 'infty'"

    if p == '2':
        cerr = np.sqrt(cerr)
    return cerr 

def eval_classify_binary(y_true, y_ans, y_pred_binary=None, threshold=0.5):
    y_parse = np.zeros(np.size(y_ans), dtype=np.int64) - 1
    y_parse[np.array(y_ans, dtype='str') == '0'] = 0
    y_parse[np.array(y_ans, dtype='str') == '1'] = 1
    
    validity = int(sum(y_parse != -1)) * 1.0 / len(y_parse)
    
    y_pred = y_parse[y_parse != -1]
    y_true = y_true[y_parse != -1]
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred >= threshold).astype(int)
    
    if validity == 0:
        metrics = {
            'Validity': validity,
        }
    else:  
        metrics = {
            'AUROC': roc_auc_score(y_true, y_pred),
            'AUPRC': average_precision_score(y_true, y_pred),
            'Accuracy': accuracy_score(y_true, y_pred),
            'F1_score': f1_score(y_true, y_pred_binary),
            'Validity': validity,
        }
    
    return metrics

def eval_classify_multiple(y_true, y_pred, confs):
    mask = [x is not None and x != '' for x in y_pred]

    if all(type(item) == str for item in y_true):    # PPI type
        validity = sum([item in set(y_true) for item in y_pred]) * 1.0 / len(y_pred)
    else:
        validity = sum(mask) * 1.0 / len(y_pred)
 
    if validity == 0:
        return {
            'Validity': validity,
        }

    y_pred = [x for i, x in enumerate(y_pred) if mask[i]]   
    y_true = [str(x) for i, x in enumerate(y_true) if mask[i]]
    confs = [x for i, x in enumerate(confs) if mask[i]]
    
    correct = np.array(y_true) == np.array(y_pred)
    # print(y_true, y_pred)
    try:
        cerr = calib_err(np.array(confs), correct, p='2', beta=10)
    except:
        cerr = 100.
    
    accuracy = accuracy_score(y_true, y_pred)

    return {
        'Accuracy': accuracy,
        'CERR': cerr,
        'Validity': validity,
    }

def is_float_regex(value):

    if not isinstance(value, str) or not value.strip():
        return False

    float_pattern = r'^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$'

    special_cases = ['inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity']   #, 'nan']
    
    value_lower = value.lower().strip()
    if value_lower in special_cases:
        return True
    
    return bool(re.match(float_pattern, value))

def calculate_regression_metrics(y_true, y_ans):
      
    mask = np.zeros(np.size(y_ans), dtype=bool)
    for idx, cont in enumerate(y_ans):
        mask[idx] = is_float_regex(cont)
    
    # print(y_true, mask, y_ans)
    validity = int(sum(mask)) * 1.0 / len(mask)
    y_pred = np.array(y_ans)[mask].astype(float)
    y_true = y_true[mask]
    # print(y_pred, '\n', y_true)

    diff = y_true - y_pred

    mean_value = np.mean(diff)
    
    std_value = np.std(diff)
    
    if validity == 0.:
        return {'Validity': validity}
    else:
        metrics = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            # 'R2': r2_score(y_true, y_pred),
            # 'MSE': mean_squared_error(y_true, y_pred),
            # 'MAPE': np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100 ,
            'MEAN': mean_value,
            'STD': std_value,
            'Validity': validity,
            'Total': len(y_ans),
            'Float': len(diff),
        }
    
        return metrics

class TextGenerationEvaluator:
    def __init__(self):
        self.rouge = Rouge()
        self.smooth_fn = SmoothingFunction().method1
    
    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:

        ref_tokens = nltk.word_tokenize(reference.lower())
        cand_tokens = nltk.word_tokenize(candidate.lower())
        
        weights = {
            'BLEU-1': (1, 0, 0, 0),
            'BLEU-2': (0.5, 0.5, 0, 0),
            'BLEU-3': (0.33, 0.33, 0.33, 0),
            'BLEU-4': (0.25, 0.25, 0.25, 0.25)
        }
        
        bleu_scores = {}
        for name, weight in weights.items():
            try:
                score = sentence_bleu([ref_tokens], cand_tokens, 
                                    weights=weight, 
                                    smoothing_function=self.smooth_fn)
                bleu_scores[name] = score
            except:
                bleu_scores[name] = 0.0
        
        return bleu_scores
    
    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:

        try:
            scores = self.rouge.get_scores(candidate, reference)[0]
            return {
                'ROUGE-1': scores['rouge-1']['f'],
                'ROUGE-2': scores['rouge-2']['f'],
                'ROUGE-L': scores['rouge-l']['f']
            }
        except:
            return {'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0}
    
    def calculate_meteor(self, reference: str, candidate: str) -> float:

        try:
            ref_tokens = nltk.word_tokenize(reference.lower())
            cand_tokens = nltk.word_tokenize(candidate.lower())
            return meteor_score([ref_tokens], cand_tokens)
        except:
            return 0.0
    
    def evaluate_single_pair(self, reference: str, candidate: str) -> Dict[str, float]:

        results = {}
        
        bleu_scores = self.calculate_bleu(reference, candidate)
        results.update(bleu_scores)
        
        rouge_scores = self.calculate_rouge(reference, candidate)
        results.update(rouge_scores)
        
        results['METEOR'] = self.calculate_meteor(reference, candidate)
        
        return results
    
    def evaluate_batch(self, references: List[str], candidates: List[str]) -> Dict[str, Any]:

        assert len(references) == len(candidates), "length does not match..."

        mask = [x is not None and x != '' for x in candidates]
    
        validity = sum(mask) * 1.0 / len(references)
    
        if validity == 0:
            return {
                'Validity': validity,
            }
            
        candidates = [x for i, x in enumerate(candidates) if mask[i]]   
        references = [str(x) for i, x in enumerate(references) if mask[i]]
        
        all_scores = {
            'BLEU-1': [], 'BLEU-2': [], 'BLEU-3': [], 'BLEU-4': [],
            'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': [],
            'METEOR': [],
        }
        
        individual_results = []
        
        for ref, cand in zip(references, candidates):
            # print(ref, '\n', cand)
            scores = self.evaluate_single_pair(str(ref), str(cand))
            individual_results.append(scores)
            
            for metric in all_scores.keys():
                if metric in scores:
                    all_scores[metric].append(scores[metric])
        
        average_scores = {
            metric: np.mean(scores) if scores else 0.0 
            for metric, scores in all_scores.items()
        }
        # print(average_scores)
        return {
            'average_scores': average_scores,
            'individual_scores': individual_results,
            'all_scores': all_scores,
            'Validity': validity,
        }
        
def eval_text(references, candidates):
    evaluator = TextGenerationEvaluator()
    metrics = evaluator.evaluate_batch(references, candidates)
    return metrics

def eval_mol(generation_list,
             groundtruth_list):
    bleu_references = []
    bleu_hypotheses = []
    levs = []
    num_exact = 0
    bad_mols = 0
    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    canon_gt_smis = []
    canon_ot_smis = []
    for (generation, groundtruth) in zip(generation_list, groundtruth_list):
        if generation == '':
            bad_mols += 1
            continue

        try:
            generation = Chem.MolToSmiles(Chem.MolFromSmiles(generation))
            groundtruth = Chem.MolToSmiles(Chem.MolFromSmiles(groundtruth))

            gt_tokens = [c for c in groundtruth]
            out_tokens = [c for c in generation]

            bleu_references.append([gt_tokens])
            bleu_hypotheses.append(out_tokens)

            m_out = Chem.MolFromSmiles(generation)
            m_gt = Chem.MolFromSmiles(groundtruth)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt):
                num_exact += 1

            MACCS_sims.append(
                DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(m_gt), MACCSkeys.GenMACCSKeys(m_out),
                                                  metric=DataStructs.TanimotoSimilarity))
            RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(m_gt), Chem.RDKFingerprint(m_out),
                                                              metric=DataStructs.TanimotoSimilarity))
            morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(m_gt, 2),
                                                              AllChem.GetMorganFingerprint(m_out, 2)))
        except Exception as e:
            bad_mols += 1

        levs.append(lev(generation, groundtruth))
        canon_gt_smis.append(groundtruth)
        canon_ot_smis.append(generation)

    bleu_score = corpus_bleu(bleu_references, bleu_hypotheses)
    exact_score = num_exact / len(generation_list)
    levenshtein_score = np.mean(levs)
    validity_score = 1 - bad_mols / len(generation_list)
    maccs_sims_score = np.mean(MACCS_sims)
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    return {'BLEU': bleu_score,
            'EXACT': exact_score,
            'LEVENSHTEIN': levenshtein_score,
            'MACCS_FTS': maccs_sims_score,
            'RDK_FTS': rdk_sims_score,
            'MORGAN_FTS': morgan_sims_score,
            'VALIDITY': validity_score}
    
def all_characters_are_amino_acids(s):
    if s == '':
        return False

    amino_acids = [
        "A", "C", "D", "E", "F",
        "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R",
        "S", "T", "V", "W", "Y"
    ]
    return all(char in amino_acids for char in s)


def percentage_identity(seq1, seq2):
    # Assuming seq1 and seq2 are strings representing protein sequences
    length = min(len(seq1), len(seq2))  # Choose the minimum length
    identical_residues = sum(a == b for a, b in zip(seq1[:length], seq2[:length]))

    if length == 0:
        return 0  # Avoid division by zero if both sequences are empty

    identity = 2 * identical_residues / (len(seq1) + len(seq2))
    return identity


def similarity_matrix_score(seq1, seq2):
    substitution_matrix = substitution_matrices.load('BLOSUM45')
    score = sum(substitution_matrix.get((a, b), substitution_matrix.get((b, a))) for a, b in zip(seq1, seq2))
    score = 2 * score / (len(seq1) + len(seq2))
    return score


def alignment_similarity(seq1, seq2):
    alignments = pairwise2.align.localxx(seq1, seq2, score_only=True)
    # similarity = alignments / len(seq2)
    similarity = (alignments * 2) / (len(seq1) + len(seq2))
    return similarity

def process_pair(generation, groundtruth):
    result = {
        'identity': None,
        'align': None,
        'matrix_score': None
    }

    if not all_characters_are_amino_acids(generation):
        generation = ''

    if generation == '':
        return None

    if not all_characters_are_amino_acids(groundtruth):
        groundtruth = ''
    if groundtruth == '':
        return None

    result['identity'] = percentage_identity(generation, groundtruth)
    result['align'] = alignment_similarity(generation, groundtruth)
    result['matrix_score'] = similarity_matrix_score(generation, groundtruth)

    return result

def eval_protein(generation_list,
                 groundtruth_list,
                 cpu=8):
    bad_num = 0
    identity_list = []
    align_list = []
    matrix_score_list = []

    # TODO:add vina score calculation

    with concurrent.futures.ProcessPoolExecutor(max_workers=cpu) as executor:
        futures = [executor.submit(process_pair, gen, gt) for gen, gt in zip(generation_list, groundtruth_list)]

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is None:
                bad_num += 1
            else:
                identity_list.append(result['identity'])
                align_list.append(result['align'])
                matrix_score_list.append(result['matrix_score'])
    return {'IDENTITY': np.mean(identity_list),
            'BLOSUM': np.mean(matrix_score_list),
            'ALIGN': np.mean(align_list),
            'VALIDITY': 1 - bad_num / len(generation_list)}
    
def eval_protein_set(generation_list,
                     groundtruth_list,
                     cpu=8):
    #
    bad_num = 0

    valid_generation_list = list()
    for gen in generation_list:
        if not all_characters_are_amino_acids(gen):
            gen = ''
        if gen == '':
            bad_num += 1
            continue
        valid_generation_list.append(gen)


    pairs = list(product(groundtruth_list, valid_generation_list))

    with mp.Pool(cpu) as pool:
        identity_results = pool.starmap(percentage_identity, pairs)
        align_results = pool.starmap(alignment_similarity, pairs)
        matrix_results = pool.starmap(similarity_matrix_score, pairs)

    identity = max(identity_results)
    align = max(align_results)
    matrix_score = max(matrix_results)

    return {'IDENTITY': identity,
            'ALIGN': align,
            'BLOSUM': matrix_score,
            'VALIDITY': 1 - bad_num / len(generation_list)}


def calculate_vina_score(protein_file, ligand_file, center, box_size, exhaustiveness=8):

    cmd = [
        "vina",  
        "--receptor", protein_file,
        "--ligand", ligand_file,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--exhaustiveness", str(exhaustiveness),
        "--score_only"  
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout
        for line in output.split('\n'):
            if "Affinity" in line:
                score = float(line.split()[1])
                return score
    except subprocess.CalledProcessError as e:
        print("Vina runtime error:", e.stderr)
        return None