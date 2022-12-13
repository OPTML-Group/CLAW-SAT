import pickle
import argparse
import tqdm
import gzip
import json
import os
import random
import dill
import re
random.seed(0)

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', action='store', dest='src_path')
    parser.add_argument('--dest_path', action='store', dest='dest_path')
    parser.add_argument('--optim_map', action='store', dest='optim_map')
    parser.add_argument('--original_map', action='store', dest='original_map')
    parser.add_argument('--idx_to_hash', action='store', dest='idx_to_hash')
    parser.add_argument('--split', action='store', dest='split')
    parser.add_argument('--transform_name', action='store', dest='transform_name')
    parser.add_argument('--vocab', action='store', dest='vocab')
    parser.add_argument('--random', action='store_true', default=False)
    parser.add_argument('--adv', action='store_true', default=False)

    args = parser.parse_args()
    return args

def generate_random_token(vocab, orig_tokens):

    all_sites = list(orig_tokens.keys())
    site = random.sample(all_sites, 1)[0]
    rand_idx = random.randrange(len(vocab.itos))
    tok = vocab.itos[rand_idx]
    while '@R_' in tok:
        rand_idx = random.randrange(len(vocab.itos))
        tok = vocab.itos[rand_idx]
    optim_tokens = orig_tokens.copy()
    optim_tokens[site] = tok
    return optim_tokens

def get_view(program, optim_tokens):
    new_program = program
    for r_token in optim_tokens:
        replaceme = r_token.replace('@R_', 'replaceme')
        replaceme = replaceme.replace('@', '')
        token = optim_tokens[r_token]
        new_program = new_program.replace(replaceme, token)
    assert '@R' not in new_program
    return new_program
def handle_replacement_tokens(line):
  ####replaceme to @R_ @
  new_line = line
  uniques = set()
  for match in re.compile('replaceme\d+').findall(line):
    uniques.add(match.strip())
  uniques = list(uniques)
  uniques.sort()
  uniques.reverse()
  for match in uniques:
    replaced = match.replace("replaceme", "@R_") + '@'
    new_line = new_line.replace(match, replaced)
  return new_line
def get_all_replacements(optim_tokens_only, orig_tokens):

    all_tokens = {}
    for tok in orig_tokens:
        if tok in optim_tokens_only:
            all_tokens[tok] = optim_tokens_only[tok]
        else:
            all_tokens[tok] = orig_tokens[tok]
    return all_tokens

def augment_data(src_path, dest_path, hash_to_idx, optim_map, orig_map, split, vocab, is_random,adv):
    
    all_views = []
    unique = 0
    data_lines = open(src_path, 'r').readlines()
    col_names = None
    for line in tqdm.tqdm(data_lines, total=len(data_lines)):
        line_list = line.strip().split('\t')
        if col_names == None:
            col_names = line_list
            continue
        program_hash, src, tgt = line_list
        program = ' '.join(['def', tgt, src])
        orig_tokens = orig_map[program_hash]
        orig_tokens = {s:orig_tokens[s][0] for s in orig_tokens}
        if len(orig_tokens) == 0: # no replacement sites
            if adv:
                all_views.append([program, program,handle_replacement_tokens(program),program_idx,program_hash])
            else:
                all_views.append([program, program])
            continue
        if is_random:
            if adv:
                if program_hash in hash_to_idx:
                    program_idx = hash_to_idx[program_hash]
            optim_tokens = generate_random_token(vocab, orig_tokens)
        else:
            if program_hash in hash_to_idx:
                program_idx = hash_to_idx[program_hash]
            else:
                continue
            if program_idx in optim_map:
                optim_tokens_only = optim_map[program_idx]
                optim_tokens = get_all_replacements(optim_tokens_only, orig_tokens)
            else:
                continue
        if adv:
            views = [get_view(program, orig_tokens), get_view(program, optim_tokens),handle_replacement_tokens(program),program_idx,program_hash]
        else:
            views = [get_view(program, orig_tokens), get_view(program, optim_tokens)]
        if len(set(views)) == 1:
            unique += 1
        #print(views[2]) 
        all_views.append(views)
       
    print('unique ', unique)
    print('total views ', len(all_views))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    with open(os.path.join(dest_path, '{}.pkl'.format(split)), 'wb') as f:
        pickle.dump(all_views, f)

    with open(os.path.join(dest_path, '{}.json'.format(split)), 'w') as f:
        json.dump(all_views, f)   

if __name__ == '__main__':

    args = parse_args()

    hash_to_idx_map = None
    optim_tokens_map = None

    if args.adv :
        idx_to_hash_map = json.load(open(args.idx_to_hash, 'r'))
        
        hash_to_idx_map = {idx_to_hash_map[i]:i for i in idx_to_hash_map}
        assert len(idx_to_hash_map) == len(hash_to_idx_map)       
    if not args.random  :

        idx_to_hash_map = json.load(open(args.idx_to_hash, 'r'))
        
        hash_to_idx_map = {idx_to_hash_map[i]:i for i in idx_to_hash_map}
        assert len(idx_to_hash_map) == len(hash_to_idx_map)

        optim_tokens_map = json.load(open(args.optim_map, 'r'))
        # optim_tokens_map = optim_tokens_map[args.transform_name]
    
    orig_tokens_map = json.load(open(args.original_map, 'r'))
    input_vocab = dill.load(open(args.vocab, 'rb'))

    augment_data(args.src_path, args.dest_path, hash_to_idx_map, optim_tokens_map, orig_tokens_map, args.split, input_vocab, args.random,args.adv)


