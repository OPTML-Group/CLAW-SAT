from enum import unique
from hashlib import new
import re
import os
import gzip
import json
import tqdm
import os.path
import multiprocessing


def camel_case_split(identifier):
  matches = re.finditer(
    '.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)',
    identifier
  )
  return [m.group(0) for m in matches]

def get_view(program, optim_tokens):
  new_program = program
  for r_token in optim_tokens:
      print("r_token:",r_token)
      replaceme = r_token.replace('@R_','replaceme')
      replaceme = replaceme.replace('@','')
      token = optim_tokens[r_token]
      print(token)
      new_program = new_program.replace(replaceme, token)
      print("new program:",new_program)
  # print(new_program)
  # print()
  assert '@R' not in new_program
  return new_program

def get_all_replace(line):
  uniques = set()
  for match in re.compile('replaceme\d+').findall(line):
    # print(match)
    uniques.add(match.strip().replace("replaceme", "@R_")+'@')
  uniques = list(uniques)
  return uniques

def subtokens(in_list):
  good_list = []
  for tok in in_list:
    for subtok in tok.replace('_', ' ').split(' '):
      if subtok.strip() != '':
        good_list.extend(camel_case_split(subtok))
  
  return good_list


def clean_name(in_list):
  return subtokens(in_list)


def normalize_subtoken(subtoken):
  normalized = re.sub(
    r'[^\x00-\x7f]', r'',  # Get rid of non-ascii
    re.sub(
      r'["\',`]', r'',     # Get rid of quotes and comma 
      re.sub(
        r'\s+', r'',       # Get rid of spaces
        subtoken.lower()
          .replace('\\\n', '')
          .replace('\\\t', '')
          .replace('\\\r', '')
      )
    )
  )

  return normalized.strip()


def process(item):
  src_whole = list(filter(None, [
    normalize_subtoken(subtok) for subtok in subtokens(item[2])
  ]))
  src = src_whole[:-6]
  tgt = src_whole[-6:]
  # tgt = list(filter(None, [
  #   normalize_subtoken(subtok) for subtok in clean_name(item[3])
  # ]))

  return (
    len(src) > 0 and len(tgt) > 0,
    item[0],
    item[1],
    ' '.join(src),
    ' '.join(tgt)
  )


if __name__ == "__main__":
  print("Loading inputs...")

  has_baselines = False

  tasks = []
  for split in ["test","train","valid"]:
    if not os.path.isfile('/mnt/inputs/{}.jsonl.gz'.format(split)):
        continue
    if split == 'baseline':
      has_baselines = True
    get_site_map = False
    if os.path.exists('/mnt/inputs/{}_site_map.json'.format(split)): 
      with open('/mnt/inputs/{}_site_map.json'.format(split), 'r') as f:
          site_map = json.load(f)
          print(len(site_map))
      get_site_map = True

    new_site_map = {}
    
    for line in gzip.open('/mnt/inputs/{}.jsonl.gz'.format(split)):
      as_json = json.loads(line)
      from_file = as_json['from_file'] if 'from_file' in as_json else '{}.java'.format(as_json['sha256_hash'])
      from_file = from_file.replace('.py', '')
      from_file = from_file.replace('.java', '')
      tasks.append((split, from_file, as_json['source_tokens'], as_json['target_tokens']))
      the_hash = as_json['sha256_hash']
      if get_site_map:
        new_site_map[from_file] = {}
        # if from_file in site_map:
        for r in site_map[from_file]:
          if site_map[from_file][r][0] == '':
            new_site_map[from_file][r] = site_map[from_file][r]
          else:
            new_site_map[from_file][r] = (' '.join([normalize_subtoken(subtok) for subtok in subtokens([site_map[from_file][r][0]])]), site_map[from_file][r][1])
    if get_site_map:
      with open('/mnt/outputs/{}_site_map.json'.format(split), 'w') as f:
          json.dump(new_site_map, f)
  
  pool = multiprocessing.Pool()
  print("  + Inputs loaded")

  out_map = {
    'test': open('/mnt/outputs/test.tsv', 'w'),
    'train': open('/mnt/outputs/train.tsv', 'w'),
    'valid': open('/mnt/outputs/valid.tsv', 'w')
  }

  if has_baselines:
    print("  + Has baselines file")
    out_map['baseline'] = open('/mnt/outputs/baseline.tsv', 'w')
    out_map['baseline'].write('from_file\tsrc\ttgt\n')
  
  print("  + Output files opened")

  out_map['test'].write('from_file\tsrc\ttgt\n')
  out_map['train'].write('from_file\tsrc\ttgt\n')
  out_map['valid'].write('from_file\tsrc\ttgt\n')

  print("  - Processing in parallel...")
  iterator =  tqdm.tqdm(
    pool.imap_unordered(process, tasks, 1000),
    desc="    - Tokenizing",
    total=len(tasks)
  )
  with open('/mnt/outputs/train_site_map.json'.format(split), 'r') as f:
    train_site_map = json.load(f)
    print(len(train_site_map))
  with open('/mnt/outputs/test_site_map.json'.format(split), 'r') as f:
    test_site_map = json.load(f)
    print(len(test_site_map))
  with open('/mnt/outputs/valid_site_map.json'.format(split), 'r') as f:
    valid_site_map = json.load(f)
    print(len(valid_site_map))  
  for good, split, from_file, src, tgt in iterator:
    if not good: # Don't let length == 0 stuff slip through
      continue
    if split == "train":
      site_map = train_site_map
    elif split == "test":
      site_map = test_site_map
    else:
      site_map = valid_site_map

    src_replace_tokens = get_all_replace(src)
    print("src:",src_replace_tokens)
    tgt_replace_tokens = get_all_replace(tgt)
    print("tgt:",tgt_replace_tokens)
    fix_tokens = [token for token in tgt_replace_tokens if token in src_replace_tokens]
    # print(new_site_map)
    print("interscation:",fix_tokens)
    orig_tokens = site_map[from_file] 
    if len(fix_tokens) > 0:
      fix_original_tokens = {s:orig_tokens[s][0] for s in orig_tokens if s in fix_tokens}
      print("fix:",fix_original_tokens)
      src = get_view(src,fix_original_tokens)
    fix_original_tokens_tgt = {s:orig_tokens[s][0] for s in orig_tokens if s in tgt_replace_tokens}
    for s in tgt_replace_tokens:
      if s not in fix_original_tokens_tgt:
        fix_original_tokens_tgt[s] = ""
    print("tgt_fix:",fix_original_tokens_tgt)
    tgt = get_view(tgt,fix_original_tokens_tgt)
    out_map[split].write(
      '{}\t{}\t{}\n'.format(from_file, src, tgt)
    )
  print("    + Tokenizing complete")
  print("  + Done extracting tokens")