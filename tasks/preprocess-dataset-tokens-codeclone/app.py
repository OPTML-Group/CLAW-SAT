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
      # print("r_token:",r_token)
      replaceme = r_token.replace('@R_','replaceme')
      replaceme = replaceme.replace('@','')
      token = optim_tokens[r_token]
      # print(token)
      new_program = new_program.replace(replaceme, token)
      # print("new program:",new_program)
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
  src = list(filter(None, [
    normalize_subtoken(subtok) for subtok in subtokens(item[2])
  ]))
  tgt = list(filter(None, [
    normalize_subtoken(subtok) for subtok in clean_name(item[4])
  ]))
  return (
    len(src) > 0 and len(tgt) > 0,
    item[0],
    item[1],
    item[3],
    ' '.join(src),
    ' '.join(tgt),
    item[5]
  )


if __name__ == "__main__":
  print("Loading inputs...")

  has_baselines = False

  tasks = []
  for split in ["test","train","valid"]:
    # if not os.path.isfile('/mnt/inputs/{}.jsonl.gz'.format(split)):
    #     continue
    if split == 'baseline':
      has_baselines = True
    get_site_map = False
    if os.path.exists('/mnt/inputs/train_site_map.json'.format(split)): 
      with open('/mnt/inputs/train_site_map.json'.format(split), 'r') as f:
          site_map = json.load(f)
          print(len(site_map))
      get_site_map = True

    new_site_map = {}
    index_number_map = {}
    print("hello")
    for line in gzip.open('/mnt/index/train.jsonl.gz'):
      as_json = json.loads(line)
      idx = as_json['idx'].replace("\u201c","\\")
      # print(idx)
      index_number_map[idx] = as_json["sha256_hash"]
    index_lst = []
    cnt = 0
    with open('mnt/train_file/{}.txt'.format(split),'r') as f:
      line = f.readline()
      # print(line)
      # exit(0)
      while(line):
        idx1,idx2,label = line.strip().split(" ")
        # idx1 = idx1.replace("\u201c","\\")
        # idx2 = idx2.replace("\u201c","\\")
        if idx1 in index_number_map and idx2 in index_number_map:
          index_lst.append([index_number_map[idx1],index_number_map[idx2],label])
          cnt+=1
        line = f.readline()
    print("how many pairs:",cnt)
    code_info = {}
    cnt = 0
    for line in gzip.open('/mnt/inputs/train.jsonl.gz'):
      as_json = json.loads(line)
      from_file = as_json['from_file'] if 'from_file' in as_json else '{}.java'.format(as_json['sha256_hash'])
      from_file = from_file.replace('.py', '')
      from_file = from_file.replace('.java', '')
      # tasks.append((split, from_file, as_json['source_tokens'], as_json['target_tokens']))
      the_hash = as_json['sha256_hash']
      code_info[from_file] = as_json['source_tokens']
      

    cnt = 0   
    for idx1,idx2,label in index_lst:
      # print(idx1)
      # if idx1 or idx2 not in code_info:
      #   cnt+=1
    

    # print("not in the map",cnt)
    # exit(0)
      tasks.append((split,idx1,code_info[idx1],idx2,code_info[idx2],label))
      if get_site_map:
        new_site_map[idx1+idx2] = {}
        # if from_file in site_map:
        for r in site_map[idx1]:
          if site_map[idx1][r][0] == '':
            new_site_map[idx1+idx2][r] = site_map[idx1][r]
          else:
            new_site_map[idx1+idx2][r] = (' '.join([normalize_subtoken(subtok) for subtok in subtokens([site_map[idx1][r][0]])]), site_map[idx1][r][1])
    
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

  out_map['test'].write('from_file\tsrc\ttgt\tlabel\n')
  out_map['train'].write('from_file\tsrc\ttgt\tlabel\n')
  out_map['valid'].write('from_file\tsrc\ttgt\tlabel\n')

  print("  - Processing in parallel...")
  iterator =  tqdm.tqdm(
    pool.imap_unordered(process, tasks, 1000),
    desc="    - Tokenizing",
    total=len(tasks)
  )
  for good, split, from_file,from_file2, src, tgt,label in iterator:
    if not good: # Don't let length == 0 stuff slip through
      continue
    # print(tgt)
    tgt_replace_tokens = get_all_replace(tgt)
    # print("tgt:",tgt_replace_tokens)
    orig_tokens = site_map[from_file2]
    fix_original_tokens_tgt = {s:orig_tokens[s][0] for s in orig_tokens if s in tgt_replace_tokens}
    # print(orig_tokens)
    tgt = get_view(tgt,fix_original_tokens_tgt)
    from_file_name = (from_file+from_file2).replace('\0', ' ')
    src = src.replace('\0', ' ')
    tgt = tgt.replace('\0',' ')
    label = label.replace('\0',' ')
    out_map[split].write(
      '{}\t{}\t{}\t{}\n'.format(from_file_name, src, tgt,label)
    )
  print("    + Tokenizing complete")
  print("  + Done extracting tokens")