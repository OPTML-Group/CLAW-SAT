import re
from typing import Optional, Tuple
import time
from subprocess import Popen, PIPE
from loguru import logger

from representjs import PACKAGE_ROOT

import torchtext
import csv
from seq2seq.dataset import SourceField, TargetField



def dispatch_to_node(node_file: str, stdin: Optional[str] = None, timeout_s: int = 5) -> Tuple[bool, str, str]:
    absolute_script_path = str((PACKAGE_ROOT / "node_src" / node_file).resolve())
    p = Popen(["timeout", timeout_s, "node", absolute_script_path], stdout=PIPE, stdin=PIPE, stderr=PIPE)
    if stdin is not None:
        p.stdin.write(stdin.encode())
    stdout, stderr = p.communicate()
    return_code = p.returncode
    if return_code != 0:
        logger.error("Got non-zero exit code {} for command {}".format(return_code, node_file))
    return (return_code == 0), stdout.decode().strip(), stderr.decode().strip()


class Timer:
    """from https://preshing.com/20110924/timing-your-code-using-pythons-with-statement/"""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start


_newline_regex = re.compile(r"\n")
_whitespace_regex = re.compile(r"[ \t\n]+")


def normalize_program(fn: str):
    if not isinstance(fn, (str, bytes)):
        logger.error(f"normalize_program got non-str: {type(fn)}, {fn}")
    fn = _newline_regex.sub(r" [EOL]", fn)
    fn = _whitespace_regex.sub(" ", fn)
    return fn


def EncodeAsIds(sp, alpha, prog):
    # Encode as ids with sentencepiece
    if alpha:
        # using subword regularization: https://arxiv.org/pdf/1804.10959.pdf
        # NOTE: what is the second argument here (-1)?
        return sp.SampleEncodeAsIds(prog, -1, alpha)

    # using the best decoding
    return sp.EncodeAsIds(prog)

def load_seq2seq_data(data_path, fields=(SourceField(), TargetField()), filter_func=lambda x: True):
    src, tgt = fields
    fields_inp = []
    with open(data_path, 'r') as f:
        first_line = f.readline()
        cols = first_line[:-1].split('\t')
        # print('COLS', cols)
        for col in cols:
            if col == 'src':
                fields_inp.append(('src', src))
            elif col == 'tgt':
                fields_inp.append(('tgt', tgt))

    data = torchtext.data.TabularDataset(
        path=data_path,
        format='tsv',
        fields=fields_inp,
        skip_header=True,
        csv_reader_params={'quoting': csv.QUOTE_NONE},
        filter_pred=filter_func
    )
    return data, src, tgt

def get_seq2seq_data_generator(data_path, input_vocab, output_vocab, batch_size, device, max_length):

    def filter_func(example):
        return len(example.src) <= max_length and len(example.tgt) <= max_length
    
    data, src, tgt = load_seq2seq_data(data_path, filter_func=filter_func)
    src.vocab = input_vocab
    tgt.vocab = output_vocab

    batch_iterator = torchtext.data.BucketIterator(
        dataset=data, batch_size=batch_size,
        sort=True, sort_within_batch=True,
        sort_key=lambda x: len(x.src),
        device=device, repeat=False
        )
    data_len = len(batch_iterator)
    print("Data length: ", data_len)
    batch_generator = batch_iterator.__iter__()
    return batch_generator, data_len