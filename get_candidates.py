import os
import torch
from fairseq.models.bart import BARTModel
import time
import numpy as np
import argparse


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeldir", help="folder contaning model", required=True, type=str)
    parser.add_argument("--datadir", help="folder contaning training data relative to model folder", required=True, type=str)
    parser.add_argument("--outdir", type=str, help="output dir folder to write generated candidates", required=True)
    args = parser.parse_args()

    outdir = args.outdir 
    cpdir = args.modeldir
    datadir = args.datadir
    readdir = os.path.join(cpdir, datadir)

    bart = BARTModel.from_pretrained(cpdir,checkpoint_file='checkpoint_best.pt'),data_name_or_path=datadir)

    bart.cuda()
    bart.eval()
    np.random.seed(4)
    torch.manual_seed(4)

    maxb = 256
    minb = 7
    t = 0.7
    f = open(outdir +'/'+'test.hypo','a',  encoding='utf-8')
    for line in open(readdir +'/' +'test.source', encoding = 'utf-8'):
        sline = line.strip()
        for val in [1,5,10,15,20,25,30,35,40,45]:
            with torch.no_grad():
                candidates_batch = bart.sample([sline], sampling=True, sampling_topk=val  ,temperature=t ,lenpen=1.0, max_len_b=maxb, min_len=minb, no_repeat_ngram_size=3)
                for candidate in candidates_batch:
                    candidate = candidate.replace('\n','')
                    f.write(candidate+'\t')
        f.write('\n')
            
