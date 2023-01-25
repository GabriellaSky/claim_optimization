# Argument Optimization

This repository contains the code associated with the following paper:


[Claim Optimization in Computational Argumentation](https://arxiv.org/abs/2212.08913)
by Gabriella Skitalinskaya, Maximilian Spliethöver, Henning Wachsmuth

## Reproducing results

### Models & Data
To obtain access to BART checkpoints, along with BERT, SBERT and SVMRank models, please reach out to Gabriella Skitalinskaya (email can be found in paper). Let us know if you have any questions.

### Generation

        pip install torch==1.3.1
        pip install numpy==1.19.2
        pip install --editable .

In the fairseq/data folder you can put the three dataset configurations each containing train.source , train.target , val.source , val.target files.


BPE preprocessing:

          for SPLIT in train val test
          do
            for LANG in source target
            do
              python -m examples.roberta.multiprocessing_bpe_encoder \
              --encoder-json encoder.json \
              --vocab-bpe vocab.bpe \
              --inputs "/data/no_context/$SPLIT.$LANG" \
              --outputs "/data/no_context/$SPLIT.bpe.$LANG" \
              --workers 60 \
              --keep-empty;
            done
          done

Binarizing dataset:

          fairseq-preprocess \
            --source-lang "source" \
            --target-lang "target" \
            --trainpref "/data/no_context/train.bpe" \
            --validpref "/data/no_context/val.bpe" \
            --destdir "/data/no_context" \
            --workers 60 \
            --srcdict dict.txt \
            --tgtdict dict.txt;

Model training:

          TOTAL_NUM_UPDATES=20000  
          WARMUP_UPDATES=500      
          LR=3e-05
          MAX_TOKENS=1024
          UPDATE_FREQ=8
          python train.py data/no_context\
            --restore-file ../bart.large/model.pt \
            --max-tokens $MAX_TOKENS \
            --task translation \
            --source-lang source --target-lang target \
            --truncate-source \
            --truncate-target \
            --layernorm-embedding \
            --share-all-embeddings \
            --share-decoder-input-output-embed \
            --reset-optimizer --reset-dataloader --reset-meters \
            --required-batch-size-multiple 1 \
            --arch bart_large \
            --criterion label_smoothed_cross_entropy \
            --label-smoothing 0.1 \
            --dropout 0.1 --attention-dropout 0.1 \
            --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
            --clip-norm 0.1 \
            --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
            --memory-efficient-fp16 --update-freq $UPDATE_FREQ \
            --save-dir "checkpoint-opt/3types/no_context" \
            --ddp-backend=no_c10d  \
            --skip-invalid-size-inputs-valid-test \
            --max-epoch 10 \
            --find-unused-parameters;


Candidate generation:

Before running SVMRank-related experiments, follow the instructions to installing PySVMRank as described [here](https://github.com/ds4dm/PySVMRank).

          python get_candidates.py \
            --modeldir checkpoint-opt/3types/no_context \
            --datadir data/no_context \
            --outdir data/no_context


Candidate ranking:

        pip install pandas==1.1.5
        pip install scikit-learn==0.24.2
        pip install sentence-transformers==2.1.0
        pip install transformers==4.11.2
        pip install nltk==3.6.3
        pip install scipy==1.5.4
  
        python rank.py \
            --mode no_context\
            --candidates checkpoint-opt/3types/no_context/test.hypo \
            --datadir data/no_context/ \
            --svmrank models/svmrank/svmrank_sbert_model.bin \
            --fluency models/fluency/ \
            --argquality models/argument_quality/ \
            --sbertmodel models/sbert/ \
            --outfile results.csv

### Citation
If you use this corpus or code in your research, please include the following citation:
```
@misc{skitalinskaya-etal-2022-optimization,
  doi = {10.48550/ARXIV.2212.08913},
  url = {https://arxiv.org/abs/2212.08913},
  author = {Skitalinskaya, Gabriella and Spliethöver, Maximilian and Wachsmuth, Henning},
  keywords = {Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Claim Optimization in Computational Argumentation},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
