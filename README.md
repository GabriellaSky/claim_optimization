# Argument Optimization

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
          python train.py ../data/no_context\
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


Output data can be found in: 


Candidate ranking:

If you want to use svmrank do this XXX: 

Else 

Output data can be found in: 
