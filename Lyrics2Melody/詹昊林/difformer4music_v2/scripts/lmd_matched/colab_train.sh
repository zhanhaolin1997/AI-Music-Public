#!/bin/bash

DIFFUSION_STEPS=3600

NOISE_SCHEDULE="sqrt"
NOISE_FACTOR=2

LR="5e-4"
MAX_TOKENS=20000
MAX_UPDATE=500000

DATASET="lmd_matched"
MODEL_DIR="models/${DATASET}"

mkdir -p $MODEL_DIR/tb

CUDA_VISIBLE_DEVICES=0 fairseq-train \
    datasets/data-bin/${DATASET} \
    --load-alignments \
    --tokenizer	"space" \
    --required-seq-len-multiple 12 \
    --save-dir $MODEL_DIR \
    --ddp-backend no_c10d \
    --user-dir difformer \
    --task difformer \
    --criterion nat_loss \
    --arch difformer_template2melody \
    --diffusion-steps $DIFFUSION_STEPS \
    --noise-schedule $NOISE_SCHEDULE --noise-factor $NOISE_FACTOR \
    --embed-norm --embed-norm-affine \
    --self-cond \
    --rescale-timesteps \
    --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 \
    --lr $LR --lr-scheduler inverse_sqrt \
    --warmup-updates 4000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.01 \
    --clip-norm 1 \
    --dropout 0.2 --weight-decay 0.001 \
    --encoder-embed-dim 396 --latent-dim 576 --model-dim 432\
    --encoder-attention-heads 6 --decoder-attention-heads 6 \
    --encoder-layers 6 --decoder-layers 6 \
    --max-source-positions 8192 --max-target-positions 8192 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --pred-length-offset \
    --length-loss-factor 0.2 \
    --apply-bert-init \
    --fp16 \
    --log-format 'json' --log-interval 100 \
    --fixed-validation-seed 6 \
    --decoding-steps 36 \
    --eval-bleu \
    --eval-tokenized-bleu \
    --eval-bleu-remove-bpe \
    --best-checkpoint-metric label_smoothed_cross_entropy_with_alignment \
    --validate-interval 6 \
    --maximize-best-checkpoint-metric \
    --max-tokens $MAX_TOKENS \
    --max-update $MAX_UPDATE \
    --keep-last-epochs 18 \
    --keep-best-checkpoints 12 \
    --num-workers 18 \
    2>&1 | tee $MODEL_DIR/train.log