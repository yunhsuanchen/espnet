#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
valid_set=dev
test_sets="dev test"

asr_config=conf/train_asr_sslr_terafused.yaml
inference_config=conf/decode_asr.yaml

lm_config=conf/train_lm.yaml
use_lm=false
use_wordlm=false

# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

./asr.sh                                               \
    --lang nan                                          \
    --audio_format wav                                 \
    --feats_type raw                                   \
    --token_type char                                  \
    --use_lm ${use_lm}                                 \
    --use_word_lm ${use_wordlm}                        \
    --lm_config "${lm_config}"                         \
    --asr_config "${asr_config}"                       \
    --inference_config "${inference_config}"           \
    --train_set "${train_set}"                         \
    --valid_set "${valid_set}"                         \
    --test_sets "${test_sets}"                         \
    --speed_perturb_factors "${speed_perturb_factors}" \
    --asr_speech_fold_length 512 \
    --asr_text_fold_length 150 \
    --lm_fold_length 150 \
    --lm_train_text "data/${train_set}/text" "$@" \
    --num_splits_lm 8 \
    --feats_normalize uttmvn \
    --nj 2 \
    --gpu_inference true \
    --local_data_opts "--speech_aug true --pseudo_label true" \
    --expdir exp-suisiann-tatvol2-pseudo-preinit \
    --stage 2 \
    --pretrained_model exp-suisiann-tatvol2-aug/asr_train_asr_sslr_terafused_raw_nan_char_sp/valid.acc.ave_5best.pth:::decoder \
    --ignore_init_mismatch true