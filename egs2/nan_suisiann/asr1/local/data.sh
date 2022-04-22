#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
help_message=$(cat << EOF
Usage: $0

Options:
    --remove_archive (bool): true or false
      With remove_archive=True, the archives will be removed after being successfully downloaded and un-tarred.
EOF
)
SECONDS=0

# Data preparation related
data_url=https://www.dropbox.com/s/rye2sd0wo718bj5/SuiSiann-0.2.1.tar
remove_archive=true
speech_aug=true # enable speech augmentation
download_opt=
# tailo or cmn
output_text=tailo

log "$0 $*"


. ./utils/parse_options.sh

. ./db.sh
. ./path.sh
. ./cmd.sh


if [ $# -gt 1 ]; then
  log "${help_message}"
  exit 2
fi

if "$remove_archive"; then
  download_opt="--remove-archive"
fi

if [ -z "${SUISIANN}" ]; then
  log "Error: \$SUISIANN is not set in db.sh."
  exit 2
fi


log "Download data to ${SUISIANN}"
if [ ! -d "${SUISIANN}" ]; then
    mkdir -p "${SUISIANN}"
fi
# To absolute path
SUISIANN=$(cd ${SUISIANN}; pwd)

echo local/data_download.sh ${download_opt} "${SUISIANN}" "${data_url}"
local/data_download.sh ${download_opt} "${SUISIANN}" "${data_url}"

if "$speech_aug"; then
  log "Data Augmentation for speech"
  local/speech_augmentation.py --pitch -1 -2 -3
fi


log "Data Preparation"
train_dir=data/train
dev_dir=data/dev
test_dir=data/test

mkdir -p $train_dir
mkdir -p $dev_dir
mkdir -p $test_dir

# data prep for suisiann dataset
python3 local/data_split_suisiann.py ${SUISIANN}/0.2.1 $output_text

# please manually download TAT-Vol2 to ${SUISIANN} directory
# we're sorry we couldn't find a easy way to download the files via script
if [ -d "${SUISIANN}/TAT-Vol2" ]; then
    # data prep for TAT-vol2-samples
    python3 local/data_prep_tat.py ${SUISIANN}/TAT-Vol2 $output_text

    # resort the files in train
    python3 local/sort_text_file.py $train_dir
fi

for dir in $train_dir $dev_dir $test_dir; do
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

python3 local/check_train_test_duplicate.py

# validate formats
utils/validate_data_dir.sh --no-feats $train_dir
utils/validate_data_dir.sh --no-feats $dev_dir
utils/validate_data_dir.sh --no-feats $test_dir

log "Successfully finished. [elapsed=${SECONDS}s]"