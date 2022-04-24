"""
Split data to train, dev, test
"""
import sys
import os
import pandas as pd
import numpy as np

from speech_augmentation import create_augment_wav_files

train_size = 0.9
dev_size = 0.05
random_state = 11737

max_audio_length = 20

data_dir = sys.argv[1]  # downloads/0.2.1
output_text = sys.argv[2] # tailo or cmn
speech_aug = bool(sys.argv[3])
spk_id = "spk001"

if output_text == "tailo":
    text_column_name = "羅馬字"
elif output_text == "cmn":
    text_column_name = "漢字"
else:
    raise NotImplementedError

df = pd.read_csv(os.path.join(data_dir, "SuiSiann.csv"))

train_df, dev_df, test_df = \
              np.split(df.sample(frac=1, random_state=random_state), 
                       [int(train_size*len(df)), 
                       int((train_size+dev_size)*len(df))])

if speech_aug:
    aug_file = os.path.join(data_dir, "SuiSiann_aug.csv")

    if os.path.exists(aug_file):
        print("data augmentation files already exist, reading from csv file:", aug_file)
        aug_train_df = pd.read_csv(aug_file)

    else:
        pitch_factors = [-1, -2, -3]
        aug_wav_dir = os.path.join(data_dir, "augmented")

        if not os.path.isdir(aug_wav_dir):
            print(f"Create directory:{aug_wav_dir}")
            os.makedirs(aug_wav_dir)

        aug_train_df = create_augment_wav_files(train_df, pitch_factors, data_dir, aug_wav_dir, spk_id=spk_id)

        # save df to csv
        aug_train_df.to_csv(aug_file, index=False)
        print(f"Finished data augmentation, augmented wav files saved at {aug_wav_dir}")
        print("Data augmentation csv saved at", aug_file)

print(
    f"# train: {len(train_df)}, # dev:{len(dev_df)}, # test:{len(test_df)}"
)

def create_files(df, directory, train=False):
    text_lines, scp_lines, utt2spk_lines = [], [], []
    human_annotation_lines = []

    for idx, row in df.iterrows():
        # skip audio files that are too long
        if row["長短"] > max_audio_length:
            continue

        if train and speech_aug:
            spk_id = row["講者"]
        else:
            # suisiann only has one speaker
            # we are not augmenting for dev and test set
            spk_id = "spk001"

        wav_file_path = os.path.join(data_dir, row["音檔"])
        utt_id = row["音檔"].split("/")[-1].split(".")[0]
        # add speaker-ids prefixes of utt-ids
        utt_id = spk_id+utt_id
        transcription = row[text_column_name]
        transcription = transcription.replace('"', '""')
        tai_han = row["漢字"]

        text_lines.append(f"{utt_id} {transcription}\n")

        scp_lines.append(f"{utt_id} {wav_file_path}\n")

        utt2spk_lines.append(f"{utt_id} {spk_id}\n")

        human_annotation_lines.append(f"{utt_id},{wav_file_path},\"{transcription}\",\"{tai_han}\",\n")

    # sort
    text_lines.sort()
    scp_lines.sort()
    utt2spk_lines.sort()

    # write to file
    with open(f"{directory}/text", "w+") as text_file:
        text_file.writelines(text_lines)

    with open(f"{directory}/wav.scp", "w+") as scp_file:
        scp_file.writelines(scp_lines)

    with open(f"{directory}/utt2spk", "w+") as utt2spk_file:
        utt2spk_file.writelines(utt2spk_lines)
    
    with open(f"{directory}/human_annotation.csv", "w+") as f:
        f.writelines(human_annotation_lines)


print("Creating files for train...", end="")
if speech_aug:
    create_files(aug_train_df, "data/train", train=True)
else:
    create_files(train_df, "data/train", train=True)
print("Done.")

print("Creating files for dev...", end="")
create_files(dev_df, "data/dev")
print("Done.")

print("Creating files for test...", end="")
create_files(test_df, "data/test")
print("Done.")
