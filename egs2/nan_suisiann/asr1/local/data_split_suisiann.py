"""
Split data to train, dev, test
"""
import sys
import os
import pandas as pd
import numpy as np
import argparse

from speech_augmentation import create_augment_wav_files

def create_files(args, df, directory, train=False):
    text_lines, scp_lines, utt2spk_lines = [], [], []
    human_annotation_lines = []

    for idx, row in df.iterrows():
        # skip audio files that are too long
        if row["長短"] > max_audio_length:
            continue

        if train and args.speech_aug:
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

        # skipping empty sentences
        if transcription:
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

def combine_indices_to_same_line(pred_filepath, index_filepath, gold_length):
    with open(pred_filepath) as f:
        pred_lines = f.readlines()
    with open(index_filepath) as f:
        index_lines = f.readlines()

    outputs = [""] * gold_length
    for pred_line, idx in zip(pred_lines, index_lines):
        idx = int(idx)

        outputs[idx] += pred_line.strip()

    return outputs
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='downloads/0.2.1',
                        help='path to SuiSiann data directory')
    parser.add_argument('--output_text', default='tailo',
                        choices = ["tailo", "cmn"],
                        help='ASR outout script')
    parser.add_argument('--speech_aug', action='store_true',
                        help="add data augmentation: pitch shift")

    # pseudo label related
    parser.add_argument('--pseudo_label', action='store_true',
                        help="read from MT output as label")


    args = parser.parse_args()

    train_size = 0.9
    dev_size = 0.05
    random_state = 11737

    max_audio_length = 20

    # data_dir = sys.argv[1]  # downloads/0.2.1
    # output_text = sys.argv[2] # tailo or cmn
    # speech_aug = bool(sys.argv[3])

    data_dir = args.data_dir
    output_text = args.output_text


    spk_id = "spk001"

    if output_text == "tailo":
        text_column_name = "羅馬字"
    elif output_text == "cmn":
        text_column_name = "漢字"
    else:
        raise NotImplementedError

    df = pd.read_csv(os.path.join(data_dir, "SuiSiann.csv"))

    if args.pseudo_label:
        print("Read pseudo labels from MT output")
        MT_dir = "/home/ubuntu/Taiwanese_ASR_MT/MT"
        suisiann_pred_filepath = os.path.join(MT_dir, "checkpoints/icorpus_nan_spm8000/nan_cmn/suisiann_b5.pred")
        suisiann_index_filepath = os.path.join(MT_dir, "data/suisiann_raw/nan_cmn/all.orig.nan.id")

        pseudo_lines = combine_indices_to_same_line(suisiann_pred_filepath, suisiann_index_filepath, len(df))

        # add pseudo_lines to dataframe
        assert len(df) == len(pseudo_lines), f"length of SuiSiann({len(df)}) doesn't match MT output({len(pseudo_lines)})"
        df["pseudo_cmn"] = pseudo_lines

        # changing the text column name
        text_column_name = "pseudo_cmn"

    train_df, dev_df, test_df = \
                np.split(df.sample(frac=1, random_state=random_state), 
                        [int(train_size*len(df)), 
                        int((train_size+dev_size)*len(df))])

    if args.speech_aug:
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

    print("Creating files for train...", end="")
    if args.speech_aug:
        create_files(args, aug_train_df, "data/train", train=True)
    else:
        create_files(args, train_df, "data/train", train=True)
    print("Done.")

    print("Creating files for dev...", end="")
    create_files(args, dev_df, "data/dev")
    print("Done.")

    print("Creating files for test...", end="")
    create_files(args, test_df, "data/test")
    print("Done.")
