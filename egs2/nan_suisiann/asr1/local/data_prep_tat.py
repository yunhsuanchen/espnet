"""
Parse data for TAT-vol2
directly goes to training data
"""
import sys
import os
import pandas as pd
import numpy as np
import json
import argparse
from collections import defaultdict

def append_files(df, directory, args):
    if args.pseudo_label:
        text_column_name = "pseudo_cmn"
    else:
        text_column_name = "transcription"

    text_lines, scp_lines, utt2spk_lines = [], [], []

    for idx, row in df.iterrows():
        wav_file_path = row["path"]
        # add speaker-ids prefixes of utt-ids
        spk_id = row["speaker"]
        utt_id = spk_id + "_" + row["utterance"]
        transcription = row[text_column_name]

        # skipping empty sentences
        if transcription:
            # utterance id and transcription
            text_lines.append(f"{utt_id} {transcription}\n")
            # utterance id and wave file path 
            scp_lines.append(f"{utt_id} {wav_file_path}\n")
            # utterance id and speaker id
            utt2spk_lines.append(f"{utt_id} {spk_id}\n")

    # sort
    text_lines.sort()
    scp_lines.sort()
    utt2spk_lines.sort()

    # write to file
    with open(f"{directory}/text", "a+") as text_file:
        text_file.writelines(text_lines)

    with open(f"{directory}/wav.scp", "a+") as scp_file:
        scp_file.writelines(scp_lines)

    with open(f"{directory}/utt2spk", "a+") as utt2spk_file:
        utt2spk_file.writelines(utt2spk_lines)

def combine_indices_to_same_line(pred_filepath, index_filepath):
    with open(pred_filepath) as f:
        pred_lines = f.readlines()
    with open(index_filepath) as f:
        index_lines = f.readlines()

    outputs = defaultdict(str)
    for pred_line, idx in zip(pred_lines, index_lines):
        idx = idx.strip()
        outputs[idx] += pred_line.strip()

    return outputs

def add_pseudo_lines_to_df(df, pseudo_lines):
    for idx, row in df.iterrows():
        new_line = pseudo_lines[row["index"]]
        df.at[idx, 'pseudo_cmn'] = new_line
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='downloads/TAT-Vol2',
                        help='path to TAT-Vol2 Samples data directory')
    parser.add_argument('--output_text', default='tailo',
                        choices = ["tailo", "tailo-num", "han-tw", "poj"],
                        help='ASR outout script')

    # pseudo label related
    parser.add_argument('--pseudo_label', action='store_true',
                        help="read from MT output as label")


    args = parser.parse_args()

    data_dir = args.data_dir  # downloads/TAT-Vol2
    output_text = args.output_text
    # 漢羅台文 (Taiwanese Han characters) 台羅 (tailo) 台羅數字調 (tailo, number tones) 白話字 (POJ)

    text_column_name = {
        "han-tw": "漢羅台文",
        "tailo": "台羅",
        "tailo-num": "台羅數字調",
        "poj": "白話字"
    }[output_text]

    data_dicts = []
    for speaker in os.listdir(os.path.join(data_dir, "json")):
        # ignore .DS_Store
        if os.path.isdir(os.path.join(data_dir, "json", speaker)):
            for utterance_f in os.listdir(os.path.join(data_dir, "json", speaker)):
                with open(os.path.join(data_dir, "json", speaker, utterance_f)) as f:
                    # adding index for MT mapping
                    index = f"{speaker}/{utterance_f}"

                    info = json.load(f)
                    utt_id = info["提示卡編號"] + '-' + info['句編號']
                    data_dicts.append({
                        "index": index, # this is for MT mapping. ex: TS_TSM0020/0035-6.13.json
                        "utterance": utt_id, # ex: 0035-5.23
                        "speaker": info["發音人"],
                        "path": os.path.join(data_dir, "condenser", speaker, utt_id + '-03.wav'),
                        "transcription": info[text_column_name],
                        "pseudo_cmn": "",
                    })
    train_df = pd.DataFrame(data_dicts)
    # goal: CSV with utterance, speaker, filepath, transcription

    if args.pseudo_label:
        MT_dir = "/home/ubuntu/Taiwanese_ASR_MT/MT"
        tat_pred_filepath = os.path.join(MT_dir, "checkpoints/icorpus_nan_spm8000/nan_cmn/tatvol2_b5.pred")
        tat_index_filepath = os.path.join(MT_dir, "data/suisiann_raw/nan_cmn/tatvol2.orig.nan.id")

        # this returns a map: filename --> 'pred line'
        # e.g., 'TS_TSM0020/0035-6.14.json' --> 'line'
        pseudo_lines = combine_indices_to_same_line(tat_pred_filepath, tat_index_filepath)

        # add pseudo_lines to dataframe
        train_df = add_pseudo_lines_to_df(train_df, pseudo_lines)

        # # add pseudo_lines to dataframe
        # assert len(df) == len(pseudo_lines), f"length of TAT-Vol2 Samples({len(train_df)}) doesn't match MT output({len(pseudo_lines)})"
        # # df["pseudo_cmn"] = pseudo_lines


    print("Appending TAT-vol2 files to train...", end="")
    append_files(train_df, "data/train", args)
    print("Done.")
