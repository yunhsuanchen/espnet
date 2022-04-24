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


def append_files(df, directory):
    text_lines, scp_lines, utt2spk_lines = [], [], []

    for idx, row in df.iterrows():
        wav_file_path = row["path"]
        # add speaker-ids prefixes of utt-ids
        spk_id = row["speaker"]
        utt_id = spk_id + "_" + row["utterance"]
        transcription = row["transcription"]

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='downloads/TAT-Vol2',
                        help='path to TAT-Vol2 Samples data directory')
    parser.add_argument('--output_text', default='tailo',
                        choices = ["tailo", "tailo-num", "han-tw", "poj"],
                        help='ASR outout script')
    parser.add_argument('--speech_aug', action='store_true',
                        help="add data augmentation: pitch shift")

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
                    info = json.load(f)
                    utt_id = info["提示卡編號"] + '-' + info['句編號']
                    data_dicts.append({
                        "utterance": utt_id, # ex: 0035-5.23
                        "speaker": info["發音人"],
                        "path": os.path.join(data_dir, "condenser", speaker, utt_id + '-03.wav'),
                        "transcription": info[text_column_name]
                    })
    train_df = pd.DataFrame(data_dicts)

    # goal: CSV with utterance, speaker, filepath, transcription

    print("Appending TAT-vol2 files to train...", end="")
    append_files(train_df, "data/train")
    print("Done.")
