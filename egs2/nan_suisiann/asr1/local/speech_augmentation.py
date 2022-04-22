import pandas as pd
import os
import sys
import numpy as np
import argparse

import librosa
import soundfile as sf

def shift_pitch(data, sampling_rate, pitch_factor):
    # negative pitch factor makes the voice sound lower
    # positive pitch factor makes the voice sound higher
    return librosa.effects.pitch_shift(data, sr=sampling_rate, n_steps=pitch_factor)

# stretching the sound
def stretch(data, rate=1):
    return librosa.effects.time_stretch(data, rate=rate)

def create_augment_wav_files(df, pitch_factors, data_dir, output_dir, spk_id):
    augment_wav_output_dir = output_dir

    new_list = []
    for idx, row in df.iterrows():
        wav_file_path = os.path.join(data_dir, row["音檔"])

        utt_id = row["音檔"].split("/")[-1].split(".")[0]
        row["講者"] = spk_id

        # add original row to new dataframe
        new_list.append(row)

        #  Load the audio as a waveform `y`
        #  Store the sampling rate as `sr`
        y, sr = librosa.load(wav_file_path)

        for n_step in pitch_factors:
            # string to int
            n_step = int(n_step)

            aug_utt_id = f"{utt_id}_shift{abs(n_step)}"
            aug_spk_id = f"{spk_id}_shift{abs(n_step)}"
            out_wav_file_path = os.path.join(augment_wav_output_dir, aug_utt_id+".wav")

            shifted_y = shift_pitch(y, sr, n_step)

            # Write out audio as 24bit PCM WAV
            sf.write(out_wav_file_path, shifted_y, sr, subtype='PCM_24')

            # new row
            new_row = row.copy()
            new_row["音檔"] = os.path.join("augmented", aug_utt_id+".wav")
            new_row["講者"] = aug_spk_id

            # add row to new dataframe
            new_list.append(new_row)
    
    # save to new csv file
    new_df = pd.DataFrame(new_list)
    return new_df

def main(args):
    # downloads/0.2.1/augmented
    augment_wav_output_dir = os.path.join(args.data_dir, "augmented")

    if not os.path.isdir(augment_wav_output_dir):
        print(f"Create directory:{augment_wav_output_dir}")
        os.makedirs(augment_wav_output_dir)

    df = pd.read_csv(os.path.join(args.data_dir, "SuiSiann.csv"))
    # suisiann only has one speaker
    spk_id = "spk001"

    new_list = []
    for idx, row in df.iterrows():
        wav_file_path = os.path.join(args.data_dir, row["音檔"])

        utt_id = row["音檔"].split("/")[-1].split(".")[0]
        row["講者"] = spk_id

        # add original row to new dataframe
        new_list.append(row)

        #  Load the audio as a waveform `y`
        #  Store the sampling rate as `sr`
        y, sr = librosa.load(wav_file_path)

        for n_step in args.pitch_factors:
            # string to int
            n_step = int(n_step)

            aug_utt_id = f"{utt_id}_shift{abs(n_step)}"
            aug_spk_id = f"{spk_id}_shift{abs(n_step)}"
            out_wav_file_path = os.path.join(augment_wav_output_dir, aug_utt_id+".wav")

            shifted_y = shift_pitch(y, sr, n_step)

            # Write out audio as 24bit PCM WAV
            sf.write(out_wav_file_path, shifted_y, sr, subtype='PCM_24')

            # new row
            new_row = row.copy()
            new_row["音檔"] = os.path.join("augmented", aug_utt_id+".wav")
            new_row["講者"] = aug_spk_id

            # add row to new dataframe
            new_list.append(new_row)

    # save to new csv file
    new_df = pd.DataFrame(new_list)
    new_df.to_csv(os.path.join(args.data_dir, "SuiSiann_aug.csv"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="downloads/0.2.1",
                    help='Suisiann data directory path')
    parser.add_argument('--pitch_factors', nargs="+", default=[0],
                    help='a list of steps for pitch factors, 0 means no change in pitch')
    args = parser.parse_args()

    main(args)