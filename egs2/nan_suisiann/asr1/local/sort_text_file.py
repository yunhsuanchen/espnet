'''
sort lines in files
'''
import os
import sys

data_dir = sys.argv[1]  # data/train

def sort_lines(data_dir, filename):
    with open(os.path.join(data_dir,filename), "r") as f:
        lines = f.readlines()

    lines.sort()

    with open(os.path.join(data_dir,filename), "w+") as f:
        f.writelines(lines)

for filename in ["text", "wav.scp", "utt2spk"]:
    sort_lines(data_dir, filename)
