import matplotlib.pyplot as plt
import pandas as pd

import os
import glob

# read png files
png_dir = 'ae\\data\\output_images\\'
input_dir = 'ae\\data\\input_images\\'
# png_dir = 'ae/data/output_images/' # linux

# get all files from directory with .png extension
input_files = glob.glob(input_dir + '*.png')
input_files.sort()

for i in input_files:
    to_process = "_".join(i.split("\\")[-1].split("_")[1:-2])
    print(to_process)

    png_files = glob.glob(png_dir + '*.png')
    png_files.sort()
    # only files matching the pattern "frame_*.png"
    only_output = [file for file in png_files if file.startswith(png_dir + f'output_{to_process}_img_1')]

    # traverse_on_component_26-10-2022 10/21/05_0_for_value_-0.10000000000000009_img_1.png
    traverse = [file for file in png_files if file.startswith(png_dir + f'traverse_on_component_{to_process}')]

    mapping = {
        "name": [],
        "dim": [],
        "value": []
    }
    for i in traverse:
        mapping["name"].append(i)
        mapping["dim"].append(i.split("_")[-6])
        mapping["value"].append(i.split("_")[-3])

    df = pd.DataFrame(mapping)
    df["dim"] = df["dim"].astype(int)
    df["value"] = df["value"].astype(float)
    df = df.sort_values(by=["dim", "value"]).reset_index(drop=True)
    print(df)

    os.mkdir(f"output_{to_process}")

    titles = ["gender", "glasses", "beard", "baldness", "expression", "age", "yaw", "pitch"]
    # titles = ["cat", "dog", "wild"]

    # plot traverse grid
    for i in range(len(titles)):
        subplots = len(df[df['dim'] == i])
        fig, axs = plt.subplots(1, subplots + 1, figsize=(len(titles)*2, 2))

        for j in range(subplots + 1):
            if j == 0:
                axs[j].imshow(plt.imread(only_output[0]))
                axs[j].set_title("Original")
                axs[j].axis("off")
            else:
                axs[j].imshow(plt.imread(df["name"][i * subplots + j - 1]))
                # axs[j].set_title(f"{round(df['value'][i * subplots + j - 1], 2)}_{df['dim'][i * subplots + j - 1]}")
                axs[j].axis("off")

        plt.tight_layout()
        plt.savefig(f"output_{to_process}/traverse_{str(titles[i])}.png")
