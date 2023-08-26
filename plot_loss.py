import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_results import normalize


def from_logs():
    df = {
        "epoch": [],
        "total": [],
        "rec": [],
        "attrib": [],
        "lambda_a": []
    }

    with open("ae_models/06_03_2023_09_13_36_loss", "r") as f:
        lines = f.readlines()
        lambda_a = 0.0
        for line in lines:
            if not line.startswith("Generating"):
                if line.startswith("Increased lambda_a:"):
                    lambda_a = float(line.split(":")[-1])
                else:
                    if line.split("|")[2] == ' Batch 999 ':
                        epoch = int(line.split("|")[1].split(" ")[-2])
                        total = float(line.split("|")[3].split(" ")[-2])
                        rec = float(line.split("|")[4].split(" ")[-2])
                        att = float(line.split("|")[5].split(" ")[-1])
                df["epoch"].append(epoch)
                df["total"].append(total)
                df["rec"].append(rec)
                df["attrib"].append(att)
                df["lambda_a"].append(lambda_a)

    tmp = pd.DataFrame.from_dict(df)

    # plot lines
    plt.plot(tmp["epoch"], tmp["total"], label="total")
    plt.plot(tmp["epoch"], tmp["rec"], label="rec")
    plt.plot(tmp["epoch"], tmp["attrib"], label="attrib")
    plt.plot(tmp["epoch"], tmp["lambda_a"], label="lambda")

    plt.ylim([0, 0.2])
    plt.legend()
    plt.show()


def from_file():
    df = pd.read_csv("ae_models/afhq-v3/09_04_2023_17_04_28_loss.csv", index_col=False)
    # df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df.sort_values(by=["epoch"], ascending=True)
    print(df.columns)
    # lambda_a = [_ if _ <= 0.3 else 0.3 for _ in normalize(np.arange(0, 100, 1)) ]
    # df_1 = pd.read_csv("ae_models/09_01_2023_06_09_03_loss.csv")
    # stack dataframes
    # df = pd.concat([df, df_1], ignore_index=True)
    plt.plot(df["epoch"], df["train_loss"], label="total - train")
    # plt.plot(df["epoch"], df["total_loss"], label="total loss")
    plt.plot(df["epoch"], df["loss_w"], label="rec loss")
    plt.plot(df["epoch"], df["loss_a"], label="attrib loss")
    # plt.plot(df["epoch"], df["pixel_loss"], label="pixel loss")
    # plt.plot(df["epoch"], df["lambda_pixel_loss"], label="lambda - pixel loss")
    plt.plot(df["epoch"], df["lambda_a"], label="lambda - attributes")

    plt.ylim([0, 0.5])
    plt.legend()
    plt.savefig("BestAE_approach.png")
    plt.show()


if __name__ == "__main__":
    from_file()
