import numpy as np
import pandas as pd

normalize = lambda x: (x - x.min()) / (x.max() - x.min())


def L2(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


df_ms = pd.read_csv("ae/data/input_images_ae_noglasses/labels_from_ms_08_03_2023_17_08_41.csv")
df_orig = pd.read_csv("ae/data/output_images_ae_noglasses/summary_data_08_03_2023_17_08_41.csv")
df_pred_on_output = pd.read_csv("ae/data/output_images_plugen_noglasses/predictions.csv")
df_pred_on_input = pd.read_csv("ae/data/input_images_ae_noglasses/predictions.csv")

# normalize all columns
df = df_orig.copy()
for col in df_orig.columns[1:]:
    df[col] = normalize(df[col])

# add predictions
for col in df_pred_on_output.columns[1:]:
    df["pred_" + col] = df_pred_on_output[col]

df_ms_ = df_ms.copy()
for col in df_ms.columns[1:]:
    df_ms_[col] = normalize(df_ms[col])

# add predictions
for col in df_pred_on_input.columns[1:]:
    df_ms_["pred_" + col] = df_pred_on_input[col]

# print(df_pred_on_input.columns)
print(df_ms_.columns)
print(
    len(df_ms_[(df_ms_["glasses"] == 0) & (round(df_ms_["pred_has_glasses"]) == 0)]) / len(df_ms_[(df_ms_["pred_has_glasses"] == 0)]))

print(len(df[(round(df_ms_["glasses"]) == 0)]) / len(df))

to_comapre = [_ for _ in df_pred_on_output.columns if _ != "filename" and _ != 'glasses']
print(to_comapre)
print(L2(df_pred_on_output[to_comapre].values, df_pred_on_input[to_comapre].values))
