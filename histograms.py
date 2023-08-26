from torch import optim
from ae.utils import load_dataset, save_model, iterate_batches, load_model, load_dataset_afhq
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, InvertibleAE, CholeskyAE, InvOrthogonalAE
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import gc

gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# StyleFlow dataset
nbr_of_attributes = 3
number_of_repeated_labels = 10
all_w, all_a = load_dataset_afhq(n=number_of_repeated_labels)

model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Only on gmum servers
model, optimizer = load_model('ae_models/afhq-v2/afhq_model_e100_05_04_2023_16_49_46.pch', model, optimizer)

batch_size = 32
df_predicted = []
df_actual = []
model.eval()
for w, a in iterate_batches(all_w, all_a, batch_size):
    w = w.to(device)
    a = a.to(device)

    reconstruction, code = model(w)

    df_predicted.append(code[:, :, 0, :nbr_of_attributes].reshape(-1, nbr_of_attributes).cpu().detach().numpy())
    df_actual.append(a.reshape(-1, nbr_of_attributes).cpu().detach().numpy())
    del w, a, reconstruction, code
    gc.collect()

df_predicted = np.concatenate(df_predicted)
df_actual = np.concatenate(df_actual)

df_pred = pd.DataFrame(df_predicted, columns=[str(i) for i in range(nbr_of_attributes)])
df_act = pd.DataFrame(df_actual, columns=[str(i) for i in range(nbr_of_attributes)])
print(len(df_pred), len(df_act))
# make subplots with all histograms

fig, axs = plt.subplots(nbr_of_attributes, 1, figsize=(7, 7))

# titles = ["gender", "glasses", "yaw", "pitch", "baldness", "beard", "age", "expression"]
titles = ["cat", "dog", "wild"]

for i in range(nbr_of_attributes):
    a_heights, a_bins = np.histogram(df_act[str(i)], bins=np.arange(-2, 3, 0.1))
    b_heights, b_bins = np.histogram(df_pred.iloc[df_act[df_act[str(i)] <= df_act[str(i)].mean()].index][str(i)],
                                     bins=a_bins)
    c_heights, c_bins = np.histogram(df_pred.iloc[df_act[df_act[str(i)] > df_act[str(i)].mean()].index][str(i)],
                                     bins=a_bins)
    width = (a_bins[1] - a_bins[0]) / 4

    axs[i].bar(a_bins[:-1], a_heights, width=width, facecolor='cornflowerblue', label='actual')
    axs[i].bar(b_bins[:-1] + width, b_heights, width=width, facecolor='orangered', label='pred_class_a')
    axs[i].bar(c_bins[:-1] + 2 * width, c_heights, width=width, facecolor='seagreen', label='pred_class_b')
    axs[i].legend(loc='upper right')
    axs[i].set_title(f'Attribute {titles[i]}')

plt.tight_layout()
now = datetime.now()
dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
plt.savefig(f'histograms_{dt_string}.png')
