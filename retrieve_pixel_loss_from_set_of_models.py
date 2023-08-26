import os
import random

import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from torch import optim
from ae.utils import load_dataset, save_model, iterate_batches, load_model
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, InvertibleAE, CholeskyAE, InvOrthogonalAE, DeepAE, \
    relu_loss
import pandas as pd

import PIL.Image
import pickle
import torch
import gc

gc.collect()
device = "cuda"
print("Using device:", device)

if device != torch.device('cpu'):
    from GPUtil import showUtilization as gpu_usage
    from numba import cuda


    def free_gpu_cache():
        print("Initial GPU Usage")
        gpu_usage()

        torch.cuda.empty_cache()

        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)

        print("GPU Usage after emptying the cache")
        gpu_usage()


    free_gpu_cache()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def generate_images_in_w_space(w, batch_size=32, fp32=True):
    img = G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const', force_fp32=fp32)
    return img


def generate_images_in_w_space_v2(ws, batch_size=32, fp32=True):
    for w in ws:
        yield G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const', force_fp32=fp32)


def save_image(img, path):
    print('Generating images ...')
    imgs = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    d = 0
    for i in imgs:
        d += 1
        PIL.Image.fromarray(i.cpu().detach().numpy(), 'RGB').save(f'{path}_img_{d}.png')


# StyleGAN2-ada-pytorch
with open('ae/data/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
    G.eval()

# StyleFlow dataset
nbr_of_attributes = 8
all_w, all_a = load_dataset(keep=False, values=2, nbr_of_attributes=nbr_of_attributes)
# del all_a
# gc.collect()

# AE Model
model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
criterion_w = torch.nn.MSELoss()
criterion_a = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Only on gmum servers
# model, optimizer = load_model('ae_models/model_e1_03_03_2023_16_40_29.pch', model, optimizer)

# training params
batch_size_train = 3
batch_size_test = 2
start_epoch = 1
end_epoch = 50

lambda_a = 0.3
lambda_pixel_loss = 0.15

df = {
    'timestamp': [],
    'epoch': [],
    'test_loss': [],
    'loss_w': [],
    'loss_a': [],
    'lambda_a': [],
    'lambda_pixel_loss': [],
    'pixel_loss':[]
}


def modify_reconstruction(reconstruction, nbr_of_attributes):
    modified_reconstruction = reconstruction.clone()
    batch_size = reconstruction.shape[0]
    for i in range(batch_size):
        attribute_to_modify = random.randint(0, nbr_of_attributes)  # select attribute to modify randomly
        if reconstruction[i][attribute_to_modify] > 0.5:
            modified_reconstruction[i][attribute_to_modify] = reconstruction[i][attribute_to_modify] - round(
                random.uniform(0.5, 5), 2)
        else:
            modified_reconstruction[i][attribute_to_modify] = reconstruction[i][attribute_to_modify] + round(
                random.uniform(0.5, 5), 2)
    return modified_reconstruction


for filename in os.listdir("ae_models/pixel_loss_v2"):
    epoch = int(filename.split("_")[1][1:])
    lambda_pixel_loss = (epoch-1) * 0.05 + 0.15

    model_name = filename
    model, _ = load_model(os.path.join("ae_models/pixel_loss_v2", filename), model, optimizer)
    all_test_w, all_test_a = all_w[9900:], all_a[9900:]
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print(f"{dt_string} | Epoch Train - end: {epoch + 1}, model:{model_name}")

    # logs
    df['timestamp'].append(dt_string)
    df['epoch'].append(epoch)
    df['lambda_a'].append(lambda_a)
    df['lambda_pixel_loss'].append(lambda_pixel_loss)

    model.eval()
    running_loss, running_loss_w, running_loss_a, running_loss_pixel_loss = 0.0, 0.0, 0.0, 0.0
    count_batches = 0
    for w, a in iterate_batches(all_test_w, all_test_a, batch_size_test):
        w = w.to(device)
        a = a.to(device)

        reconstruction, code = model(w[:, 0, 0, :])
        modified_reconstruction = modify_reconstruction(reconstruction, nbr_of_attributes)

        reconstruction = reconstruction.repeat(1, 18).reshape(w.shape[0], 1, 18, 512)
        modified_reconstruction = modified_reconstruction.repeat(1, 18).reshape(w.shape[0], 1, 18, 512)
        code = code.repeat(1, 18).reshape(w.shape[0], 1, 18, 512)

        gens = generate_images_in_w_space_v2(
            (w, reconstruction, modified_reconstruction), w.shape[0])

        input_images = next(gens)
        output_images = next(gens)
        modified_images = next(gens)

        loss_w = criterion_w(input_images, output_images)
        loss_a = criterion_a(code[:, :, 0, :nbr_of_attributes].reshape(a.shape), a)
        pixel_loss = relu_loss(input_images, modified_images) / \
                     (input_images.shape[0] * input_images.shape[1] * input_images.shape[2] * input_images.shape[3])
        loss = (1 - lambda_a) * loss_w + lambda_a * loss_a + lambda_pixel_loss * pixel_loss

        running_loss += loss.item()
        running_loss_w += loss_w.item()
        running_loss_a += loss_a.item()
        running_loss_pixel_loss += pixel_loss.item()

        count_batches += 1

    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")
    print(f"{dt_string} | Epoch Test - end: {epoch + 1} | {running_loss / count_batches:.4f}")
    df['test_loss'].append(running_loss / count_batches)
    df["loss_w"].append(running_loss_w / count_batches)
    df["loss_a"].append(running_loss_a / count_batches)
    df['pixel_loss'].append(running_loss_pixel_loss / count_batches)

    # increase lambda_a
    print(f"Increased lambda_pixel_loss: {lambda_pixel_loss}")

pd.DataFrame(df).to_csv(f'ae_models/{dt_string}_loss.csv', index=False)
