import math

import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from torch import optim
from ae.utils import load_dataset, iterate_batches, load_model, load_dataset_afhq_for_faces
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, InvertibleAE, CholeskyAE, InvOrthogonalAE

import PIL.Image
import pickle
import torch
import gc

gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def generate_images_in_w_space(w, batch_size=32):
    img = G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const',
                      force_fp32=True)
    return img


def save_image(img, path):
    print('Generating images ...')
    imgs = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    d = 0
    for i in imgs:
        d += 1
        PIL.Image.fromarray(i.cpu().detach().numpy(), 'RGB').save(f'{path}_img_{d}.png')


# StyleGAN2-ada-pytorch
with open('ae/data/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to("cpu")

# StyleFlow dataset
nbr_of_attributes = 8
all_w, all_a = load_dataset_afhq_for_faces()

# del all_a
# gc.collect()

# AE Model
model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
criterion_w = torch.nn.MSELoss()
criterion_a = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Only on gmum servers
model, optimizer = load_model('ae_models/pixel_loss_v1/model_e100_25_01_2023_04_06_16.pch', model, optimizer)

# training params
batch_size = 1

model.eval()

for w, _ in iterate_batches(all_w[:10], all_a[:10], batch_size):
    w = w.to(device)

    input_images = generate_images_in_w_space(w.to("cpu"), batch_size).to(device)
    reconstruction, code = model(w)

    output_images = generate_images_in_w_space(reconstruction.to("cpu"), batch_size).to(device)

    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    save_image(input_images, f'ae/data/input_images/input_{dt_string}')
    save_image(output_images, f'ae/data/output_images/output_{dt_string}')

    del input_images, output_images, reconstruction
    gc.collect()

    for j in range(nbr_of_attributes):
        to_modify = torch.clone(code)
        x = to_modify[:, :, :, j][0][0][0].data.cpu().numpy()
        for i in np.arange(x-10, x+11, 1):
            print(f"Traverse latent space ... ({j} _ {i})")
            to_modify[:, :, :, j] = i

            # project on hypersphere with radius sqrt(512)
            x_to_project = to_modify[:, :, 0, :]

            x_to_project = x_to_project.repeat(1, 1, 18).reshape(1, 1, 18, 512)

            # use projection
            output_images = generate_images_in_w_space(model.decode(x_to_project).to("cpu"), batch_size).to(device)
            save_image(output_images,
                       f'ae/data/output_images/traverse_on_component_{dt_string}_{j}_for_value_{i}')
            del output_images
            gc.collect()
