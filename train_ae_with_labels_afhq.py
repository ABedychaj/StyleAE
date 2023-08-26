import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from torch import optim
from ae.utils import load_dataset, save_model, iterate_batches, load_model, load_dataset_afhq
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, InvertibleAE, CholeskyAE, InvOrthogonalAE, DeepAE, \
    relu_loss, loss_on_animal_faces_2, loss_on_animal_faces_1, loss_on_animal_faces_3
import pandas as pd

import PIL.Image
import pickle
import torch
import gc

gc.collect()
device = 'cuda'
print('Using device:', device)

if device != torch.device('cpu'):
    from GPUtil import showUtilization as gpu_usage
    from numba import cuda


    def free_gpu_cache():
        print('Initial GPU Usage')
        gpu_usage()

        torch.cuda.empty_cache()

        cuda.select_device(0)
        cuda.close()
        cuda.select_device(0)

        print('GPU Usage after emptying the cache')
        gpu_usage()


    free_gpu_cache()


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def generate_images_in_w_space(w, batch_size=32, fp32=True):
    img = G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const', force_fp32=fp32)
    return img


def save_image(img, path):
    print('Generating images ...')
    imgs = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    d = 0
    for i in imgs:
        d += 1
        PIL.Image.fromarray(i.cpu().detach().numpy(), 'RGB').save(f'{path}_img_{d}.png')


# StyleGAN2-ada-pytorch
with open('ae/data/afhq/stylegan2-afhqv2-512x512.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
    G.eval()

# StyleFlow dataset
nbr_of_attributes = 3
number_of_repeated_labels = 50
all_w, all_a = load_dataset_afhq(n=number_of_repeated_labels)

# AE Model
model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
criterion_w = torch.nn.MSELoss()
criterion_a = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Only on gmum servers
model, optimizer = load_model('ae_models/afhq_retrained_model_e86_24_04_2023_09_36_01.pch', model, optimizer)

# training params
batch_size_train = 10
start_epoch = 86
end_epoch = 100

# linear_scale = normalize(np.arange(0, end_epoch + 1, 1))
# lambda_a = linear_scale[start_epoch + 1]
lambda_a = 0.3

df = {
    'timestamp': [],
    'epoch': [],
    'train_loss': [],
    'loss_w': [],
    'loss_a': [],
    'lambda_a': []
}

for epoch in range(start_epoch, end_epoch):
    # train loop
    model.train()
    running_loss, running_loss_w, running_loss_a = 0.0, 0.0, 0.0
    count_batches = 0
    all_w, all_a = shuffle(all_w, all_a)

    for w, a in iterate_batches(all_w, all_a, batch_size_train):
        w = w.to(device)
        a = a.to(device)

        input_images = generate_images_in_w_space(w, w.shape[0])  # w.shape[0] is batch_size, beside last batch

        reconstruction, code = model(w[:, 0, 0, :])

        reconstruction = reconstruction.repeat(1, 16).reshape(w.shape[0], 1, 16, 512)
        code = code.repeat(1, 16).reshape(w.shape[0], 1, 16, 512)

        output_images = generate_images_in_w_space(reconstruction, w.shape[0])
        loss_w = criterion_w(input_images, output_images)
        # loss_a = criterion_a(code[:, :, 0, :nbr_of_attributes * number_of_repeated_labels].reshape(a.shape), a)

        # custom cost function to punish <1
        loss_a = loss_on_animal_faces_3(code[:, :, 0, :nbr_of_attributes * number_of_repeated_labels].reshape(a.shape), a)

        loss = (1 - lambda_a) * loss_w + lambda_a * loss_a

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_loss_w += loss_w.item()
        running_loss_a += loss_a.item()

        count_batches += 1
        if (count_batches + 1) % 10 == 0:
            now = datetime.now()
            dt_string = now.strftime('%d_%m_%Y_%H_%M_%S')
            print(
                f'{dt_string} | Epoch {epoch + 1} | Batch {count_batches} '
                f'| Total Loss {running_loss / count_batches:.4f} '
                f'| Loss W {running_loss_w / count_batches:.4f} '
                f'| Loss A {running_loss_a / count_batches:.4f}')

    now = datetime.now()
    dt_string = now.strftime('%d_%m_%Y_%H_%M_%S')
    print(f'{dt_string} | Epoch Train - end: {epoch + 1} | {running_loss / count_batches:.4f}')

    # logs
    df['timestamp'].append(dt_string)
    df['epoch'].append(epoch + 1)
    df['train_loss'].append(running_loss / count_batches)
    df['loss_w'].append(running_loss_w / count_batches)
    df['loss_a'].append(running_loss_a / count_batches)
    df['lambda_a'].append(lambda_a)

    if (epoch + 1) % 1 == 0:
        save_image(output_images, f'ae/data/output_images/afhq_epoch{epoch + 1}_{dt_string}')
        save_image(input_images, f'ae/data/input_images/afhq_epoch{epoch + 1}_{dt_string}')
        save_model(f'ae_models/afhq_retrained_model_e{epoch + 1}_{dt_string}.pch', model, optimizer)

    # increase lambda_a
    # if linear_scale[epoch + 1] <= 0.3:
    #     lambda_a = linear_scale[epoch + 1]
    #     print(f'Increased lambda_a: {lambda_a}')

    del input_images, output_images, reconstruction, code, w, a, \
        running_loss_a, running_loss_w, running_loss, loss_w, loss_a, loss, count_batches
    gc.collect()
    torch.cuda.empty_cache()

pd.DataFrame(df).to_csv(f'ae_models/{dt_string}_loss.csv')
