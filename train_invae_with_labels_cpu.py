import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from torch import optim

from ae.utils import load_dataset, save_model, iterate_batches, load_model
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, AE_single_layer_prelu
import pandas as pd

import PIL.Image
import pickle
import torch
import gc

import FrEIA.framework as Ff
import FrEIA.modules as Fm

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


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


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
all_w, all_a = load_dataset(keep=False, values=2, nbr_of_attributes=nbr_of_attributes)

# del all_a
# gc.collect()

# AE Model
# model = AE_single_layer(input_shape=512, hidden_dim=512).to(device)

in1 = Ff.InputNode(512, name='Input')  # 1D vector
enc_1 = Ff.Node(in1, Fm.InvAutoFC, {}, name='InvAutoFC_Enc_1')
leaky_relu_1 = Ff.Node(enc_1, Fm.InvAutoAct, {'slope_init': 0.9, 'learnable': True}, name='LeakyReLU_1')
enc_2 = Ff.Node(leaky_relu_1, Fm.InvAutoFC, {}, name='InvAutoFC_Enc_2')
leaky_relu_2 = Ff.Node(enc_2, Fm.InvAutoAct, {'slope_init': 0.9, 'learnable': True}, name='LeakyReLU_2')
dec_1 = Ff.Node(leaky_relu_2, Fm.InvAutoFC, {}, name='InvAutoFC_Dec_1')
leaky_relu_3 = Ff.Node(dec_1, Fm.InvAutoAct, {'slope_init': 0.9, 'learnable': True}, name='LeakyReLU_2')
dec_2 = Ff.Node(leaky_relu_3, Fm.InvAutoFC, {}, name='InvAutoFC_Dec_2')
output1 = Ff.OutputNode(dec_2, name='Output')
output2 = Ff.OutputNode(enc_2, name='Output2')

model = Ff.GraphINN([in1, enc_1, leaky_relu_1, enc_2, leaky_relu_2, dec_1, leaky_relu_3, dec_2, output1, output2],
                    verbose=True)
criterion_w = torch.nn.MSELoss()
criterion_a = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Only on gmum servers
# model, optimizer = load_model('ae_models/model_e10.pch', model, optimizer)

# training params
batch_size = 2
start_epoch = 0
end_epoch = 10

linear_scale = normalize(np.arange(start_epoch, end_epoch + 1, 1))
lambda_a = 0.1

df = {
    'timestamp': [],
    'epoch': [],
    'loss': [],
    'loss_w': [],
    'loss_a': []
}

for epoch in range(start_epoch, end_epoch):
    all_w_tmp, all_a_tmp = shuffle(all_w, all_a)
    model.train()
    running_loss, running_loss_w, running_loss_a = 0.0, 0.0, 0.0
    count_batches = 0
    for w, a in iterate_batches(all_w_tmp[:10], all_a_tmp[:10], batch_size):
        w = w.to(device)
        a = a.to(device)

        input_images = generate_images_in_w_space(w.to("cpu"), batch_size).to(device)
        (reconstruction, code), _ = model(w, jac=False)

        output_images = generate_images_in_w_space(reconstruction.to("cpu"), batch_size).to(device)
        loss_w = criterion_w(input_images, output_images)
        loss_a = criterion_a(code[:, :, 0, :nbr_of_attributes].reshape(a.shape), a)
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
            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
            print(
                f"{dt_string} | Epoch {epoch + 1} | Batch {count_batches} "
                f"| Total Loss {running_loss / count_batches:.4f} "
                f"| Loss W {running_loss_w / count_batches:.4f} "
                f"| Loss A {running_loss_a / count_batches:.4f}")

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print(f"{dt_string} | Epoch - end: {epoch + 1} | {running_loss / count_batches:.4f}")

    # logs
    df['timestamp'].append(dt_string)
    df['epoch'].append(epoch + 1)
    df['loss'].append(running_loss / count_batches)
    df["loss_w"].append(running_loss_w / count_batches)
    df["loss_a"].append(running_loss_a / count_batches)

    if (epoch + 1) % 1 == 0:
        save_image(output_images, f'ae/data/output_images/epoch{epoch + 1}_{dt_string}')
        save_image(input_images, f'ae/data/input_images/epoch{epoch + 1}_{dt_string}')
        save_model(f"ae_models/model_e{epoch + 1}_single_{dt_string}.pch", model, optimizer)

        # increase lambda_a
        if lambda_a + linear_scale[epoch + 1] < 1:
            lambda_a += linear_scale[epoch + 1]
            print(f"Increased lambda_a: {lambda_a}")

    del input_images, output_images, reconstruction, code, w, a
    gc.collect()

pd.DataFrame(df).to_csv(f'ae_models/{dt_string}_loss.csv')
