import PIL.Image
import gc
import pandas as pd
import pickle
import torch
from GPUtil import showUtilization as gpu_usage
from datetime import datetime
from numba import cuda
from random import shuffle
from torch import optim

from ae.AutoEncoder import AE_single_layer, AE_multiple_layers
from ae.utils import load_dataset, save_model

gc.collect()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


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


def iterate_batches(all_w, batch_size):
    num_batches = (len(all_w) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        w = all_w[batch * batch_size: (batch + 1) * batch_size]
        yield w


def generate_images_in_w_space(w, batch_size=32):
    if torch.cuda.is_available():
        img = G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const')
    else:
        img = G.synthesis(w.reshape(batch_size, w.shape[2], w.shape[3]), noise_mode='const',
                          force_fp32=True)
    return img


def save_image(img, path):
    print('Generating images ...')
    imgs = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    d = 0
    for i in imgs:
        d += 1
        PIL.Image.fromarray(i.cpu().detach().numpy(), 'RGB').save(f'{path}/epoch{epoch + 1}_{dt_string}_{d}.png')


# StyleGAN2-ada-pytorch
with open('ae/data/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

# StyleFlow dataset
all_w, all_a = load_dataset(keep=False, values=[0] * 17)
del all_a
gc.collect()

# AE Model
model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
criterion = torch.nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# training params
batch_size = 20
start_epoch = 0
end_epoch = 100

df = {
    'timestamp': [],
    'epoch': [],
    'loss': []
}

for epoch in range(start_epoch, end_epoch):
    shuffle(all_w)
    model.train()
    running_loss = 0.0
    count_batches = 0
    for w in iterate_batches(all_w, batch_size):
        w = w.to(device)

        input_images = generate_images_in_w_space(w.to(device), batch_size).to(device)
        encoded, decoded = model(w)

        output_images = generate_images_in_w_space(encoded.to(device), batch_size).to(device)
        loss = criterion(input_images, output_images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count_batches += 1
        if (count_batches + 1) % 50 == 0:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
            print(
                f"{dt_string} | Epoch {epoch + 1} | Batch {count_batches} | Loss {running_loss / count_batches:.4f}"
            )

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H:%M:%S")
    print(f"{dt_string} | Epoch - end: {epoch + 1} | {running_loss / count_batches:.4f}")

    # logs
    df['timestamp'].append(dt_string)
    df['epoch'].append(epoch + 1)
    df['loss'].append(running_loss / count_batches)

    if (epoch + 1) % 10 == 0:
        save_image(output_images, f'ae/data/output_images')
        save_image(input_images, f'ae/data/input_images')
        save_model(f"ae_models/model_e{epoch + 1}.pch", model, optimizer)

pd.DataFrame(df).to_csv(f'ae_models/{dt_string}_loss.csv')
