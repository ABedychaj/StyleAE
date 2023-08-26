from datetime import datetime

import PIL.Image
import pickle
import torch
import gc

from styleflow.NICE import NiceFlow
from styleflow.utils import load_dataset

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
    img = G.synthesis(w.reshape(batch_size, w.shape[1], w.shape[2]), noise_mode='const',
                      force_fp32=True)
    return img


def eval_attribute_change(prior, w, a, change, steps, features):
    a_source = a.clone()
    a_target = a_source.clone()
    if not isinstance(change, list):
        attr, val_source, val_target = change
        a_source[:, attr] = val_source
        a_target[:, attr] = val_target
    else:
        for attr, val_source, val_target in change:
            a_source[:, attr] = val_source
            a_target[:, attr] = val_target

    imgs, ws = [], [w.clone()]
    z = prior(w)[0]

    for step in range(steps):
        a = torch.lerp(a_source, a_target, step / (steps - 1))
        a = a.squeeze(-1).squeeze(-1)

        z[:, :, 0:features] = a[:, 9: 9 + features]
        w = prior.inv_flow(z)

        img = generate_images_in_w_space(w.cpu(), 1)
        imgs += [img]
        ws += [w.clone()]

    return imgs, ws


def eval_single_change(prior, w, a, change, features):
    if isinstance(change, tuple):
        attr, val = change
    else:
        attr = None

    batch_size = min(10, w.shape[0])
    assert w.shape[0] % batch_size == 0

    for i in range(0, len(w), batch_size):
        print(f"{i}/{len(w)}")
        curr_w, curr_a = w[i: i + batch_size].to(device), a[i: i + batch_size].to(
            device
        )
        new_a = curr_a.clone()
        if attr is not None:
            new_a[:, attr] = val

        z = prior(curr_w)[0]
        if attr is not None:
            for j in range(18):
                z[:, j, 0:features] = new_a[:, 9: 9 + features, 0]
        curr_w = prior.inv_flow(z)

        imgs = generate_images_in_w_space(curr_w.cpu(), z.shape[0])
        for j in range(len(imgs)):
            dt_string = datetime.now().strftime("%H-%M-%S")
            save_image(torch.tensor(imgs[j]).reshape(1, imgs[j].shape[0], imgs[j].shape[1], imgs[j].shape[2]),
                       f"ae/data/images_plugen/{dt_string}_{i + j:04}")


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
all_w, all_a = load_dataset(keep=False, values=[0] * 17)

layers = "-".join(["512"] * 4)
model = NiceFlow(input_dim=512, n_layers=4, n_couplings=4, hidden_dim=512)
model.load_state_dict(torch.load(f"stylegan_plugen.pch")["model"])

all_test_w, all_test_a = all_w[9900:], all_a[9900:]
batch_size_test = 1
count_batches = 0

dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
zero_padding = torch.zeros(1, 18, 1).to(device)
data = []

# training params
batch_size = 1

model.eval()

abso = 1
experiments = [(i, -abso, abso) for i in range(9, 17)]
steps = 5
features = 8

ws = []
changes = [(9, -1), (9, -0.75), (9, -0.5), (9, -0.25), (9, 0.25), (9, 0.5), (9, 0.75), (9, 1),
           (10, -1), (10, -0.75), (10, -0.5), (10, -0.25), (10, 0.25), (10, 0.5), (10, 0.75), (10, 1),
           (11, -1), (11, -0.75), (11, -0.5), (11, -0.25), (11, 0.25), (11, 0.5), (11, 0.75), (11, 1),
           (12, -1), (12, -0.75), (12, -0.5), (12, -0.25), (12, 0.25), (12, 0.5), (12, 0.75), (12, 1),
           (13, -1), (13, -0.75), (13, -0.5), (13, -0.25), (13, 0.25), (13, 0.5), (13, 0.75), (13, 1),
           (14, -1), (14, -0.75), (14, -0.5), (14, -0.25), (14, 0.25), (14, 0.5), (14, 0.75), (14, 1),
           (15, -1), (15, -0.75), (15, -0.5), (15, -0.25), (15, 0.25), (15, 0.5), (15, 0.75), (15, 1),
           (16, -1), (16, -0.75), (16, -0.5), (16, -0.25), (16, 0.25), (16, 0.5), (16, 0.75), (16, 1)]
N = 10
for change in changes:
    eval_single_change(model, all_w[0:N, 0], all_a[0:N], change, features)
