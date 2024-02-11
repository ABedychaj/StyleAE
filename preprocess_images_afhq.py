import PIL.Image
import copy
import gc
import numpy as np
import os
import pickle
import torch
import torch.nn.functional as F
from GPUtil import showUtilization as gpu_usage
from numba import cuda
from time import perf_counter

import dnnlib

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
        PIL.Image.fromarray(i.cpu().detach().numpy(), 'RGB').save(f'./{path}.png')


def project(
        G,
        vgg16,
        target: torch.Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device
):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device)  # type: ignore

    # Compute w stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name}

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    target_features = vgg16(target_images, resize_images=False, return_lpips=True)

    w_opt = torch.tensor(w_avg, dtype=torch.float32, device=device, requires_grad=True)  # pylint: disable=not-callable
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Features for synth images.
        synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step + 1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    closest_example = w_opt.clone().detach()[0]
    return closest_example.repeat([1, G.mapping.num_ws, 1])


def prepare_image(img):
    # Prepare image
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(img, dtype=np.uint8)
    return target_uint8


def load_images_from_folder(folder, G, vgg16):
    ws = []
    labels = []
    filenames = []
    t = 0
    start_time = perf_counter()
    for filename in os.listdir(folder):
        img = PIL.Image.open(os.path.join(folder, filename)).convert('RGB')

        if img is not None:
            filenames.append(filename)

            labels.append(filename.split('_')[1])

            target_uint8 = prepare_image(img)

            w = project(G, vgg16, torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), device=device,
                        num_steps=500)
            ws.append(w[0, 0, :])

            t += 1
            if t % 10 == 0:
                print(f'Loaded {t} images from {folder}...')
                print(f'Loaded {t} images from {folder} in {perf_counter() - start_time:.2f} sec')
                start_time = perf_counter()
    return filenames, labels, ws


# StyleGAN2-ada-pytorch
with open('ae/data/afhq/stylegan2-afhqv2-512x512.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)

# Load VGG16 feature detector.
url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
with dnnlib.util.open_url(url) as f:
    vgg16 = torch.jit.load(f).eval().to(device)

# Open and load images from ae/data/afhq/train
dog_path = 'ae/data/afhq/test/dog'
cat_path = 'ae/data/afhq/test/cat'
wild_path = 'ae/data/afhq/test/wild'

fs, ls, ws = load_images_from_folder(dog_path, G, vgg16)

data = {'filenames': fs, 'labels': ls, 'w': ws}
with open('ae/data/afhq/test/dog_w.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

fs, ls, ws = load_images_from_folder(cat_path, G, vgg16)

data = {'filenames': fs, 'labels': ls, 'w': ws}
with open('ae/data/afhq/test/cat_w.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

fs, ls, ws = load_images_from_folder(wild_path, G, vgg16)

data = {'filenames': fs, 'labels': ls, 'w': ws}
with open('ae/data/afhq/test/wild_w.pkl', 'wb') as handle:
    pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
