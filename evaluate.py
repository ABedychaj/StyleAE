import numpy as np
from sklearn.utils import shuffle
from datetime import datetime
from torch import optim
from ae.utils import load_dataset, save_model, iterate_batches, load_model
from ae.AutoEncoder import AE_single_layer, AE_multiple_layers, InvertibleAE, CholeskyAE, InvOrthogonalAE, DeepAE, \
    relu_loss
import pandas as pd
import os
import PIL.Image
import pickle
import torch
import gc

from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from resnets import ResNet18

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


def generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm, transform_eval, a):
    _, code = reconstruction_model(w)
    code[:, :, :, 1] = a
    output_images = generate_images_in_w_space(reconstruction_model.decode(code), w.shape[0])
    pred = sigm(classification_model(transform_eval(output_images)))
    return pred.detach().cpu().numpy().round(2)[0][1]


def naive_select(reconstruction_model,
                 classification_model,
                 sigm,
                 transform_eval,
                 w,
                 arr,
                 target):
    values = []
    for a in arr:
        values.append(generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                                 transform_eval, a))
    if arr[0] < 0:
        values = values[::-1]
        arr = arr[::-1]
    return arr[np.argmin(np.abs(np.array(values) - target))]


def binary_search(
        reconstruction_model,
        classification_model,
        sigm,
        transform_eval,
        w,
        arr,
        target):
    n = len(arr)

    # Corner cases
    if target <= generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                            transform_eval, arr[0]):
        return arr[0]
    if target >= generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                            transform_eval, arr[n - 1]):
        return arr[n - 1]

    # Doing binary search
    i = 0
    j = n
    mid = 0
    while i < j:
        mid = (i + j) // 2

        val1 = generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                          transform_eval, arr[mid - 1])
        val2 = generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                          transform_eval, arr[mid])
        val3 = generate_and_classify_images_for_a_given_w(w, reconstruction_model, classification_model, sigm,
                                                          transform_eval, arr[mid + 1])

        if target == val2:
            return arr[mid]

        # If target is less than array
        # element, then search in left
        if target < val2:

            # If target is greater than previous
            # to mid, return closest of two
            if mid > 0 and target > val1:
                return getClosest(
                    (val1, arr[mid - 1]),
                    (val2, arr[mid]),
                    target)

            # Repeat for left half
            j = mid

        # If target is greater than mid
        else:
            if mid < n - 1 and target < val3:
                return getClosest(
                    (val2, arr[mid]),
                    (val3, arr[mid + 1]),
                    target)

            # update i
            i = mid + 1

    # Only single element left after search
    return arr[mid]

    # Method to compare which one is the more close.
    # We find the closest by taking the difference
    # between the target and both values. It assumes
    # that val2 is greater than val1 and target lies
    # between these two.


def getClosest(val1, val2, target):
    if (target - val1[0] >= val2[0] - target):
        return val2[1]
    else:
        return val1[1]


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


# Specify path
path_output = 'ae/data/output_images_ae_naive_search_noglasses'
path_diff = 'ae/data/diff_images_ae_naive_search_noglasses'
# path_input = 'ae/data/input_images_ae_smile'

if not os.path.exists(path_output):
    os.makedirs(path_output)

if not os.path.exists(path_diff):
    os.makedirs(path_diff)

# if not os.path.exists(path_input):
#     os.makedirs(path_input)

# StyleGAN2-ada-pytorch
with open('ae/data/ffhq.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].to(device)
    G.eval()

# StyleFlow dataset
nbr_of_attributes = 8
all_w, all_a = load_dataset(keep=False, values=2, nbr_of_attributes=nbr_of_attributes)

# AE Model
model = AE_multiple_layers(input_shape=512, hidden_dim=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Only on gmum servers
model, optimizer = load_model('ae_models/pixel_loss_v1/model_e100_25_01_2023_04_06_16.pch', model, optimizer)

# training params
batch_size_test = 1

# all_w_tmp, all_a_tmp = shuffle(all_w[:9500], all_a[:9500])
all_test_w, all_test_a = all_w[9900:], all_a[9900:]

model.eval()

count_batches = 0
dt_string = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
data = []
attribs = []

# classifier
mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
transform_eval = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BILINEAR),
    transforms.CenterCrop(224),
    transforms.Normalize(mean, std),
])

net = ResNet18(num_classes=9).to(device)
net.load_state_dict(torch.load(f"resnet18_sd_25012023.pt"))
net.eval()

sigm = nn.Sigmoid()
print("Classification model is loaded")

for w, a in iterate_batches(all_test_w, all_test_a, batch_size_test):
    w = w.to(device)
    a = a.to(device)

    val = naive_select(
        reconstruction_model=model,
        classification_model=net,
        sigm=sigm,
        transform_eval=transform_eval,
        w=w,
        arr=np.linspace(-20, -3, 20, dtype=float),
        target=0.09
    )

    input_images = generate_images_in_w_space(w, w.shape[0])

    _, orig_code = model(w)
    code = orig_code.clone()
    code[:, :, :, 1] = val
    reconstruction = model.decode(code)

    output_images = generate_images_in_w_space(reconstruction, w.shape[0])

    pred = sigm(net(transform_eval(output_images)))
    print(pred.detach().cpu().numpy().round(2)[0][1])

    save_image(output_images, f'{path_output}/epoch{count_batches + 1}_{dt_string}')
    save_image(torch.abs(output_images - input_images), f'{path_diff}/diff_epoch{count_batches + 1}_{dt_string}')
    # save_image(input_images, f'{path_input}/epoch{count_batches + 1}_{dt_string}')

    tmp = [f"epoch{count_batches + 1}_{dt_string}"]
    tmp.extend([val])
    tmp.extend(orig_code[0, 0, 0, :8].detach().cpu().numpy().round(2).flatten())
    tmp.extend(code[0, 0, 0, :8].detach().cpu().numpy().round(2).flatten())
    data.append(tmp)
    tmp_a = [f"epoch{count_batches + 1}_{dt_string}"]
    tmp_a.extend(a.reshape(1, 1, 8).detach().cpu().numpy().round(2).flatten())
    attribs.append(tmp_a)
    count_batches += 1
    print("Saved: ", count_batches)

titles = ["gender", "glasses", "beard", "baldness", "expression", "age", "yaw", "pitch"]

regular_list = [['orig_code_' + i for i in titles], ['code_' + i for i in titles]]
columns = ["name"] + ["value_of_changed_code"] + [item for sublist in regular_list for item in sublist]
df = pd.DataFrame(data, columns=columns)
df.to_csv(f'{path_output}/summary_data_{dt_string}.csv', index=False)

df_attrib = pd.DataFrame(attribs, columns=["name"] + titles)
# df_attrib.to_csv(f'{path_input}/labels_from_ms_{dt_string}.csv', index=False)
print(df)
