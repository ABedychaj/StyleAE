import numpy as np
import os
import pickle

import pandas as pd
import torch
from PIL import Image


def save_model(path, model, optimizer):
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)


def load_model(path, model, optimizer):
    loaded_state = torch.load(path)
    model.load_state_dict(loaded_state["model"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    model.eval()
    return model, optimizer


def make_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def save_img(img, filename):
    Image.fromarray(img, 'RGB').save(filename)


def iterate_batches(all_w, all_a, batch_size):
    num_batches = (len(all_w) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        w = all_w[batch * batch_size: (batch + 1) * batch_size]
        a = all_a[batch * batch_size: (batch + 1) * batch_size]
        yield w, a


def parse_attr(raw_attr, attr, default_val):
    values = []
    for a in raw_attr:
        data = None
        if type(a) is dict:
            data = a
        elif type(a) is list and len(a) > 0:
            data = a[0]
        else:
            values.append(default_val)
        if data is not None:
            if type(attr) is tuple:
                values.append(data["faceAttributes"][attr[0]][attr[1]])
            else:
                values.append(data["faceAttributes"][attr])
    return values


def normalize(x, values, not_age, keep=False):
    x = (x - x.min()) / (x.max() - x.min())
    if keep or values == "continuous":
        return 2 * x - 1
    elif values > 2:
        x[x == 1.] = 0.9999
        x = ((values * x).int().float() / (values - 1)).float()
        x = 2 * x - 1
    else:
        x = 2 * x - 1
    return x


def load_dataset(keep=False, values="continuous", keep_indexes=None, nbr_of_attributes=17):
    # keep_indexes = [2, 5, 25, 28, 16, 32, 33, 34, 55, 75, 79, 162, 177, 196, 160, 212, 246, 285, 300, 329, 362, 369, 462, 460, 478, 551, 583, 643, 879, 852, 914, 999, 976, 627, 844, 237, 52, 301,599] # Indexes from StyleFLow project
    if keep_indexes is None:
        keep_indexes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        keep_indexes = np.array(keep_indexes).astype(np.int)

    raw_attr = pickle.load(open("ae/data/all_att.pickle", "rb"))["Attribute"][0]
    gender = np.array(
        [float(v == "male") for v in parse_attr(raw_attr, "gender", "male")]
    ).reshape(
        10000, 1
    )  # 1000/1000
    glasses = np.array(
        [float(v != "NoGlasses") for v in parse_attr(raw_attr, "glasses", "NoGlasses")]
    ).reshape(
        10000, 1
    )  # 999/1000
    yaw = np.array(
        [v for v in parse_attr(raw_attr, ("headPose", "yaw"), None)]
    ).reshape(
        10000, 1
    )  # 999/1000
    pitch = np.array(
        [v for v in parse_attr(raw_attr, ("headPose", "pitch"), None)]
    ).reshape(
        10000, 1
    )  # 999/1000
    baldness = np.array(
        [v for v in parse_attr(raw_attr, ("hair", "bald"), None)]
    ).reshape(
        10000, 1
    )  # 999/1000
    beard = np.array(
        [v for v in parse_attr(raw_attr, ("facialHair", "beard"), None)]
    ).reshape(
        10000, 1
    )  # 999/1000
    age = np.array([v for v in parse_attr(raw_attr, "age", None)]).reshape(
        10000, 1
    )  # 999/1000
    expression = np.array([v for v in parse_attr(raw_attr, "smile", None)]).reshape(
        10000, 1
    )  # 999/1000

    raw_attr = np.concatenate(
        [gender, glasses, beard, baldness, expression, age, yaw, pitch], axis=1
    ).astype(float)
    # Replace NaN values
    col_mean = np.nanmean(raw_attr, axis=0)
    inds = np.where(np.isnan(raw_attr))
    raw_attr[inds] = np.take(col_mean, inds[1])

    raw_attr = torch.Tensor(raw_attr)
    del gender, glasses, yaw, pitch, baldness, beard, age, expression

    raw_lights = pickle.load(open("ae/data/all_light10k.pickle", "rb"))["Light"]
    raw_lights = torch.Tensor(raw_lights)
    all_a = torch.cat((raw_attr.unsqueeze(2), raw_lights.squeeze(1).squeeze(-1)), dim=1)

    # thr = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + thr + [0, 0]
    if values == "continuous":
        for i in range(17):
            all_a[:, i] = normalize(all_a[:, i], values, False, keep=keep)
    else:
        for i in range(17):
            all_a[:, i] = normalize(all_a[:, i], values, True, keep=keep)

    all_a = all_a[:, :nbr_of_attributes]

    if keep:
        raw_w = pickle.load(open("ae/data/sg2latents.pickle", "rb"))
    else:
        raw_w = pickle.load(open("ae/data/all_latents.pickle", "rb"))

    all_w = np.array(raw_w["Latent"])
    all_w = torch.tensor(all_w)

    if keep:
        all_a = all_a[keep_indexes]
        all_w = all_w[keep_indexes]
    return all_w, all_a


def label_record(x, n=1):
    if x == "cat":
        return torch.tensor([[1.0] * n + [0.0] * n + [0.0] * n])  # for n=1, this is [1,0,0]
    elif x == "dog":
        return torch.tensor([[0.0] * n + [1.0] * n + [0.0] * n])
    elif x == "wild":
        return torch.tensor([[0.0] * n + [0.0] * n + [1.0] * n])


def load_dataset_afhq(train=True, n=1):
    if train:
        with open("ae/data/afhq/train/cat_w.pkl", 'rb') as handle:
            cat = pickle.load(handle)

        with open("ae/data/afhq/train/dog_w.pkl", 'rb') as handle:
            dog = pickle.load(handle)

        with open("ae/data/afhq/train/wild_w.pkl", 'rb') as handle:
            wild = pickle.load(handle)
    else:
        with open("ae/data/afhq/test/cat_w.pkl", 'rb') as handle:
            cat = pickle.load(handle)

        with open("ae/data/afhq/test/dog_w.pkl", 'rb') as handle:
            dog = pickle.load(handle)

        with open("ae/data/afhq/test/wild_w.pkl", 'rb') as handle:
            wild = pickle.load(handle)

    ws_c = [x.cpu().clone().detach().repeat(1, 16, 1) for x in cat['w']]
    ws_d = [x.cpu().clone().detach().repeat(1, 16, 1) for x in dog['w']]
    ws_w = [x.cpu().clone().detach().repeat(1, 16, 1) for x in wild['w']]

    all_w = torch.stack(ws_c + ws_d + ws_w, dim=0).squeeze(2).squeeze(2).squeeze(2)

    all_a = np.concatenate((cat['labels'], dog['labels'], wild['labels']), axis=0)
    all_a = torch.stack([label_record(x, n) for x in all_a], dim=0)

    return all_w, all_a


def load_dataset_afhq_for_faces(n=1):
    with open("ae/data/afhq/test/dog_w_faces.pkl", 'rb') as handle:
        dog = pickle.load(handle)

    ws_d = [x.cpu().clone().detach().repeat(1, 18, 1) for x in dog['w']]

    all_w = torch.stack(ws_d, dim=0).squeeze(2).squeeze(2).squeeze(2)

    all_a = dog['labels']
    all_a = torch.stack([label_record(x, n) for x in all_a], dim=0)

    return all_w, all_a


if __name__ == "__main__":
    w, a = load_dataset_afhq_for_faces(n=1)
    print(w.shape)
    # print(len(w))
    # print(a[0].shape)
