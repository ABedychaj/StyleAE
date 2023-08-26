import os

import cv2
import torch

input_path = 'data/afhq/train/dog'


def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            labels.append(filename)
            images.append(torch.from_numpy(img).float() / 255.0)
    return torch.stack(images), labels


def iterate_batches(tensors, labels, batch_size):
    num_batches = (len(tensors) + batch_size - 1) // batch_size
    for batch in range(num_batches):
        w = tensors[batch * batch_size: (batch + 1) * batch_size]
        a = labels[batch * batch_size: (batch + 1) * batch_size]
        yield w, a


if __name__ == '__main__':
    images, labels = load_images_from_folder(input_path)
    for batch in iterate_batches(images, labels, 2):
        print(batch)
        break
