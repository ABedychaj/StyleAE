import cv2
import face_recognition as fr
import math
import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from arcface import ArcFace
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def load_images_from_folder(folder, list_to_compare=None):
    images = []
    labels = []
    list_of_files = []
    for filename in os.listdir(folder):
        if list_to_compare is not None and filename.split("_")[0] in list_to_compare:
            img = cv2.imread(os.path.join(folder, filename))
            if img is not None:
                labels.append(filename)
                images.append(torch.from_numpy(img).float())
                list_of_files.append(os.path.join(folder, filename))
    return torch.stack(images), labels, list_of_files


def calculate_mse(img1, img2):
    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean(((img1 - img2) / 255.0) ** 2)
    if mse == 0:
        return float('inf')
    return math.sqrt(mse)


def intersection(lst1, lst2, lst3):
    return list(set(lst1) & set(lst2) & set(lst3))


def sum(lst1, lst2):
    return list(set(lst1) | set(lst2))


def L2(x, y):
    return np.sqrt(np.sum(np.square(x - y)))


def prepare_results(path_input, path_output, face_rec, list_to_compare=None):
    print("Loading images from folder...")
    input_images, input_labels, input_list_of_files = load_images_from_folder(path_input, list_to_compare)
    output_images, output_labels, output_list_of_files = load_images_from_folder(path_output, list_to_compare)

    input_images = input_images.permute(0, 3, 1, 2)
    output_images = output_images.permute(0, 3, 1, 2)

    print("Images loaded...")
    input_df = {'labels': input_labels, 'images': input_images, 'files': input_list_of_files}
    output_df = {'labels': output_labels, 'images': output_images, 'files': output_list_of_files}

    print("Correctly sorted: ", input_df['labels'] == output_df['labels'])

    input_embs = face_rec.calc_emb(input_df['files'])
    output_embs = face_rec.calc_emb(output_df['files'])

    print("Calculating metrics...")
    print("Mean L2 between embeddings:", face_rec.get_distance_embeddings(input_embs, output_embs))

    psnr_measure = 0
    ssim_measure = 0
    mse = 0

    for i in range(len(input_df['images'])):
        psnr_measure += psnr(input_df['images'][i].numpy(), output_df['images'][i].numpy(), data_range=255)
        mse += calculate_mse(input_df['images'][i].numpy(), output_df['images'][i].numpy())
        ssim_measure += ssim(input_df['images'][i].numpy(), output_df['images'][i].numpy(), data_range=255,
                             channel_axis=0)
    psnr_measure /= len(input_df['images'])
    print("PSNR: ", abs(psnr_measure))

    ssim_measure /= len(input_df['images'])
    print("SSIM: ", ssim_measure)

    mse /= len(input_df['images'])
    print("MSE: ", mse)


if __name__ == '__main__':
    face_rec = ArcFace.ArcFace()

    path_plugen_output = "<output_images_plugen>"
    path_ae_output = "<output_images_styleae>"
    path_styleflow_output = "<output_images_styleflow>"
    path_input = "<input_images>"

    input_predictions = pd.read_csv("<classifications>")
    ae_predictions = pd.read_csv(
        "<classifications_styleae>")
    plugen_predictions = pd.read_csv(
        "<classifications_plugen>")
    styleflow_predictions = pd.read_csv(
        "<classifications_styleflow>")
    attribute = 'is_male'

    input_predictions["name"] = input_predictions["filename"].apply(lambda x: x.split("_")[0])

    ae_predictions = ae_predictions[ae_predictions[attribute] > 0.5]
    ae_predictions["name"] = ae_predictions["filename"].apply(lambda x: x.split("_")[0])

    plugen_predictions = plugen_predictions[plugen_predictions[attribute] > 0.5]
    plugen_predictions["name"] = plugen_predictions["filename"].apply(lambda x: x.split("_")[0])

    styleflow_predictions = styleflow_predictions[styleflow_predictions[attribute] > 0.5]
    styleflow_predictions["name"] = styleflow_predictions["filename"].apply(lambda x: x.split("_")[0])

    list_to_compare = intersection(ae_predictions["name"].tolist(), plugen_predictions["name"].tolist(),
                                   styleflow_predictions["name"].tolist())
    # list_to_compare = ae_predictions["name"].tolist()
    print(len(list_to_compare))

    print("PLUGEN")
    print("from 100:", len(plugen_predictions["name"].tolist()))
    prepare_results(path_input, path_plugen_output, face_rec, plugen_predictions["name"].tolist())

    print("AE")
    print("from 100:", len(ae_predictions["name"].tolist()))
    prepare_results(path_input, path_ae_output, face_rec, ae_predictions["name"].tolist())

    print("STYLEFLOW")
    print("from 100:", len(styleflow_predictions["name"].tolist()))
    prepare_results(path_input, path_styleflow_output, face_rec, styleflow_predictions["name"].tolist())

    # intersections
    ae_predictions = ae_predictions[ae_predictions["name"].isin(list_to_compare)]
    plugen_predictions = plugen_predictions[plugen_predictions["name"].isin(list_to_compare)]
    styleflow_predictions = styleflow_predictions[styleflow_predictions["name"].isin(list_to_compare)]
    input_predictions = input_predictions[input_predictions["name"].isin(list_to_compare)]

    ae_predictions = ae_predictions.sort_values(by=['name'])
    plugen_predictions = plugen_predictions.sort_values(by=['name'])
    styleflow_predictions = styleflow_predictions.sort_values(by=['name'])
    input_predictions = input_predictions.sort_values(by=['name'])

    print(ae_predictions.columns)
    print(len(list_to_compare))
    to_comapre = [_ for _ in ae_predictions.columns if _ != 'filename' and _ != attribute and _ != 'name']
    print("AE L2:", L2(ae_predictions[to_comapre].values, input_predictions[to_comapre].values))
    print("Plugen L2:", L2(plugen_predictions[to_comapre].values, input_predictions[to_comapre].values))
    print("Styleflow L2:", L2(styleflow_predictions[to_comapre].values, input_predictions[to_comapre].values))

    print("PLUGEN - intersection")
    prepare_results(path_input, path_plugen_output, face_rec, list_to_compare)

    print("AE - intersection")
    prepare_results(path_input, path_ae_output, face_rec, list_to_compare)

    print("Styleflow - intersection")
    prepare_results(path_input, path_styleflow_output, face_rec, list_to_compare)
