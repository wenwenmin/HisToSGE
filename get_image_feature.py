from PIL import Image
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download
import torch
from torch.utils.data import Dataset
import cv2
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from numpy.fft import fft2, fftshift

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
login("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")

# login with your User Access Token, found at https://huggingface.co/settings/tokens
# pretrained=True needed to load UNI weights (and download weights for the first time)
# init_values need to be passed in to successfully load LayerScale parameters (e.g. - block.0.ls1.gamma)
model = timm.create_model("hf-hub:MahmoodLab/uni", pretrained=True, init_values=1e-5, dynamic_img_size=True)
transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))


class roi_dataset(Dataset):
    def __init__(self, img,
                 ):
        super().__init__()
        self.transform = transform

        self.images_lst = img

    def __len__(self):
        return len(self.images_lst)

    def __getitem__(self, idx):
        pil_image = Image.fromarray(self.images_lst[idx].astype('uint8'))
        image = self.transform(pil_image)
        return image


def crop_image(img, x, y, crop_size=None):
    # 计算子图左上角的坐标
    if crop_size is None:
        crop_size = [50, 50]
    left = x - crop_size[0] // 2
    top = y - crop_size[1] // 2

    # 计算子图右下角的坐标
    right = left + crop_size[0]
    bottom = top + crop_size[1]

    if img.ndim == 3:
        cropped_img = img[top:bottom, left:right, :]
    else:
        cropped_img = img[top:bottom, left:right]

    return cropped_img


def UNI_features(img_path, spatial):
    model.eval()
    model.to(device)

    img = cv2.imread(img_path)
    img = np.array(img)

    sub_images = []
    spot_num = len(spatial)
    loc = spatial
    if isinstance(loc, pd.DataFrame):
        loc = loc.values
    # 遍历spot以提取子图像
    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]
        sub_image = crop_image(img, x, y)
        sub_images.append(sub_image)
    sub_images = np.array(sub_images)

    test_datat = roi_dataset(sub_images)
    database_loader = torch.utils.data.DataLoader(test_datat, batch_size=512, shuffle=False)

    feature_embs = []

    with torch.inference_mode():
        for batch in database_loader:
            batch = batch.to(device)
            feature_emb = model(batch)
            feature_embs.append(feature_emb.cpu())
        feature_embs = np.concatenate(feature_embs, axis=0)

    return feature_embs


def extract_glcm_features(img_path, spatial):
    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    levels = 256

    features = []

    image = cv2.imread(img_path)
    image = np.array(image)
    sub_images = []
    spot_num = len(spatial)
    loc = spatial
    if isinstance(loc, pd.DataFrame):
        loc = loc.values

    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]
        sub_image = crop_image(image, x, y)
        sub_images.append(sub_image)
    sub_images = np.array(sub_images)
    for sub_image in sub_images:
        sub_features = []
        for channel in range(sub_image.shape[2]):
            glcm = graycomatrix(sub_image[:, :, channel], distances=distances, angles=angles, levels=levels,
                                symmetric=True,
                                normed=True)
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            ASM = graycoprops(glcm, 'ASM').flatten()

            sub_features.extend([contrast, dissimilarity, homogeneity, energy, correlation, ASM])
        sub_features = np.concatenate(np.array(sub_features), axis=0)
        features.append(sub_features)
    features = np.array(features)
    return np.concatenate(features, axis=0)


def extract_fft_features(img_path, spatial):
    image = cv2.imread(img_path)
    img = np.array(image)
    sub_images = []
    spot_num = len(spatial)
    loc = spatial
    if isinstance(loc, pd.DataFrame):
        loc = loc.values

    for i in range(spot_num):
        x = loc[i, 0]
        y = loc[i, 1]
        sub_image = crop_image(image, x, y)
        sub_images.append(sub_image)
    sub_images = np.array(sub_images)
    frequency_features = []
    for sub_image in sub_images:
        sub_image_frequency_features = []
        for channel in range(sub_image.shape[2]):
            f_transform = fft2(sub_image[:, :, channel])
            f_transform_shifted = fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
            sub_image_frequency_features.append(magnitude_spectrum)
        sub_image_frequency_features = np.stack(sub_image_frequency_features, axis=-1)
        frequency_features.append(sub_image_frequency_features)
    # frequency_features = np.stack(frequency_features, axis=0)

    return frequency_features




def frequency_features(img_path, spatial):
    image = extract_fft_features(img_path, spatial)
    return np.array(image)
    # test_datat = roi_dataset(image)
    # database_loader = torch.utils.data.DataLoader(test_datat, batch_size=512, shuffle=False)
    #
    # feature_embs = []
    #
    # with torch.inference_mode():
    #     for batch in database_loader:
    #         batch = batch.to(device)
    #         feature_emb = model(batch)
    #         feature_embs.append(feature_emb.cpu())
    #     feature_embs = np.concatenate(feature_embs, axis=0)
    #
    # return feature_embs

# section_id = "151507"
# input_dir = rf'../Data/DLPFC/{section_id}'
# image_path = input_dir + f"/spatial/{section_id}_full_image.tif"
# import scanpy as sc
#
# adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
# adata.var_names_make_unique()
# fft_image = extract_fft_features(image_path, adata.obsm["spatial"])
# frequency_feature = frequency_features(image_path, adata.obsm["spatial"])
# print(frequency_feature.shape)

