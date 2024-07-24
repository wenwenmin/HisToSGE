import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

import scipy.sparse as sp
from scipy.stats import pearsonr

from get_image_feature import UNI_features, frequency_features
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scanpy as sc
from PIL import Image


def setToArray(
        setInput,
        dtype='int64'
):
    """ This function transfer set to array.
        Args:
            setInput: set need to be trasnfered to array.
            dtype: data type.

        Return:
            arrayOutput: trasnfered array.
    """
    arrayOutput = np.zeros(len(setInput), dtype=dtype)
    index = 0
    for every in setInput:
        arrayOutput[index] = every
        index += 1
    return arrayOutput


def Euclidean_distance(
        feature_1,
        feature_2
):
    """ This function generates Euclidean distance between two vectors.
        Args:
            feature_1, feature_2: two vectors.

        Return:
            dist: distance between feature_1 and feature_2.
    """
    dist = np.linalg.norm(feature_1 - feature_2)
    return dist


def dist_with_slice(
        new_spot,
        origin_coor_df
):
    """ This function generates Euclidean distance between two vectors.
        Args:
            new_spot: coordinate of spot.
            origin_coor_df: coordinate of all original spots.
        Return:
            min_dist_with_spot: minimun of distance between new spot and all original spots.
    """
    dist_with_spot = []
    for it in range(origin_coor_df.shape[0]):
        dist_with_spot.append(Euclidean_distance(new_spot, origin_coor_df.iloc[it, :]))
    min_dist_with_spot = min(dist_with_spot)
    return min_dist_with_spot


def generation_coord_10x(
        adata,
        name='coord'
):
    """ This function generates spatial location for 10x Visium data.
        Args:
            adata: AnnData object storing original data.
            name: Item in adata.obsm used for get spatial location. Default is "coord".
        Return:
            coor_df: Spatial location of original data.
            fill_coor_df: Spatial location of generated data.
    """
    coor_df = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_ori = pd.DataFrame(adata.obsm['spatial'].copy())

    coor_df_1 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_1 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_1.iloc[:, 0] = coor_df_1.iloc[:, 0] + 1
    fill_spatial_1.iloc[:, 0] = fill_spatial_1.iloc[:, 0] + 68  # 137

    coor_df_2 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_2 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_2.iloc[:, 1] = coor_df_2.iloc[:, 1] + 0.5
    coor_df_2.iloc[:, 0] = coor_df_2.iloc[:, 0] + 0.5
    fill_spatial_2.iloc[:, 1] = fill_spatial_2.iloc[:, 1] + 60  # 120
    fill_spatial_2.iloc[:, 0] = fill_spatial_2.iloc[:, 0] + 34  # 69

    coor_df_3 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_3 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_3.iloc[:, 1] = coor_df_3.iloc[:, 1] + 0.5
    coor_df_3.iloc[:, 0] = coor_df_3.iloc[:, 0] - 0.5
    fill_spatial_3.iloc[:, 1] = fill_spatial_3.iloc[:, 1] + 60  # 120
    fill_spatial_3.iloc[:, 0] = fill_spatial_3.iloc[:, 0] - 34  # 69

    # Additional copies for 8x fill
    coor_df_4 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_4 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_4.iloc[:, 1] = coor_df_4.iloc[:, 1] - 0.5
    coor_df_4.iloc[:, 0] = coor_df_4.iloc[:, 0] + 0.5
    fill_spatial_4.iloc[:, 1] = fill_spatial_4.iloc[:, 1] - 60
    fill_spatial_4.iloc[:, 0] = fill_spatial_4.iloc[:, 0] + 34

    coor_df_5 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_5 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_5.iloc[:, 1] = coor_df_5.iloc[:, 1] - 0.5
    coor_df_5.iloc[:, 0] = coor_df_5.iloc[:, 0] - 0.5
    fill_spatial_5.iloc[:, 1] = fill_spatial_5.iloc[:, 1] - 60
    fill_spatial_5.iloc[:, 0] = fill_spatial_5.iloc[:, 0] - 34

    coor_df_6 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_6 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_6.iloc[:, 0] = coor_df_6.iloc[:, 0] - 1
    fill_spatial_6.iloc[:, 0] = fill_spatial_6.iloc[:, 0] - 68

    coor_df_7 = pd.DataFrame(adata.obsm[name].copy())
    fill_spatial_7 = pd.DataFrame(adata.obsm['spatial'].copy())
    coor_df_7.iloc[:, 1] = coor_df_7.iloc[:, 1] + 1
    fill_spatial_7.iloc[:, 1] = fill_spatial_7.iloc[:, 1] + 120

    fill_coor_df = pd.concat([coor_df, coor_df_1, coor_df_2, coor_df_3])  # , coor_df_4, coor_df_5, coor_df_6, coor_df_7
    # fill_coor_df = fill_coor_df.drop_duplicates(subset=fill_coor_df.columns)  # 删除重复行

    fill_spatial = pd.concat(
        [fill_spatial_ori, fill_spatial_1, fill_spatial_2,
         fill_spatial_3])  # , fill_spatial_4, fill_spatial_5, fill_spatial_6, fill_spatial_7
    # fill_spatial = fill_spatial.drop_duplicates(subset=fill_spatial.columns)  # 删除重复行

    coor_df.index = adata.obs.index
    coor_df.columns = ["x", "y"]
    fill_coor_df.columns = ["x", "y"]
    fill_coor_df.columns = ["x", "y"]

    return coor_df, fill_coor_df, fill_spatial


def generation_feature_10x(image_path, ori_spatial, fill_spatial):

    sample_features = UNI_features(img_path=image_path, spatial=ori_spatial)
    fill_features = UNI_features(img_path=image_path, spatial=fill_spatial)

    return sample_features, fill_features


def generation_coord_ST(
        adata,
        name='coord'
):
    """ This function generates spatial location for Spatial Transcriptomics data.
        Args:
            adata: AnnData object storing original data.
            name: Item in adata.obsm used for get spatial location. Default is "coord".
        Return:
            coor_df: Spatial location of original data.
            fill_coor_df: Spatial location of generated data.
    """
    coor_df = pd.DataFrame(adata.obsm[name])

    coor_df_1 = pd.DataFrame(adata.obsm[name])
    coor_df_1.iloc[:, 1] = coor_df_1.iloc[:, 1] + 0.5

    coor_df_2 = pd.DataFrame(adata.obsm[name])
    coor_df_2.iloc[:, 0] = coor_df_2.iloc[:, 0] + 0.5

    coor_df_3 = pd.DataFrame(adata.obsm[name])
    coor_df_3.iloc[:, 0] = coor_df_3.iloc[:, 0] + 0.5
    coor_df_3.iloc[:, 1] = coor_df_3.iloc[:, 1] + 0.5

    fill_coor_df = pd.concat([coor_df, coor_df_1, coor_df_2, coor_df_3])
    fill_coor_df = fill_coor_df.drop_duplicates(subset=fill_coor_df.columns)

    coor_df.index = adata.obs.index
    coor_df.columns = ["x", "y"]
    fill_coor_df.columns = ["x", "y"]

    return coor_df, fill_coor_df


def recovery_coord(
        adata,
        name='coord',
        down_ratio=0.5,
        path1='input_data',
):
    """ This function generates spatial location for Spatial Transcriptomics data.
        Args:
            adata: AnnData object storing original data.
            name: Item in adata.obsm used for get spatial location. Default is "coord".
            down_ratio: Down-sampling ratio. Default is 0.5.
        Return:
            coor_df: Spatial location of dowm-sampled data.
            fill_coor_df: Spatial location of recovered data.
            sample_index: Index of downsampled data.
            sample_barcode: Barcode of downsampled data.
    """
    spot_coord_ori = pd.DataFrame(adata.obsm[name])
    image_spatial_ori = pd.DataFrame(adata.obsm["spatial"].copy())
    spot_coord_ori.index = adata.obs.index
    spot_coord_ori.columns = ["x", "y"]
    sample_index = np.random.choice(range(spot_coord_ori.shape[0]), size=round(down_ratio * spot_coord_ori.shape[0]),
                                    replace=False)
    sample_index = setToArray(set(sample_index))
    spot_coord_sample = spot_coord_ori.iloc[sample_index]
    image_spatial_sample = image_spatial_ori.iloc[sample_index]
    sample_barcode = spot_coord_ori.index[sample_index]

    del_index = setToArray(set(range(spot_coord_ori.shape[0])) - set(sample_index))

    if not os.path.isdir(path1):
        os.makedirs(path1, exist_ok=True)

    np.savetxt(path1 + "/all_barcode.txt", adata.obs.index, fmt='%s')
    np.savetxt(path1 + "/sample_index.txt", sample_index, fmt='%s')
    np.savetxt(path1 + "/del_index.txt", del_index, fmt='%s')
    np.savetxt(path1 + "/sample_barcode.txt", spot_coord_ori.index[sample_index], fmt='%s')
    np.savetxt(path1 + "/del_barcode.txt", spot_coord_ori.index[del_index], fmt='%s')

    return spot_coord_sample, spot_coord_ori, image_spatial_sample, image_spatial_ori, sample_index, sample_barcode


def get_data(
        adata,
        experiment='generation',
        sample_index=None,
        sample_barcode=None,
        sec_name='section',
        select_section=[1, 3, 5, 6, 8],
        path1='input_data',
):
    """ Get training data used to generation from original AnnData object

        Args:
            adata: AnnData object storing original data. Raw data should to be normalized. Highly variable genes should be identified.
            experiment: Different tasks. Available options are: "generation", "recovery" or "3d_model". Default is "generation".
            sample_index: Index of downsampled data. Available when experiment = "recovery".
            sample_barcode: Barcode of downsampled data. Available when experiment = "recovery".
            sec_name: Item in adata.obs.columns used for choosing training sections. Available when experiment = "3d_model".
            select_section: Index of training sections. Available when experiment = "3d_model".

        Return:
            used_gene: Highly variable genes used to generation from original AnnData object
            normed_data: Normalized data extracted from original AnnData object.
            adata_sample: Down-sampled AnnData object. Available when experiment = "recovery".
    """
    used_gene = np.array(adata.var.index[adata.var.highly_variable])

    if not os.path.isdir(path1):
        os.mkdir(path1)

    np.savetxt(path1 + "/used_gene.txt", used_gene, fmt='%s')

    if experiment == 'generation':
        normed_data = sp.coo_matrix(adata.X[:, adata.var.highly_variable].T).todense()
        normed_data = pd.DataFrame(normed_data)
        return used_gene, normed_data
    elif experiment == 'recovery':
        adata_sample = adata[sample_barcode]
        normed_data = sp.coo_matrix(adata.X[sample_index][:, adata.var.highly_variable].T).todense()  # 5000,1452
        np.savetxt(path1 + "/normed_data.txt", normed_data)

        normed_data_all = sp.coo_matrix(adata.X[:, adata.var.highly_variable].T).todense()
        np.savetxt(path1 + "/normed_data_all.txt", normed_data_all)

        normed_data = pd.DataFrame(normed_data)
        normed_data_all = pd.DataFrame(normed_data_all)
        return used_gene, normed_data, normed_data_all, adata_sample
    elif experiment == '3d_model':
        normed_data = sp.coo_matrix(
            adata.X[adata.obs[sec_name].isin(select_section)][:, adata.var.highly_variable].T).todense()
        normed_data = pd.DataFrame(normed_data)
        return used_gene, normed_data


def show_train_hist(
        hist,
        loss_type,
        label,
        show=False,
        save=False,
        path='Train_hist.png'
):
    x = range(len(hist[loss_type]))

    y = hist[loss_type]

    plt.plot(x, y, label=label)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.legend(loc='best')
    plt.grid(True)
    plt.tight_layout()

    if save:
        plt.savefig(path)

    if show:
        plt.show()
    else:
        plt.close()


def pca_reduction(data_matrix, target_dim=128):
    # 1. 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_matrix)

    # 2. PCA降维
    pca = PCA(n_components=target_dim)
    reduced_data = pca.fit_transform(scaled_data)

    return reduced_data


def get_R(data1, data2, dim=1, func=pearsonr):
    adata1 = data1.X
    adata2 = data2.X
    r1, p1 = [], []
    for g in range(data1.shape[dim]):
        if dim == 1:
            r, pv = func(adata1[:, g], adata2[:, g])
        elif dim == 0:
            r, pv = func(adata1[g, :], adata2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1


def get_sectionData(section_id):
    section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                    "151675", "151676"]
    if section_id in section_list:
        input_dir = rf'../Data/DLPFC/{section_id}'
        adata = sc.read_visium(path=input_dir, count_file=f'{section_id}_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        image_path = input_dir + f"/spatial/{section_id}_full_image.tif"
        Ann_df = pd.read_csv(os.path.join(input_dir, section_id + '_truth.txt'), sep='\t', header=None,
                             index_col=0)
        Ann_df.columns = ['Ground Truth']

        # Labels (layer)
        adata.obs['layer'] = Ann_df.loc[adata.obs_names, 'Ground Truth']
        adata.uns['layer_colors'] = ['#1f77b4', '#ff7f0e', '#49b192', '#d62728', '#aa40fc', '#8c564b', '#e377c2']

    if section_id == "mouse_brain":
        input_dir = '../Data/mouse_brain_dataset'
        adata = sc.read_visium(path=input_dir, count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        image_path = input_dir + "/V1_Adult_Mouse_Brain_Coronal_Section_1_image.tif"

    if section_id == "Breast Cancer":
        input_dir = '../Data/Breast_Cancer'
        adata = sc.read_visium(path=input_dir,
                               count_file='V1_Breast_Cancer_Block_A_Section_1_filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        image_path = input_dir + "/V1_Breast_Cancer_Block_A_Section_1_image.tif"

    if section_id == "FFPE":
        input_dir = '../Data/FFPE'
        adata = sc.read_visium(path=input_dir,
                               count_file='filtered_feature_bc_matrix.h5')
        adata.var_names_make_unique()
        image_path = input_dir + "/image.tif"

    # Coordinates (array_col, array_row)
    adata.obsm["coord"] = adata.obs.loc[:, ['array_col', 'array_row']].to_numpy()

    # Normalization
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=1000)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata, image_path


def get_trainData(adata, image_path, experiment, data_type, path1, coord_sf=77):
    if experiment == 'generation' and data_type == '10x':
        spot_coord_ori, spot_coord_fill, image_spatial_fill = generation_coord_10x(adata)
        used_gene, exp_data = get_data(adata, experiment=experiment)
        image_features_ori, image_features_fill = generation_feature_10x(image_path, adata.obsm['spatial'],
                                                                         image_spatial_fill)
        return exp_data, spot_coord_ori, spot_coord_fill, image_spatial_fill, image_features_ori, image_features_fill, used_gene
    elif experiment == 'recovery' and data_type == '10x':
        spot_coord_sample, spot_coord_ori, image_spatial_sample, image_spatial_ori, sample_index, sample_barcode = recovery_coord(
            adata,
            down_ratio=0.5,
            path1=path1)
        used_gene, exp_data_sample, exp_data_ori, adata_sample = get_data(adata, experiment=experiment,
                                                                          sample_index=sample_index,
                                                                          sample_barcode=sample_barcode, path1=path1)
        image_features_ori, image_features_fill = generation_feature_10x(image_path, adata_sample.obsm['spatial'],
                                                                         adata.obsm['spatial'])

        return (exp_data_sample, exp_data_ori, adata_sample, image_features_ori, image_features_fill, spot_coord_sample,
                spot_coord_ori, image_spatial_sample, image_spatial_ori, used_gene)


def load_mask(filename, verbose=True):
    mask = load_image(filename, verbose=verbose)
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    return mask


def load_image(filename, verbose=True):
    img = Image.open(filename)
    img = np.array(img)
    if img.ndim == 3 and img.shape[-1] == 4:
        img = img[..., :3]  # remove alpha channel
    if verbose:
        print(f'Image loaded from {filename}')
    return img


def load_pickle(filename, verbose=True):
    with open(filename, 'rb') as file:
        x = pickle.load(file)
    if verbose:
        print(f'Pickle loaded from {filename}')
    return x


def save_pickle(x, filename):
    os.mkdir(filename)
    with open(filename, 'wb') as file:
        pickle.dump(x, file)
    print(filename)


def sort_labels(labels, descending=True):
    labels = labels.copy()
    isin = labels >= 0
    labels_uniq, labels[isin], counts = np.unique(
        labels[isin], return_inverse=True, return_counts=True)
    c = counts
    if descending:
        c = c * (-1)
    order = c.argsort()
    rank = order.argsort()
    labels[isin] = rank[labels[isin]]
    return labels, labels_uniq[order]


def save_image(img, filename):
    os.mkdir(filename)
    Image.fromarray(img).save(filename)
    print(filename)


def get_most_frequent(x):
    # return the most frequent element in array
    uniqs, counts = np.unique(x, return_counts=True)
    return uniqs[counts.argmax()]
