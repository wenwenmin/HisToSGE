import numpy as np
from sklearn.metrics import adjusted_rand_score
from utils import *
from model import *
from dataset import *
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
# section_id = "mouse_brain"
# save_model_path = f"../T2/{section_id}"
section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                    "151675", "151676"]
## 508 669 670 671 672
# for section_id in section_list:
section_id = "151676"
save_model_path = f"../T1/{section_id}"
adata, _ = get_sectionData(section_id)
adata_HistoSGE = sc.read_h5ad(save_model_path + '/recovered_data.h5ad')
print(adata_HistoSGE)
adata_sample = sc.read_h5ad(save_model_path + '/sampled_data.h5ad')
pr_stage = np.zeros(adata_HistoSGE.shape[1])
P_value = np.ones(adata_HistoSGE.shape[1])
mse_values = np.zeros(adata_HistoSGE.shape[1])
mae_values = np.zeros(adata_HistoSGE.shape[1])
used_gene = adata_HistoSGE.var.index

for it in tqdm(range(adata_HistoSGE.shape[1])):
    pr_stage[it], P_value[it] = \
        pearsonr(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(), adata[:, used_gene[it]].X.toarray().squeeze())
    mse_values[it] = mean_squared_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
                                        adata[:, used_gene[it]].X.toarray().squeeze())
    mae_values[it] = mean_absolute_error(adata_HistoSGE[:, used_gene[it]].X.toarray().squeeze(),
                                         adata[:, used_gene[it]].X.toarray().squeeze())
mask = ~np.isnan(pr_stage)
pr_stage_n = pr_stage[mask]
used_gene_n = used_gene[mask]
p_value = P_value[mask]
print("section_id:", section_id, "PCC:", np.mean(pr_stage_n))
print("section_id:", section_id, "AVG MSE:", np.mean(mse_values))
print("section_id:", section_id, "AVG MAE:", np.mean(mae_values))
