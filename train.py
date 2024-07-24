from utils import *
from model import *
from dataset import *
from config import *
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def train(section_id, experiment, data_type, save_model_path, path1):
    seed, batch_size, learning_rate, step_size, gamma, train_epoch = get_setting()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    adata, image_path = get_sectionData(section_id)
    if experiment == "recovery":
        (exp_data_sample, exp_data_ori, adata_sample, image_features_sample, image_features_ori, spot_coord_sample,
         spot_coord_ori, image_spatial_sample, image_spatial_ori, used_gene) = get_trainData(adata, image_path,
                                                                                             experiment, data_type,
                                                                                             path1)
        transformed_dataset = MyDatasetTrans(normed_data=exp_data_sample, coor_df=spot_coord_sample,
                                             image=image_features_sample)
    elif experiment == "generation":
        exp_data, spot_coord_ori, spot_coord_fill, image_spatial_fill, image_features_ori, image_features_fill, used_gene = get_trainData(
            adata, image_path, experiment,
            data_type, path1)
        transformed_dataset = MyDatasetTrans(normed_data=exp_data, coor_df=spot_coord_ori,
                                             image=image_features_ori)
    dataloader_loader = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                   drop_last=False)

    model = HistoSGE(in_features=1024, n_genes=1000, depth=1, heads=4, dropout=0.)
    model.train()
    model = model.to(device)
    model_optim = optim.Adam(model.parameters(), lr=learning_rate)
    model_sche = optim.lr_scheduler.StepLR(model_optim, step_size=step_size, gamma=gamma)

    with tqdm(range(train_epoch), total=train_epoch, desc='Epochs') as epoch:
        for j in epoch:
            train_re_loss = []

            for exp, pos, image in dataloader_loader:
                exp, pos, image = exp.to(device), pos.to(device), image.to(device)

                model_optim.zero_grad()
                _, xrecon = model(image, pos)
                recon_loss = mse_loss(xrecon, exp)  # + 0.1 * L1_loss(xrecon,exp)  # + 0.1 * sliced_wasserstein_distance(xrecon, exp, 1000, device=device)
                recon_loss.backward()
                model_optim.step()
                model_sche.step()
                train_re_loss.append(recon_loss.item())
                epoch_info = 'recon_loss: %.5f' % \
                             (torch.mean(torch.FloatTensor(train_re_loss)))
                epoch.set_postfix_str(epoch_info)

    if experiment == "generation":
        torch.save(model, save_model_path + '/generation_4x.pth')
    else:
        torch.save(model, save_model_path + '/recover.pth')

    model.eval()
    if experiment == "generation":
        transformed_dataset = MyDatasetTrans2(coor_df=spot_coord_fill, image=image_features_fill)
        dataloader_test = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False)
        preds = None

        with torch.no_grad():
            for pos, image in tqdm(dataloader_test):
                pos, image = pos.to(device), image.to(device)
                h, pred = model(image, pos)

                if preds is None:
                    preds = pred.squeeze()
                else:
                    pred = pred.squeeze()
                    preds = torch.cat((preds, pred), dim=0)

        generate_profile = preds.squeeze().cpu().detach().numpy()
        adata_stage = sc.AnnData(generate_profile)
        adata_stage.obsm["coord"] = spot_coord_fill.to_numpy()
        adata_stage.var.index = used_gene

        adata.write(save_model_path + '/original_data.h5ad')
    else:
        transformed_dataset = MyDatasetTrans(normed_data=exp_data_ori, coor_df=spot_coord_ori, image=image_features_ori)
        dataloader_test = DataLoader(transformed_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                     drop_last=False)
        preds = None
        all_h_features = []
        with torch.no_grad():
            for exp, pos, image in tqdm(dataloader_test):
                exp, pos, image = exp.to(device), pos.to(device), image.to(device)
                h1, pred = model(image, pos)
                h1 = h1.squeeze()
                all_h_features.append(h1.cpu().detach().numpy())
                if preds is None:
                    preds = pred.squeeze()
                else:
                    pred = pred.squeeze()
                    preds = torch.cat((preds, pred), dim=0)
        all_h_features = np.concatenate(all_h_features, axis=0)
        with open(save_model_path + '/h_features.pkl', 'wb') as f:
            pickle.dump(all_h_features, f)
        generate_profile = preds.squeeze().cpu().detach().numpy()
        adata_stage = sc.AnnData(generate_profile)
        adata_stage.obsm["coord"] = spot_coord_ori.to_numpy()
        adata_stage.var.index = used_gene

        adata.write(save_model_path + '/original_data.h5ad')
    if experiment == 'recovery':
        np.savetxt(save_model_path + "/fill_data.txt", generate_profile)

    if experiment == 'generation':
        adata_stage.write(save_model_path + '/generated_data_4x.h5ad')
        return adata_stage
    elif experiment == 'recovery' and data_type == '10x':
        adata_sample.write(save_model_path + '/sampled_data.h5ad')
        adata_stage.obs = adata.obs
        adata_stage.write(save_model_path + '/recovered_data.h5ad')


if __name__ == "__main__":
    section_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674",
                    "151675", "151676"]
    for section_id in section_list:
        section_id = "151676"  # mouse_brain Breast Cancer FFPE
        experiment = "recovery"  # generation recovery
        save_model_path = f"../T1/{section_id}"
        path1 = save_model_path + "/file_tmp"
        train(section_id, experiment=experiment, data_type="10x", save_model_path=save_model_path, path1=path1)
        break