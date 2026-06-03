from utils.experiment_utils import set_seed
from data.utils import get_generation_models
from omegaconf import DictConfig
from models.deep_mmd import deep_MMD
from models.tall import SingleSwinBlockDiscriminator
from utils.train_utils import *
from utils.data_utils import *
from omegaconf import OmegaConf
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from tabulate import tabulate
import torch
import copy
import os
import pandas as pd

@hydra.main(config_path="configs/nsg-vd-mp-224x224", config_name="model-test.yaml", version_base=None)
def main(cfg: DictConfig):
    # Setup Logging
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # ----------------------------------- Model ---------------------------------- #
    logger.info(f"Loading model: {cfg.model.name}")
    if cfg.model.name in ["Velocity_Single_TALL_MMD", "Score_Single_TALL_MMD"]:
        logger.info("Using single layer of TALL_MMD model")
        discriminator = SingleSwinBlockDiscriminator(num_features=cfg.model.feature_dim)
    else:
        raise ValueError(f"Unsupported model: {cfg.model.name}")
    
    model = deep_MMD(discriminator=discriminator, 
                        sigma=cfg.model.sigma, 
                        sigma0=cfg.model.sigma0, 
                        epsilon=cfg.model.epsilon, 
                        img_size=cfg.model.img_size, 
                        is_yy_zero=cfg.model.is_yy_zero,
                        is_smooth=cfg.model.is_smooth)
    model.load_state_dict(torch.load(cfg.ckpt_path, weights_only=True))
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model.net = nn.DataParallel(model.net, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    model = model.to(device)
    model.eval()
    
    # ----------------------------------- Data ----------------------------------- #
    generation_models = get_generation_models(cfg.data.dataset_name)
    if cfg.data.get("ref_models", None) is not None:
        ref_dataloaders = get_ref_dataloaders(cfg.data, 
                                              cfg.data.ref_models,
                                              mode="test",)
    else:
        ref_dataloader = get_ref_dataloader(cfg.data, 
                                        cfg.data.ref_model,
                                        mode="val",
                                        resolution_size=cfg.data.resolution_size)
    test_dataloaders = {}
    for fake_model in generation_models["fake"]["test"]:
        for real_model in get_generation_models(cfg.data.dataset_name)["real"]["test"]:
            real_model = cfg.data.test_real_model
            if fake_model == "Sora":
                load_len = 56
            else:
                load_len = cfg.data.test_load_len
            test_datasets = get_score_datasets(cfg.data, 
                                              "test",
                                              load_len=load_len,
                                              real_model=real_model,
                                              generation_model=fake_model, filter=False, resolution_size=cfg.data.resolution_size)
            test_loader = get_data_loaders_for_mmd(cfg.data, test_datasets, batch_size=cfg.data.val_batch_size)
            test_dataloaders[f"{fake_model}"] = test_loader

    if cfg.data.get("ref_models", None) is not None and cfg.data.get("ref_ratio", None) is not None:
        feature_ref, ref_data = get_ref_features_multi_source(model, ref_dataloaders, cfg.data.ref_ratio, cfg.data.ref_load_len)
    else:
        feature_ref, ref_data = get_ref_features(model, ref_dataloader, cfg.data.ref_load_len)
    feature_ref = feature_ref.cuda()
    regression_model = None
        
    #  Evaluate Model
    results = []
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
            test_results = test_dMMD(
                model, test_loader, feature_ref, ref_data,
                regression_model
            )
            results.append([name, 
                            test_results["recall"], 
                            test_results["accuracy"], 
                            test_results["f1"], 
                            test_results["auroc"],
                            test_results["precision"], 
                            ])
            logger.info(
                f"Dataset: {name} | Recall: {test_results['recall']:.4f} | F1: {test_results['f1']:.4f} | "
                f"Accuracy: {test_results['accuracy']:.4f} | Precision: {test_results['precision']:.4f} | "
                f"AUROC: {test_results['auroc']:.4f}"
            )
    # Save results to CSV
    headers = ["Dataset", "Recall", "Accuracy", "F1", "AUROC", "Precision"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    ref_result = [cfg.data.ref_load_len]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
        ref_result.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))
    
    csv_path = os.path.join(cfg.log_path, cfg.save_csv_file)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save results to CSV with four decimal places
    df = pd.DataFrame(results, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df = df.transpose()  # Transpose the table
    df.to_csv(csv_path, index=True, header=False)
    logger.success(f"Test results saved to {csv_path}")

@torch.no_grad()
def test_dMMD(model, test_dataloader, feature_ref, ref_data, regression_model=None):
    model.eval()
    is_smooth = model.is_smooth
    sigma = model.sigma
    sigma0_u = model.sigma0_u
    ep = model.ep
    net = model.net    
    
    fake_dataloader = test_dataloader["fake"]
    real_dataloader = test_dataloader["real"]
    dt_clean = []
    dt_adv = []
    with torch.no_grad():
        for idx, (real_data, fake_data) in tqdm(enumerate(zip(real_dataloader, fake_dataloader)), total=len(real_dataloader), desc="Evaluating", leave=False):
            x_real = real_data[0].float().cuda()
            x_fake = fake_data[0].float().cuda()
            
            _,feature_cln = net(x_real,out_feature=True)
            _,feature_adv = net(x_fake,out_feature=True)
            dt_clean.append(MMD_batch2(torch.cat([feature_ref,feature_cln],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_real],dim=0).view(ref_data.shape[0]+x_real.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())
            
            dt_adv.append(MMD_batch2(torch.cat([feature_ref,feature_adv],dim=0), feature_ref.shape[0], torch.cat([ref_data,x_fake],dim=0).view(ref_data.shape[0]+x_fake.shape[0],-1), sigma, sigma0_u, ep, is_smooth=is_smooth).cpu())

        dt_clean = torch.cat(dt_clean)
        dt_adv = torch.cat(dt_adv)
        raw_predict = torch.cat([dt_clean, dt_adv], dim=0)
        if regression_model is None:
            predict = (raw_predict > 1).int()
        else:
            predict = regression_model.predict(raw_predict.reshape(-1,1).cpu())
        labels = torch.cat([torch.zeros(len(dt_clean)), torch.ones(len(dt_adv))], dim=0)
        try:
            auroc = roc_auc_score(labels.cpu(), raw_predict.cpu())
            precision = precision_score(labels, predict)
            recall = recall_score(labels, predict)
            f1 = f1_score(labels, predict)
            acc = accuracy_score(labels, predict)
            # auroc, info = plot_mi(dt_clean, dt_adv, plot=False)
        except ValueError as e:
            logger.error(f"DeepMMD testing failed due to {e}. Exiting the program.")
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
    }

if __name__ == "__main__":
    main()