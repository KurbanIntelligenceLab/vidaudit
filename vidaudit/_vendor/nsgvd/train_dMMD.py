from utils.experiment_utils import set_seed
from data.utils import get_generation_models
from omegaconf import DictConfig
from models.deep_mmd import deep_MMD
from models.discriminators import SWINDiscriminator
from models.tall import SingleSwinBlockDiscriminator
from utils.train_utils import *
from utils.data_utils import *
from omegaconf import OmegaConf
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from tabulate import tabulate
import torch
from torchinfo import summary
import os
from torch.utils.tensorboard import SummaryWriter

@hydra.main(config_path="configs/nsg-vd-224x224", config_name="standard.yaml", version_base=None)
def main(cfg: DictConfig):
    log_dir = os.path.join(cfg.log_path, cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"output.txt")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    logger.info(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(log_dir=log_dir)
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------- Model ---------------------------------- #
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
    model = model.to(device)
    
    summary(model)

    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model.net = nn.DataParallel(model.net, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
    
    # ----------------------------------- Data ----------------------------------- #
    if cfg.task_type == "standard":
        pn_ratio = 1
    elif cfg.task_type == "unbalance":
        pn_ratio = cfg.data.pn_ratio
    else:
        raise ValueError(f"Unsupported task type: {cfg.task_type}")

    filter_frames = cfg.data.filter_frames if "filter_frames" in cfg.data else False
    train_datasets = get_score_datasets(cfg.data, 
                                        "train",
                                        load_len=cfg.data.train_load_len,
                                        generation_model=cfg.data.generation_model,
                                        real_model=cfg.data.train_real_model,
                                        filter=cfg.data.filter_nsg,
                                        pn_ratio=pn_ratio,
                                        filter_frames=filter_frames,
                                        resolution_size=cfg.data.resolution_size)
    train_loaders = get_data_loaders_for_mmd(cfg.data, 
                                            train_datasets,
                                            batch_size=cfg.data.batch_size)
          
    if cfg.data.get("ref_models", None) is not None:
        ref_loaders = get_ref_dataloaders(cfg.data, 
                                          cfg.data.ref_models,
                                        mode=cfg.data.ref_mode,
                                        filter_frames=filter_frames)
    else:
        ref_loader = get_ref_dataloader(cfg.data, 
                                        cfg.data.ref_model,
                                        mode=cfg.data.ref_mode,
                                        filter_frames=filter_frames,
                                        resolution_size=cfg.data.resolution_size)                              
    val_dataloaders = {}
    generation_models = get_generation_models(cfg.data.dataset_name)
    for fake_model in generation_models["fake"]["val"]:
        for real_model in get_generation_models(cfg.data.dataset_name)["real"]["test"]:
            val_datasets = get_score_datasets(cfg.data, 
                                              "val",
                                              real_model=real_model,
                                              generation_model=fake_model,
                                              load_len=cfg.data.val_load_len,
                                              filter=False,
                                              pn_ratio=1,
                                              filter_frames=filter_frames,
                                              resolution_size=cfg.data.resolution_size,)
            val_loaders = get_data_loaders_for_mmd(cfg.data, val_datasets, batch_size=cfg.data.val_batch_size)
            val_dataloaders[f"{fake_model}/{real_model}"] = val_loaders

    # ----------------------------------- Train ---------------------------------- #
    if cfg.trainer.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    elif cfg.trainer.optimizer.name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer.name}")

    # log vals
    global_step = 0
    best_val_auroc = - float("inf")
    
    # train logics
    with tqdm(range(cfg.trainer.max_epochs), desc="Epochs", unit="epoch", position=0) as epoch_pbar:
        for epoch in epoch_pbar:
            if cfg.task_type == "standard":
                train_results = train_dMMD(model, train_loaders, optimizer, device, global_step, writer)
            elif cfg.task_type == "unbalance":
                train_results = train_dMMD_unbalance(model, train_loaders, optimizer, device, global_step, writer)
            else:
                raise ValueError(f"Unsupported task type: {cfg.task_type}")
            global_step = train_results['global_step']
            epoch_pbar.set_postfix({
                **train_results,
            })
            train_info = " | ".join([f"{key}: {value:.4e}" if isinstance(value, float) else f"{key}: {value}"
                         for key, value in train_results.items()])
            if (epoch+1) % cfg.trainer.val_check_interval == 0:
                if cfg.data.get("ref_models", None) is not None and ref_loaders is not None:
                    val_results = val_dMMD(model, val_dataloaders, ref_dataloaders=ref_loaders, ref_len=cfg.data.ref_load_len, global_step=global_step, writer=writer, ref_ratio=cfg.data.ref_ratio)
                else:
                    val_results = val_dMMD(model, val_dataloaders, ref_dataloader=ref_loader, ref_len=cfg.data.ref_load_len, global_step=global_step, writer=writer)
                val_info = tabulate(val_results, headers=["Fake", "Real", "Auroc"], tablefmt="grid")
                logger.info(
                    f"Epoch {epoch+1:2}/{cfg.trainer.max_epochs:2}\nTrain Info: {train_info} \n{val_info}"
                )
                
                val_auroc = val_results[-1][-1]
                if val_auroc > best_val_auroc:
                    logger.info(f"Current auroc({val_auroc:.4f}) > Best auroc({best_val_auroc:.4f})")
                    best_val_auroc = val_auroc
                    best_model_save_path = os.path.join(cfg.save_ckpt_path, f"best_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")
            if cfg.data.feature_type == "score":
                ckpt_path = os.path.join(cfg.save_ckpt_path, f"ckpt_{epoch+1}.pth")
                os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
                torch.save(model.state_dict(), ckpt_path)
                logger.success(f"Model saved at {ckpt_path}")
            else:
                logger.info(
                    f"Epoch {epoch+1:2}/{cfg.trainer.max_epochs:2}\nTrain Info: {train_info}"
                )
    
    # save model ckpts
    model_save_path = os.path.join(cfg.save_ckpt_path, f"final_ckpt.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.success(f"Model saved at {model_save_path}")
    
if __name__ == "__main__":
    main()
