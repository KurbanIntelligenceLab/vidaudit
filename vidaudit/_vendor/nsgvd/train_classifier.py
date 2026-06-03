from utils.experiment_utils import set_seed
from data.utils import get_generation_models
from omegaconf import DictConfig
from models.discriminators import *
from utils.train_utils import *
from utils.data_utils import *
from models.demamba import XCLIP_DeMamba
from models.npr import resnet50_npr
from models.tall import TALL_SWIN
from models.stil import Det_STIL
from omegaconf import OmegaConf
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from torchinfo import summary
import torch
import time  
from torch.utils.tensorboard import SummaryWriter
import os
from tabulate import tabulate

@hydra.main(config_path="configs/classifier-224x224", config_name="npr.yaml", version_base=None)
def main(cfg: DictConfig):
    log_dir = os.path.join(cfg.log_path, cfg.experiment_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{cfg.model.name}_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    logger.add(log_file, format="{time} {level} {message}", level="INFO", rotation="10 MB", compression="zip")
    logger.info(OmegaConf.to_yaml(cfg))
    writer = SummaryWriter(log_dir=log_dir)
    set_seed(cfg.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------------- Model ---------------------------------- #
    if cfg.model.name == "DeMamba":
        model = XCLIP_DeMamba()
    elif cfg.model.name == "NPR":
        model = resnet50_npr()
    elif cfg.model.name == "TALL":
        model = TALL_SWIN(pretrained=True)
    elif cfg.model.name == "STIL":
        model = Det_STIL()
    else:
        raise NotImplementedError("Model Not supported")
    model = model.to(device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        logger.info(f"Using {cfg.trainer.num_gpus} GPUs for data parallelism.")
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])

    summary(model)
    
    if cfg.task_type == "standard":
        generation_models = get_generation_models(cfg.data.dataset_name)
        pn_ratio = 1
    elif cfg.task_type == "unbalance":
        generation_models = get_generation_models(cfg.data.dataset_name)
        pn_ratio = cfg.data.pn_ratio
    
    # ----------------------------------- Data ----------------------------------- #
    train_dataset = get_classifier_dataset(cfg.data, generation_model=cfg.data.generation_model, real_model=cfg.data.train_real_model, mode="train", load_len=cfg.data.train_load_len, pn_ratio=pn_ratio)
    train_loader = get_data_loader_for_classifer(cfg.data, train_dataset)
    val_dataloaders = {}

    generation_models = get_generation_models(cfg.data.dataset_name)
    for fake_model in generation_models["fake"]["val"]:
        for real_model in get_generation_models(cfg.data.dataset_name)["real"]["test"]:
            val_dataset = get_classifier_dataset(cfg.data, "val", generation_model=fake_model, real_model=real_model, pn_ratio=1, load_len=cfg.data.val_load_len)
            val_loader = get_data_loader_for_classifer(cfg.data, val_dataset)
            val_dataloaders[f"{fake_model}/{real_model}"] = val_loader
    
    # ----------------------------------- Train ---------------------------------- #
    global_step = 0
    best_val_auroc = - float("inf")
    best_val_acc = - float("inf")
    # loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    if cfg.trainer.optimizer.name == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    elif cfg.trainer.optimizer.name == "adamW":
        optimizer = optim.AdamW(model.parameters(), lr=cfg.trainer.optimizer.lr, weight_decay=cfg.trainer.optimizer.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.trainer.optimizer.name}")

    # train logics
    with tqdm(range(cfg.trainer.max_epochs), desc="Epochs", unit="epoch", position=0) as epoch_pbar:
        for epoch in epoch_pbar:
            train_results = train_classifer(model, train_loader, optimizer, criterion, device, writer, global_step, val_dataloaders, criterion, cfg)
            global_step = train_results["global_step"]
            train_info = " | ".join([f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}"
                         for key, value in train_results.items()])
            
            if (epoch+1) % cfg.trainer.val_check_interval == 0:
                headers, val_results = val_classifer(model, val_dataloaders, criterion, device, writer, global_step)
                val_info = tabulate(val_results, headers=headers, tablefmt="grid")
                
                val_auroc = val_results[-1][-1]
                val_acc = val_results[-1][-3]
                if val_acc > best_val_acc:
                    logger.info(f"Current acc ({val_acc:.4f}) > Best acc ({best_val_acc:.4f})")
                    best_val_acc = val_acc
                    best_model_save_path = os.path.join(cfg.save_ckpt_path, f"best_acc_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")
                if val_auroc > best_val_auroc:
                    logger.info(f"Current auroc ({val_auroc:.4f}) > Best auroc ({best_val_auroc:.4f})")
                    best_val_auroc = val_auroc
                    best_model_save_path = os.path.join(cfg.save_ckpt_path, f"best_auroc_ckpt.pth")
                    os.makedirs(os.path.dirname(best_model_save_path), exist_ok=True)
                    torch.save(model.state_dict(), best_model_save_path)
                    logger.success(f"Model saved at {best_model_save_path}")

                logger.info(
                    f"Epoch {epoch+1:2}/{cfg.trainer.max_epochs:2}\nTrain Info: {train_info} \n{val_info}"
                )
                
            epoch_pbar.set_postfix({
                **train_results,
            })

    csv_path = os.path.join(cfg.log_path, f"{cfg.experiment_name}/{cfg.data.dataset_name}/{cfg.model.name}_train_results.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    import pandas as pd
    df = pd.DataFrame(val_results, columns=headers)
    df.to_csv(csv_path)
    # save model ckpts
    model_save_path = os.path.join(cfg.save_ckpt_path, f"final_ckpt.pth")
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    logger.success(f"Model saved at {model_save_path}")
    
if __name__ == "__main__":
    main()