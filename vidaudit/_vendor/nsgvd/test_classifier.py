from utils.experiment_utils import set_seed
from data.utils import get_generation_models
from omegaconf import DictConfig
from models.discriminators import *
from utils.train_utils import *
from utils.data_utils import *
from models.tall import SingleSwinBlockDiscriminator, SingleSwinBlock
from models.demamba import XCLIP_DeMamba
from models.npr import resnet50_npr
from models.stil import Det_STIL
from models.tall import CrossAttnTripleTALL_SWIN, TALL_SWIN, GatedTripleTALL_SWIN, SwinTransformer
from omegaconf import OmegaConf
import torch.optim as optim
from loguru import logger
from tqdm import tqdm
import torch.nn as nn
import hydra
from tabulate import tabulate
from torchinfo import summary
import pandas as pd
import torch
import time  
from torch.utils.tensorboard import SummaryWriter
import os

@hydra.main(config_path="configs/experiments", config_name="standard-Pika-TALL.yaml", version_base=None)
def test(cfg: DictConfig):
    # ------------------------------- Setup Logging ------------------------------ #
    logger.info(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    # ------------------------------- Load Model --------------------------------- #
    logger.info(f"Loading model: {cfg.model.name}")
    if cfg.model.name == "DeMamba":
        model = XCLIP_DeMamba()
    elif cfg.model.name == "NPR":
        model = resnet50_npr()
    elif cfg.model.name == "TALL":
        model = TALL_SWIN(pretrained=False)  # We will load weights from checkpoint
    elif cfg.model.name == "STIL":
        model = Det_STIL()
    else:
        raise NotImplementedError(f"Model {cfg.model.name} is not supported.")

    # Load checkpoint
    ckpt_path = cfg.ckpt_path
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    logger.info(f"Loading checkpoint from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    if torch.cuda.device_count() >= cfg.trainer.num_gpus and cfg.trainer.num_gpus > 1:
        model = nn.DataParallel(model, device_ids=cfg.trainer.device_ids[:cfg.trainer.num_gpus])
        model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # ------------------------------- Load Test Data ----------------------------- #
    logger.info("Loading test data...")
    generation_models = get_generation_models(cfg.data.dataset_name)
    test_dataloaders = {}
    for fake_model in generation_models["fake"]["test"]:
        for real_model in generation_models["real"]["test"]:
            if fake_model == "Sora":
                load_len = 56
                test_dataset = get_classifier_dataset(
                    cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=load_len
                )
            else:
                test_dataset = get_classifier_dataset(
                    cfg.data, mode="test", generation_model=fake_model, real_model=real_model, load_len=cfg.data.test_load_len
                )
            test_loader = get_data_loader_for_classifer(cfg.data, test_dataset)
            test_dataloaders[f"{fake_model}"] = test_loader

    # ------------------------------- Evaluate Model ----------------------------- #
    results = []

    logger.info("Starting evaluation...")
    with torch.no_grad():
        for name, test_loader in tqdm(test_dataloaders.items(), desc="Testing", unit="dataset"):
            test_results = test_classifier(
                model, test_loader, device
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
    # headers = ["Dataset", "Recall", "F1", "Accuracy", "Precision", "Auroc"]
    headers = ["Dataset", "Recall", "Accuracy", "F1", "Auroc", "Precision"]
    # Calculate mean of all metrics
    mean_metrics = ["Avg."]
    for i in range(1, len(headers)):
        mean_metrics.append(sum(result[i] for result in results) / len(results))
    results.append(mean_metrics)
    
    csv_path = os.path.join(cfg.log_path, f"{cfg.save_csv_file}")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    # Save results to CSV with four decimal places
    df = pd.DataFrame(results, columns=headers)
    df = df.applymap(lambda x: f"{100*x:.2f}" if isinstance(x, float) else x)
    df = df.transpose()  # Transpose the table
    df.to_csv(csv_path, index=True, header=False)
    logger.success(f"Test results saved to {csv_path}")

    # Print results in table format
    logger.info("\n" + tabulate(results, headers=headers, tablefmt="grid"))

@torch.no_grad()
def test_classifier(model, test_dataloader, device):
    model.eval()
    all_labels = []
    all_predicted = []
    all_raw_preds = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating", leave=False, ncols=100):
            inputs, labels = batch
            inputs, labels = inputs.float().to(device), labels.to(device)

            logits = model(inputs)

            output_pred = logits[:,0].sigmoid().cpu()
            predicted = output_pred > 0.5
            
            # Collect labels and predictions for metric calculation
            all_labels.extend(labels.cpu().numpy())
            all_predicted.extend(predicted.cpu().numpy())
            all_raw_preds.extend(output_pred.cpu().numpy())

    # Calculate Precision, Recall, and F1 Score using sklearn
    precision = precision_score(all_labels, all_predicted)
    recall = recall_score(all_labels, all_predicted)
    f1 = f1_score(all_labels, all_predicted)
    acc = accuracy_score(all_labels, all_predicted)
    auroc = roc_auc_score(all_labels, all_raw_preds)
        
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
        "auroc": auroc,
    }
if __name__ == "__main__":
    test()