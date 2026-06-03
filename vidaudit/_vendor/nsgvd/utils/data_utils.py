from data.feature_dataset.score_feature_dataset import ScoreFeaturesDataset
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from data.video_dataset import VideoTensorDataset
from loguru import logger
from torch.utils.data import DataLoader

def get_score_datasets(data_cfg, mode, 
                       load_len=1000, generation_model = None,
                       real_model = None, filter=True, pn_ratio=1, filter_frames=False, resolution_size=224):
    fake_len = int(load_len * pn_ratio)
    fake_dataset = ScoreFeaturesDataset(
                score_config_path="./libs/eps_ad/imagnet.yml",
                score_args_path="./libs/eps_ad/args.yml",
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=data_cfg.generation_model if generation_model is None else generation_model,
                input_shape=(224, 224),
                process_batch_size=3,
                diffuse_steps=data_cfg.diffuse_steps,
                device="cuda",
                verbose=False,
                feature_type=data_cfg.feature_type,
                num_frames=8,
                mode=mode, 
                load_len=fake_len,
                filter_nsg=filter,
                filter_frames=filter_frames,
                resolution_size=resolution_size
                )
    real_dataset = ScoreFeaturesDataset(
                score_config_path="./libs/eps_ad/imagnet.yml",
                score_args_path="./libs/eps_ad/args.yml",
                data_path=data_cfg.data_path, 
                dataset_name=data_cfg.dataset_name,
                generation_model=real_model,
                input_shape=(224, 224),
                process_batch_size=3,
                diffuse_steps=data_cfg.diffuse_steps,
                device="cuda",
                verbose=False,
                feature_type=data_cfg.feature_type,
                num_frames=8,
                mode=mode, 
                load_len=load_len,
                filter_nsg=filter,
                filter_frames=filter_frames,
                resolution_size=resolution_size,
                start_idx=200,
                )
    return {"fake": fake_dataset, "real": real_dataset}

def get_data_loaders_for_mmd(data_cfg, datasets, batch_size):
    train_dataloaders = {}
    train_dataloaders["fake"]= DataLoader(datasets["fake"], batch_size=batch_size,
                        shuffle=True, num_workers=data_cfg.num_workers)
    train_dataloaders["real"] = DataLoader(datasets["real"], batch_size=batch_size,
                        shuffle=True, num_workers=data_cfg.num_workers)
    return train_dataloaders
  
def get_ref_dataloaders(data_cfg, ref_model_names, mode="test", filter_frames=False):
    ref_dataloaders = {}
    for ref_model_name in ref_model_names:
        ref_dataset = ScoreFeaturesDataset(
            score_config_path="./libs/eps_ad/imagnet.yml",
            score_args_path="./libs/eps_ad/args.yml",
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=ref_model_name,
            input_shape=(224, 224),
            process_batch_size=3,
            diffuse_steps=data_cfg.diffuse_steps,
            device="cuda",
            verbose=False,
            feature_type=data_cfg.feature_type,
            num_frames=8,
            mode=mode, 
            filter_frames=filter_frames,
            load_len=200,
            )
        ref_dataloaders[ref_model_name] = DataLoader(ref_dataset, batch_size=data_cfg.batch_size,
                        shuffle=False, num_workers=data_cfg.num_workers)
    return ref_dataloaders

def get_ref_dataloader(data_cfg, ref_model_name, mode="test", filter_frames=False, resolution_size=224):
    ref_dataset = ScoreFeaturesDataset(
        score_config_path="./libs/eps_ad/imagnet.yml",
        score_args_path="./libs/eps_ad/args.yml",
        data_path=data_cfg.data_path, 
        dataset_name=data_cfg.dataset_name,
        generation_model=ref_model_name,
        input_shape=(224, 224),
        process_batch_size=3,
        diffuse_steps=data_cfg.diffuse_steps,
        device="cuda",
        verbose=False,
        feature_type=data_cfg.feature_type,
        num_frames=8,
        mode=mode, 
        filter_frames=filter_frames,
        resolution_size=resolution_size,
        )
    return DataLoader(ref_dataset, batch_size=data_cfg.batch_size,
                        shuffle=False, num_workers=data_cfg.num_workers)
  
def get_classifier_dataset(data_cfg, mode, 
                            load_len=1000, generation_model = None,
                            real_model = None, pn_ratio = 1):
    feature_type = data_cfg.feature_type
    logger.info(f"Using feature type : {feature_type.upper()}")
    if feature_type == "image":
        real_dataset = VideoTensorDataset(
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=real_model,
            mode=mode, 
            num_frames=8,
            input_shape=(224, 224),
            load_len=load_len,
            )
        fake_len = int(load_len * pn_ratio)
        fake_dataset = VideoTensorDataset(
            data_path=data_cfg.data_path, 
            dataset_name=data_cfg.dataset_name,
            generation_model=generation_model,
            mode=mode, 
            num_frames=8,
            input_shape=(224, 224),
            load_len=fake_len,
            )
    return ConcatDataset([fake_dataset, real_dataset])
  
def get_data_loader_for_classifer(data_cfg, dataset):
    return DataLoader(dataset, batch_size=data_cfg.batch_size,
                        shuffle=True, num_workers=data_cfg.num_workers)