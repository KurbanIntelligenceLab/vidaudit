from data.utils import get_score_fn
from data.video_frames_dataset import VideoFramesDataset
from torch.utils.data import Dataset
from torchvision import transforms
from data.utils import *
from loguru import logger 
from PIL import Image
from tqdm import tqdm
import numpy as np
import os

def get_data(input_dir, idx, feature_type, filter_frames=False):
    if feature_type == "score":
        score = np.load(os.path.join(input_dir, f"score_{idx}.npy"))
        data = score
    elif feature_type == "velocity":
        velocity = np.load(os.path.join(input_dir, f"nsg_{idx}.npy")) # (T,C,H,W)
        if filter_frames:
            velocity = velocity[1:-1]
            data = velocity
        else:
            velocity[0] = velocity[0] * 100
            velocity[-1] = velocity[-1] * 100
            data = velocity
    elif feature_type == "frame_diff":
        # #############################################################################################
        # numerator = score # (B,T,C,H,W) 
        # denominator = (score * pixel_diff).sum(dim=(1,2,3)).view(T,1,1,1)  + epsilon # (B,T,1,1,1)
        # velocity = numerator / denominator
        # #############################################################################################
        f_diff_path = os.path.join(input_dir, f"f_diff_{idx}.npy")
        if os.path.exists(f_diff_path):
            frame_diff = np.load(f_diff_path)
        else:
            score = np.load(os.path.join(input_dir, f"score_{idx}.npy"))
            velocity = np.load(os.path.join(input_dir, f"nsg_{idx}.npy")) # (T,C,H,W)
            # velocity[0] = velocity[0] * 100
            # velocity[-1] = velocity[-1] * 100
            frame_diff = score.sum(axis=(1,2,3)) / velocity.sum(axis=(1,2,3)) # (T)
            assert frame_diff.shape == (score.shape[0],), f"Frame diff shape is {frame_diff.shape}, but score shape is {score.shape}"
            np.save(f_diff_path, frame_diff)
        # frame_diff = frame_diff[1:-1]
        # frame_diff = frame_diff / 1e4
        data = frame_diff
    else:
        raise NotImplementedError(f"Feature type of {feature_type} is not supported")
    data = torch.from_numpy(data).float()
    return data

class ScoreFeaturesDataset(Dataset):
    def __init__(self,data_path:str,
                 dataset_name:str="GenVideo",
                 generation_model:str="None",
                 verbose:bool=False,
                 mode:str="train",
                 num_frames: int=4,
                 input_shape: tuple=(256, 256),
                 diffuse_steps: int=5,
                 score_config_path: str="./libs/eps_ad/args.yml",
                 score_args_path: str="./libs/eps_ad/imagnet.yml",
                 regenerate: int=0,
                 process_batch_size: int=2,
                 pixel_difference: bool=True,
                 feature_type: str="all",
                 load_len: int=100,
                 device:str="cuda",
                 eps:bool=False, # Whether to sum score features
                 cache_all_diffuse_steps:bool=False, # Cache like (# #diffuse_steps,#num_frames,C,H,W)
                 filter_nsg: bool=False,
                 filter_frames: bool=False,
                 resolution_size: int=224,
                 start_idx: int=0,
                 ids_file: str=None
                 ):
        super().__init__()
        self.label = get_label_from_generation_model(generation_model)
        self.cache_all_diffuse_steps = cache_all_diffuse_steps
        self.process_batch_size = process_batch_size
        self.score_config_path = score_config_path
        self.pixel_difference = pixel_difference
        self.generation_model = generation_model
        self.score_args_path = score_args_path
        self.diffuse_steps = diffuse_steps
        self.dataset_name = dataset_name
        self.input_shape = input_shape
        self.num_frames = num_frames
        self.resolution_size = resolution_size
        self.start_idx = start_idx
        self.regenerate = regenerate
        self.data_path = data_path
        self.load_len = load_len
        self.verbose = verbose
        self.device = device
        self.mode = mode
        self.feature_type = feature_type
        self.filter_frames = filter_frames
        self.eps = eps
        self.ids_file = ids_file
        self.label = get_label_from_generation_model(self.generation_model)
        self.dataset = self._get_dataset() # Access dataset for `_cache_exists method` and `_extract_feature`
        self.video_ids = self.dataset.video_ids
        self._load_cache()
        if filter_nsg:
            self.video_ids = self.filter_nsg(self.video_ids)
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} / {self.generation_model}]")

    def filter_nsg(self, video_ids):
        filtered_ids = []
        for idx in tqdm(range(len(video_ids)), desc="Filtering NSG videos", ncols=100):
            feature = get_data(self.input_dir, self.video_ids[idx], self.feature_type)
            if feature.abs().mean() <= 10:  # Filter condition
                filtered_ids.append(self.video_ids[idx])
        return filtered_ids

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # Load features on-demand
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
                    
        return get_data(self.input_dir, self.video_ids[idx], self.feature_type, filter_frames=self.filter_frames), np.array([0 if self.label=="real" else 1], dtype=np.float32)

    def _load_cache(self):
        self._setup_cache_paths()
        feature_exists, missing_ids = self._cache_exists(self.input_dir)
        if feature_exists and not self.regenerate:
            if self.verbose:
                logger.success("Cache score features exists, loading from cache...")    
        else:
            if self.regenerate:
                logger.warning("Regenerate score features cache...")
            else:
                logger.info(f"No score features cache, extract score features... of {self.dataset_name} / {self.mode} / {self.generation_model}")
            self._extract_feature(missing_ids)
        
    def _load_features(self):
        outputs = np.load(self.output_file)
        if self.load_len is not None:
            outputs = outputs[:self.load_len]
        
        return outputs
        
    def _setup_score_fn(self):
        device = self.device
        args_path = self.score_args_path
        config_path = self.score_config_path
        score_fn = get_score_fn(device, args_path, config_path)
        self.score_fn = score_fn

    def _cache_exists(self, input_dir):
        # Input features exists?
        entries = os.listdir(input_dir)
        missing_ids = []
        for vid in self.video_ids:
            score_file = f"score_{vid}.npy"
            if score_file not in entries:
                missing_ids.append(vid)
        if missing_ids:
            logger.warning(f"Missing feature files for video ids: {missing_ids}")
        self.missing_ids = missing_ids
        input_cache_exist = len(missing_ids) == 0
        return input_cache_exist, missing_ids
    
    def _setup_cache_paths(self):
        path_type = normalize_name(f"steps_{self.diffuse_steps}")
        self.cache_path = f"{self.data_path}/nsg-vd/{path_type}/{self.label}/{self.generation_model}/{self.mode}"
        
        self.input_dir = f"{self.cache_path}/input"
        self.output_file = f"{self.cache_path}/output.npy"
        print(self.cache_path)
        os.makedirs(self.input_dir, exist_ok=True)
        
    def _preprocess_frame_from_path(self, frame_path):
        frame_img = Image.open(frame_path)
        frame_img = frame_img.resize(self.input_shape, Image.Resampling.LANCZOS) # imagnet's shape 224x224
        
        frame_tensor = torch.from_numpy(np.array(frame_img)).float() # (h,w,3)
        frame_tensor = frame_tensor / 255.0 # normalize to (0,1)
        frame_tensor = frame_tensor.permute(2, 0, 1) # (3,h,w)
        return frame_tensor # (3,h,w)
    
    @torch.no_grad()
    def _get_score_velocity(self, score, image, epsilon=1e-10):
        """
        args:
            score: #(B,T,C,H,W)
            image: #(B,T,C,H,W)
        return:
            velocity: #(B,T,C,H,W)
        """
        B,T,C,H,W = score.shape
        
        pixel_diff = torch.zeros((B,T,C,H,W)) # (B,T,C,H,W)
        for t in range(1, T-1):
            pixel_diff[:,t] = (image[:,t+1,...] - image[:,t-1,...]) / 2 # (B,C,H,W)
        pixel_diff[:,0] = (image[:,1,...] - image[:,0,...]) # (B,C,H,W)
        pixel_diff[:,-1] = (image[:,-1,...] - image[:,-2,...]) # (B,C,H,W)
        pixel_diff = pixel_diff.to(score.device)
        
        numerator = score # (B,T,C,H,W) 
        denominator = (score * pixel_diff).sum(dim=(2,3,4)).view(B,T,1,1,1)  + epsilon # (B,T,1,1,1)
        assert denominator.shape == (B,T,1,1,1)
        velocity = numerator / denominator
        assert velocity.shape == (B,T,C,H,W)
        return velocity
    
    @torch.no_grad()
    def _get_score_features(self, frame_tensor, batch_size):
        """
        args: 
            frame_tensor: (B,T,C,H,W)
            batch_size: B
        return:
            scores: (B,T,C,H,W)
            velocity: (B,T,C,H,W)
        """
        t_value = (self.diffuse_steps+1)/1000
        curr_t_temp = torch.tensor(t_value,device=frame_tensor.device)

        image = frame_tensor.view(batch_size, -1, 3, 224, 224)
        frame_tensor = F.interpolate(frame_tensor, size=(self.resolution_size, self.resolution_size), mode='bilinear', align_corners=False) # (B,T,C,H,W)
        score = self.score_fn(2*frame_tensor-1, curr_t_temp.expand(frame_tensor.shape[0]))
        score, _ = torch.split(score, score.shape[1]//2, dim=1) # (B*T,C,H,W)
        
        score = F.interpolate(score, size=(self.input_shape[0], self.input_shape[1]), mode='bilinear', align_corners=False) # (B,T,C,H,W)

        image = image.view(batch_size, -1, *score.shape[1:]) # (B,T,C,H,W)
        score = score.view(batch_size, -1, *score.shape[1:]) # (B,T,C,H,W)
        
        velocity = self._get_score_velocity(score, image) # (B,T,C,H,W)
        
        return score, velocity
    
    @torch.no_grad()
    def _extract_feature(self, missing_ids):
        self.features_label = []
        self._setup_score_fn()
        video_frames = []
        
        for video_id in missing_ids:
            sample = self.dataset.get_video_frames(video_id)
            if sample["num_frames"] < self.num_frames:
                continue
            video_frames.append((video_id, sample["frames"][:self.num_frames]))
        
        # Extract features and cache inputs
        batch_size = self.process_batch_size
        
        # Get imags from paths of frames
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor() as executor:
            all_frames = list(tqdm(
                executor.map(self._preprocess_video, video_frames),# video_frames: paths for frames images
                total=len(video_frames),
                desc="Preprocessing frames",
                ncols=100,
            )) # process to tensors like (T, C, H, W)
        
        for chunk_idx in tqdm(range(0, len(all_frames), batch_size)):
            chunk = all_frames[chunk_idx:chunk_idx + batch_size]
            ids = [item[0] for item in chunk]
            chunk = [item[1] for item in chunk]
            video_batch = torch.stack(chunk).to(self.device)  # (B, T, C, H, W)
            batch_size, num_frames = video_batch.shape[:2]
            flat_batch = video_batch.view(-1, *video_batch.shape[-3:])  # (B, T, C, H, W)
            
            score_features, velocity_features = self._get_score_features(flat_batch, batch_size) # (B, T, C, H, W)
                
            score_features = score_features.to("cpu").detach().numpy()
            velocity_features = velocity_features.to("cpu").detach().numpy()
            self._cache_features(score_features, velocity_features, ids, batch_size)

    def _preprocess_video(self, input):
        video_id, frame_paths = input
        return (video_id, torch.stack([self._preprocess_frame_from_path(p) for p in frame_paths],dim=0))  # (T, C, H, W) T is number of frames
    
    def _cache_output(self):
        np.save(self.output_file, self.features_label)
        
    def _cache_features(self, score_features, velocity_features, ids, num_features):
        if score_features is not None:
            assert score_features[0].shape.__len__() == 4, "Features shape of score should be like (T,C,H,W) 4D, but got {}".format(score_features[0].shape)
            for i in range(num_features):
                np.save(f"{self.input_dir}/score_{ids[i]}.npy", score_features[i])
        if velocity_features is not None:
            assert velocity_features[0].shape.__len__() == 4, "Features shape of score should be like (T,C,H,W) 4D, but got {}".format(score_features[0].shape)
            for i in range(num_features):
                np.save(f"{self.input_dir}/nsg_{ids[i]}.npy", velocity_features[i])

    def _get_diff(self,idx):
        score = get_data(self.input_dir, idx, "score") # (T,C,H,W)
        velocity = get_data(self.input_dir, idx, "velocity") # (T,C,H,W)
        # velocity(T,C,H,W) = score(T,C,H,W) * diff(T,1,1,1)
        diff = velocity.sum(axis=(1,2,3)) / score.sum(axis=(1,2,3)) # (T)
        return diff

    def _get_dataset(self):
        return VideoFramesDataset(
            data_path=self.data_path, 
            dataset_name=self.dataset_name, 
            generation_model=self.generation_model, 
            mode=self.mode, 
            verbose=self.verbose,
            num_frames=self.num_frames,
            len_load=self.load_len,
            start_idx=self.start_idx,
            ids_file=self.ids_file)