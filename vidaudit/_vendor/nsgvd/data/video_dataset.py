import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Optional, Callable
from data.utils import *
from pprint import pprint as print
from loguru import logger
from torchvision import transforms

class VideoTensorDataset(Dataset):
    def __init__(
        self,
        data_path:str,
        dataset_name:str="GenVideo",
        generation_model:str="None",
        verbose:bool=False,
        mode:str="train",
        num_frames: int=8,
        transform: Optional[Callable] = None,
        load_len: int=None,
        input_shape: tuple=(224, 224)):
        super().__init__()
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.generation_model = generation_model
        self.verbose = verbose
        self.mode = mode
        self.num_frames = num_frames
        self.input_shape = input_shape
        self.transform = transforms.Compose([
                            transforms.Resize(input_shape),
                            transforms.ToTensor(),
                            transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],  # Mean of ImageNet
                            std=[0.229, 0.224, 0.225]),    # Std of ImageNet
                            ])
        
        generation_models = get_all_generation_models(self.dataset_name)
        self.label = get_label_from_generation_model(self.generation_model)
        
        # different generation AI models, e.g., sora, zeroscope, ...
        assert self.generation_model in generation_models, f"generation model {self.generation_model} is not supported in {self.dataset_name} dataset"
        
        # data_dir
        self.base_dir = os.path.join(
            self.data_path, 'video_frames', self.label, self.generation_model, self.mode
        )

        # all videos path
        self.video_dirs = sorted(
            [os.path.join(self.base_dir, d) for d in os.listdir(self.base_dir)],
            key=lambda x: os.path.basename(x)  # sort by video number
        )

        # all video frames path
        self.video_frame_paths = []
        for video_dir in self.video_dirs:
            # Check if the directory exists and contains .jpg files
            if os.path.isdir(video_dir):
                frames = sorted(
                    [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith('.jpg')],
                    key=lambda x: int(os.path.splitext(os.path.basename(x))[0].replace('frame', ''))
                )
                # Only add the frames list if it is not empty
                if len(frames) == self.num_frames:
                    self.video_frame_paths.append(frames)
        if load_len is not None:
            self.video_frame_paths = self.video_frame_paths[:load_len]
        logger.success(f"[{self.dataset_name} / {self.mode} / {len(self)} / {self.generation_model}]")
            
    def __len__(self) -> int:
        return len(self.video_frame_paths)

    def __getitem__(self, idx: int) -> np.ndarray:
        frame_paths = self.video_frame_paths[idx]
        
        video_data = []
        for frame_path in frame_paths:
            img = Image.open(frame_path).convert('RGB')
            
            if self.transform:
                img = self.transform(img)
            
            # convert (H, W, C) to (C, H, W)
            img_array = np.array(img)
            video_data.append(img_array)
        
        # merge frames to video
        return np.stack(video_data, axis=0), np.array([0 if self.label=="real" else 1], dtype=np.float32)
    
if __name__ == "__main__":
    
    dataset = VideoTensorDataset(data_path="/U_20240905_ZSH_SMIL/lzh/Data/GenVideo",dataset_name="GenVideo",  generation_model="Sora", mode="test", input_shape=(224,224))
    print(dataset[0][0].shape)
    print(dataset[0][1])
    