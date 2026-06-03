from data.utils import *
from torch.utils.data import Dataset
from  loguru import logger
from glob import glob
import numpy as np
import torch
import os
from moviepy import VideoFileClip
import multiprocessing

class VideoFramesDataset(Dataset):
    """
    VideoFramesDataset for different video datasets
        1. Dataset consists of multiple video folers
        2. Video folder consists of multiple frames like `frame1.jpg`, `frame2.jpg`, ... `frameN.jpg`
        3. All videos in dataset has the same label: real or fake
    """
    def __init__(self, 
                 data_path:str,
                 dataset_name:str="GenVideo",
                 generation_model:str="None",
                 verbose:bool=False,
                 mode:str="test",
                 len_load: int=None,
                 num_frames: int=8,
                 start_idx: int=0,
                 ids_file: str=None):
        super()
        assert mode in ["train", "test", "val"], f"Mode {mode} is not supported"
        assert dataset_name in ["GenVideo"], f"Dataset {dataset_name} is not supported"
        assert os.path.exists(data_path), f"Data path {data_path} does not exist"
        assert len_load is None or len_load > 0, f"len_load should be None or positive integer"
        
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.num_frames = num_frames

        generation_models = get_all_generation_models(self.dataset_name)
        # different generation AI models, e.g., sora, zeroscope, ...
        self.generation_model = generation_model
        assert self.generation_model in generation_models, f"generation model {self.generation_model} is not supported in {self.dataset_name} dataset"
        # train / test / val
        self.mode = mode
        self.len_load = len_load
        self.start_idx = start_idx
        # real or fake
        self.label = get_label_from_generation_model(self.generation_model)
        self.verbose = verbose
        self.ids_file = ids_file
        self._setup_dataset()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # single video consists of multiple frames
        frames = glob(os.path.join(self.paths[idx],"*"))
        # Sort the list of frame paths
        frames.sort()  
        # logger.debug(frames)

        fake = self.fakes[idx]

        # only prepare frames paths and label for feature datasets
        sample = {
            "frames": frames, # ["frame1.jpg", "frame2.jpg", ...]
            "fake": fake, # 0 for real, 1 for fake
            "num_frames": len(frames) # number of frames in frames
        }
        return sample

    def get_video_frames(self, video_id):
        # single video consists of multiple frames
        frames = glob(os.path.join(os.path.join(self.frame_dir, video_id),"*"))
        # Sort the list of frame paths
        frames.sort()  
        # logger.debug(frames)

        # only prepare frames paths and label for feature datasets
        sample = {
            "frames": frames, # ["frame1.jpg", "frame2.jpg", ...]
            "fake": 0, # 0 for real, 1 for fake
            "num_frames": len(frames) # number of frames in frames
        }
        return sample

    def _setup_dataset(self,):
        # data_path/real(or fake)/GEN2/train(or test, val)/*
        assert os.path.exists(self.data_path), f"Data path {self.data_path} does not exist"
        if self.ids_file is not None:
            video_ids_txt = self.ids_file
            assert os.path.exists(video_ids_txt), f"video_ids_txt not found"
            with open(video_ids_txt, "r") as f:
                video_ids = [line.strip() + ".mp4" for line in f if line.strip()]
        else:
            video_ids_txt = os.path.join(self.data_path, "split", self.label, self.generation_model, f"{self.mode}_ids.txt")
            assert os.path.exists(video_ids_txt), f"video_ids_txt not found"
            with open(video_ids_txt, "r") as f:
                video_ids = [line.strip() for line in f if line.strip()]
        video_ids = [vid for vid in video_ids if vid.endswith('.mp4')]
        video_ids = video_ids[self.start_idx:self.start_idx + self.len_load]
        video_ids = [video_id.split('.')[0] for video_id in video_ids]
        assert len(video_ids) <= self.len_load
        logger.info(f"len of video_ids is {len(video_ids)}")
        # {Dataset e.g., GenVideo}/video_frames/{label e.g., fake or real}/{generation_model e.g., Sora}/{self.mode}/{video_id e.g., Sora_1}/frames{1-8}.jpg
        video_dir = os.path.join(self.data_path, "video", self.label, self.generation_model)
        frame_dir = os.path.join(self.data_path, "video_frames", self.label, self.generation_model, self.mode)
        unproceesed_ids = [
                    video_id for video_id in video_ids
                    if not os.path.isdir(os.path.join(frame_dir, video_id)) or len(os.listdir(os.path.join(frame_dir, video_id))) != self.num_frames
        ]
        if len(unproceesed_ids) > 0:
            process_video2frames(video_dir, unproceesed_ids, self.num_frames, frame_dir)
        video_ids = [
                    video_id for video_id in video_ids
                    if os.path.isdir(os.path.join(frame_dir, video_id)) and len(os.listdir(os.path.join(frame_dir, video_id))) == self.num_frames
        ]
        self.paths = [
                    os.path.join(frame_dir, video_id) for video_id in video_ids
        ]
        self.frame_dir = frame_dir
        logger.info(f"Len of self.paths {len(self.paths)}")
        assert len(self.paths) > 0, f"No video found in {self.data_path}/frames/{self.label}/{self.generation_model}/{self.mode}"
        
        # 0 for real, 1 for fake
        if self.label == "real":
            self.fakes = np.zeros(len(self.paths)) # real 
        elif self.label == "fake":
            self.fakes = np.ones(len(self.paths)) # fake
        if self.verbose:
            logger.success(f"Successfully load Dataset [name/{self.dataset_name}|model/{self.generation_model}|mode/{self.mode}|label/{self.label}|num_videos/{len(self)}]")
        self.video_ids = video_ids

def get_video_length(file_path):
    """get length of video by seconds"""
    video = VideoFileClip(file_path)
    return video.duration

def get_video_frame_count(file_path):
    """get total frame count of video"""
    video = VideoFileClip(file_path)
    return int(video.fps * video.duration)

def process_video(args):
    """process single video"""
    video_path, num_frames, output_dir = args  # index, video_path
    if video_path.endswith(".mp4"):
        try:
            output_dir = output_dir 
            os.makedirs(output_dir, exist_ok=True)

            total_frames = get_video_frame_count(video_path)
            frame_interval = max(1, total_frames // num_frames)

            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vf select='not(mod(n\,{frame_interval}))',setpts=N/FRAME_RATE/TB "
                f"-vframes {num_frames} {output_dir}/frame%d.jpg"
                # f" > /dev/null 2>&1"
            )
            ret = os.system(cmd)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
    elif video_path.endswith(".gif"):
        try:
            output_dir = os.path.join(output_dir)
            os.makedirs(output_dir, exist_ok=True)

            total_frames = get_video_frame_count(video_path)
            frame_interval = max(1, total_frames // num_frames)

            cmd = (
                f"ffmpeg -i {video_path} "
                f"-vf select='not(mod(n\,{frame_interval}))',setpts=N/FRAME_RATE/TB "
                f"-vframes {num_frames} {output_dir}/frame%d.jpg"
                f" > /dev/null 2>&1"
            )
            os.system(cmd)
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")

def process_video2frames(dir, ids, num_frames, output_base_dir):
    video_args = [(os.path.join(dir, f"{id}.mp4"), num_frames, os.path.join(output_base_dir, id)) for id in ids]
    logger.info(f"Processing {len(ids)} videos")
    pool = multiprocessing.Pool(processes=24)
    pool.map(process_video, video_args)
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    for dataset_name in ["GenVideo"]:
        for generation_model in get_all_generation_models(dataset_name):
            for mode in ["train", "test", "val"]:
                dataset = VideoFramesDataset(
                    data_path=f"../Data/{dataset_name}", 
                    generation_model=generation_model, 
                    dataset_name=dataset_name, 
                    mode=mode, 
                    len_load=4,
                    )
            # print(f"Video 0: {dataset[0]}")