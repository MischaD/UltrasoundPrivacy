import os
from torch.utils import data
import torch
import pandas as pd
import numpy as np
import random
import os
import cv2
import pandas as pd
from PIL import Image
from einops import rearrange


PHASE_TO_SPLIT = {"training": "TRAIN", "validation": "VAL", "testing": "TEST"}


class SimaseUSDataset(data.Dataset):
    def __init__(self, 
                 phase='training', 
                 transform=None,
                 latents_csv='./', 
                 training_latents_base_path="./", 
                 in_memory=True, 
                 generator_seed=None):
        self.phase = phase
        self.training_latents_base_path = training_latents_base_path

        self.in_memory = in_memory
        self.videos = []

        self.df = pd.read_csv(latents_csv)
        self.df = self.df[self.df["Split"] == PHASE_TO_SPLIT[self.phase]].reset_index(drop=True)

        self.transform = transform

        if generator_seed is None: 
            self.generator = np.random.default_rng() 
            #unseeded
        else:             
            self.generator_seed = generator_seed
            print(f"Set {self.phase} dataset seed to {self.generator_seed}")

        if self.in_memory: 
            self.load_videos()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        
        vid_a = torch.clone(self.get_vid(index))
        if self.generator.uniform() < 0.5: 
            vid_b = torch.clone(self.get_vid((index + self.generator.integers(low=1, high=len(self))) % len(self))) # random different vid
            y = 0.0
        else: 
            vid_b = torch.clone(vid_a)
            y = 1.0

        if self.transform is not None:
            vid_a = self.transform(vid_a)
            vid_b = self.transform(vid_b)

        frame_a = self.generator.integers(len(vid_a))
        frame_b = (frame_a + self.generator.integers(low=1, high=len(vid_b))) % len(vid_b)
        #print(f"Dataloader: framea {frame_a} - frame_b {frame_b} - y: {y}")
        return vid_a[frame_a], vid_b[frame_b], y

    def reset_generator(self): 
        self.generator = np.random.default_rng(self.generator_seed) 

    def get_vid(self, index, from_disk=False): 
        if self.in_memory and not from_disk: 
            return self.videos[index]
        else: 
            return torch.load(os.path.join(self.training_latents_base_path, self.df.iloc[index]["FileName"] + ".pt"))

    def load_videos(self): 
        self.videos = []
        print("Preloading videos")
        for i in range(len(self)):
            self.videos.append(self.get_vid(i, from_disk=True))


class SimaseImageSpaceUSDataset(SimaseUSDataset):
    def __init__(self, *args, **kwargs): 
        super().__init__(*args, **kwargs)

    def get_vid(self, index, from_disk=False): 
        if self.in_memory and not from_disk: 
            return self.videos[index]
        else: 
            file_name = self.df.iloc[index]["FileName"] + ".avi" if not self.df.iloc[index]["FileName"].endswith(".avi") else self.df.iloc[index]["FileName"]
            video_file_path = os.path.join(self.training_latents_base_path, file_name)
            video_frames = load_avi(video_file_path)
            video_frames = torch.tensor(np.stack(video_frames))
            video_frames = video_frames / 127.5 - 1 
            return rearrange(video_frames, "b h w c -> b c h w")


def load_avi(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames
