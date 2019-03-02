import os
import imageio
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import torch

class DRLoader:
    def __init__(self,root_dir, window_size, transforms, shuffle_f):
        assert os.path.exists(root_dir), root_dir+' not exists'
        self.window_size = window_size
        self.transforms = transforms
        
        classes = sorted(os.listdir(root_dir))
        vid_path,labels=[],[]
        for i,c in enumerate(classes):
            videos = sorted(os.listdir(root_dir+'/'+c))
            for vid in videos:
                ext = os.path.splitext(vid)[1]
                if ext not in ['.mpg']:
                    continue
                vid_path.append(os.path.join(root_dir,c,vid))
                labels.append(i)
                
        self.videos = vid_path
        self.labels = labels
        
        if shuffle_f:
            self.shuffle()
        
    def shuffle(self):
        z = zip(self.videos, self.labels)
        random.shuffle(z)
        self.videos, self.labels = [list(l) for l in zip(*z)]
        
    def __len__(self):
        return len(self.videos)
    
    def batches(self,batchsize):
        n = len(self.videos)
        for i in range(0,n-batchsize,batchsize):
            video_batch = self.videos[i:i+batchsize]
            label_batch = self.labels[i:i+batchsize]
            x = torch.zeros(batchsize, self.window_size, 3, 224, 224)
            y = torch.LongTensor(label_batch)
            for indx,vid in enumerate(video_batch):
                F = imageio.get_reader(vid)
                Frames=[]
                for f in F:
                    Frames.append(f)
                #F.close()
                n=len(Frames)
                if n>self.window_size:
                    rnd_idx = int((n-self.window_size)*np.random.rand())
                    frames=Frames[rnd_idx:rnd_idx+self.window_size]
                else:
                    frames=Frames
                    print('frames',len(frames))
                Window=None
                for img in frames:
                    img = self.transforms(Image.fromarray(img))
                    img = img.unsqueeze(0)
                    if type(Window)==type(None):
                        Window=img
                    else:
                        Window=torch.cat([Window,img],0)
                x[indx] = Window
            
            yield x,y