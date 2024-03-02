"""
This code incorporates modifications to software originally developed by the Turaga Lab, available at https://github.com/TuragaLab/DECODE. 
The original work is licensed under the GNU General Public License version 3.0 (GPL-3.0). 
We have adapted and extended this code under the same GPL-3.0 license. 
"""

import torch
import numpy as np
from helpers import cum_count_per_group, save_high_trajectory


def sample_mb_params (n, b_mean, b_std, t_dur_min, t_dur_max, t_start_min, t_start_max, t_end,
                       v_min, v_max, Na, Nl, Nt_in, epoch=None):
    
    t_dur = torch.distributions.uniform.Uniform(t_dur_min, t_dur_max) 
    blinktime = t_dur.sample((n,))

    t_start = torch.distributions.uniform.Uniform(t_start_min, t_start_max)
    t0 = t_start.sample((n,))

    te = t0 + blinktime - 1

    id = torch.arange(n).long()

    frame_start = torch.floor(t0).long()
    frame_last = torch.floor(te).long()
    frame_count = (frame_last - frame_start).long()

    xy = torch.rand((n,2)) * Na
    vel_dis = torch.distributions.uniform.Uniform(v_min, v_max) 
    vel = vel_dis.sample((n,))

    brightness_dis = torch.distributions.normal.Normal(b_mean, b_std)
    brightness = brightness_dis.sample((n,))

    direction = (torch.rand((n,2)) - 0.5) * 2 

    id_long = id.repeat_interleave(frame_count + 1, dim=0)
    vel_long = vel.repeat_interleave(frame_count + 1, dim=0)
    brightness_long = brightness.repeat_interleave(frame_count + 1, dim=0)
    frame_ix = frame_start.repeat_interleave(frame_count + 1, dim=0) + cum_count_per_group(id_long) 

    xy_final = []
    for id in range(n):
        position_list = save_high_trajectory(Na, Nl, Nt_in, xy[id], vel[id], direction[id], frame_start[id], frame_last[id], t_end, show_trajectory = 0)
        if id == 0:
            xy_final = position_list.cpu().detach().numpy()
        else: 
            xy_final = np.concatenate((xy_final, position_list.cpu().detach().numpy()), axis=0)
    MB_xy = torch.from_numpy(xy_final)
    MB_subset = MBSet(MB_xy, vel_long, brightness_long, frame_ix, id_long)

    return MB_subset



class MBSet:
    """
    MB Class, set of MBs and parameters
    """

    def __init__(self, xy: torch.Tensor, vel: torch.Tensor, brightness: torch.Tensor, frame_ix: torch.LongTensor,
                 id: torch.LongTensor = None):

        self.xy = xy
        self.vel = vel
        self.brightness = brightness
        self.frame_ix = frame_ix
        self.id = id 

    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):

        if self.n <= len(self) - 1:
            self.n += 1
            return self._get_subset(self.n - 1)
        else:
            raise StopIteration

    def __getitem__(self, item):
    
        if isinstance(item, int) and item >= len(self):
            raise IndexError(f"Index {item} out of bounds of MBset of size {len(self)}")

        return self._get_subset(item)
    
    def _get_subset(self, ix):

        if isinstance(ix, int):
            ix = [ix]

        if not isinstance(ix, torch.BoolTensor) and isinstance(ix, torch.Tensor) and ix.numel() == 1:
            ix = [int(ix)]

        if isinstance(ix, (np.ndarray, np.generic)) and ix.size == 1: 
            ix = [int(ix)]
        return MBSet(xy=self.xy[ix], vel=self.vel[ix], brightness=self.brightness[ix], frame_ix=self.frame_ix[ix], id=self.id[ix])

    def get_subset_frame(self, frame_start, frame_end, frame_ix_shift=None):

        ix = (self.frame_ix >= frame_start) * (self.frame_ix <= frame_end)
        mb = self[ix]

        if not frame_ix_shift:
            return mb
        elif len(mb) != 0: 
            mb.frame_ix += frame_ix_shift 

        return mb
    
