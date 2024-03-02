"""
This code incorporates modifications to software originally developed by the Turaga Lab, available at https://github.com/TuragaLab/DECODE. 
The original work is licensed under the GNU General Public License version 3.0 (GPL-3.0). 
We have adapted and extended this code under the same GPL-3.0 license. 
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


def cum_count_per_group(arr: torch.Tensor):

    def grp_range(counts: torch.Tensor):
        assert counts.dim() == 1

        idx = counts.cumsum(0)
        id_arr = torch.ones(idx[-1], dtype=int)
        id_arr[0] = 0
        id_arr[idx[:-1]] = -counts[:-1] + 1
        return id_arr.cumsum(0)

    if arr.numel() == 0:
        return arr

    _, cnt = torch.unique(arr, return_counts=True)
    return grp_range(cnt)[np.argsort(np.argsort(arr, kind='mergesort'), kind='mergesort')]

def create_trajectory(Na, Nl, initial_position, velocity, direction, f_start, f_last, t_end, show_trajectory):

    time_end = t_end
    time_start = f_start
    time_steps = f_last - f_start
    time_step = 1

    trajectory = torch.zeros((time_end+1, 2))
    trajectory[time_start] = initial_position

    for t in range(1, time_steps+1):
        
        position = trajectory[time_start+t-1]
        direction = direction + np.random.normal(scale=0.2, size=2)
        direction = direction / np.linalg.norm(direction)  
             
        position = position + direction * velocity * time_step
        # position = position + velocity * time_step 

        trajectory[time_start+t] = position

    position_list = trajectory[time_start:f_last+1]
    return trajectory, position_list

def save_high_trajectory(Na, Nl, Nt, initial_position, velocity, direction, f_start, f_last, t_end, show_trajectory):

    time_end = t_end
    time_start = f_start
    time_steps = f_last - f_start
    time_step = 1

    trajectory = torch.zeros((time_end, 2))
    trajectory[time_start] = initial_position


    for t in range(1, time_steps+1):

        position = trajectory[time_start+t-1]
        direction = direction + np.random.normal(scale=0.2, size=2)
        direction = direction / np.linalg.norm(direction)

        position = position + direction * velocity * time_step
        # position = position + velocity * time_step

        trajectory[time_start+t] = position
        
    position_list = trajectory[time_start:f_last+1]

    return position_list

def generate_image(indx, xpos, ypos, brightness, img, out_im, loc_stack_TR, frame, Na, Nl):

    xpos = np.round(xpos)
    ypos = np.round(ypos)

    img = (img / np.max(img)) * brightness[indx].numpy()

    w = img.shape[0] // 2
    h = img.shape[1] // 2
    p_w = w  # in case padding is changed
    p_h = h

    gaussCdf_tmp = np.zeros((Na + 2 * p_w, Nl + 2 * p_h, 3))

    xpmin = int(xpos) + p_w - w
    ypmin = int(ypos) + p_h - h
    xpmax = int(xpos) + p_w + w + 1
    ypmax = int(ypos) + p_h + h + 1


    if xpmax <= Na + 2 * p_w and ypmax <= Nl + 2 * p_h:
        gaussCdf_tmp[xpmin:xpmax, ypmin:ypmax, :] = img
        gaussCdf_tmp = gaussCdf_tmp / 255
        out_im[:, :, :, indx] = gaussCdf_tmp[p_w:-p_w, p_h:-p_h, :]
        loc_stack_TR[int(xpos), int(ypos), frame] += 1

    return out_im, loc_stack_TR