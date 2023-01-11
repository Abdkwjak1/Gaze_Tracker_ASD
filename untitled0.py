# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 21:53:26 2021

@author: Abd_Kwjak
"""

import numpy as np 
import os
import skvideo.io as skv


mm2px_scaling = np.linalg.norm([480,640]) / np.linalg.norm([3.6,4.8])
video_name_with_ext = os.path.split("AABB2.mp4")[1]
video_name_root, ext = os.path.splitext(video_name_with_ext)
vreader = skv.FFmpegReader("AABB2.mp4")
m, w, h, channels = vreader.getShape()
image_scaling_factor = np.linalg.norm((240, 320)) / np.linalg.norm((h, w))

batch_size=32
initial_frame, final_frame = 0, m
final_batch_size = m % batch_size
final_batch_idx = m - final_batch_size
X_batch = np.zeros((batch_size, 240, 320, 3))
X_batch_final = np.zeros((m % batch_size, 240, 320, 3))
for idx, frame in enumerate(vreader.nextFrame()):
    print(idx % 32)