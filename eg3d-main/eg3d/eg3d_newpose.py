
import argparse
# import math
import os
import json
import pickle
import numpy as np
from PIL import Image
import sys
sys.path.append('../../')
from helper import to_img
import torch
import dnnlib
import legacy


def eg3d_newpose(args):
    device = torch.device('cuda')

    # load latent
    latent_path = f'{args.input_dir}/eg3d_projection/{args.input_name}.npy'
    print('load latent: '+latent_path)
    latent = torch.from_numpy(np.load(latent_path,allow_pickle=True)) # if 18[0,0,:])
    if len(latent.shape) == 2:
        latent = latent.repeat([1, 14, 1])
    latent = latent.to(device)

    # gan weight
    g_path = f'{args.input_dir}/eg3d_projection/{args.input_name}_ganweight.pkl'
    print('load gan weight: '+g_path)
    G = pickle.load(open(g_path, 'rb'))
    print(G)

    # camera pose
    f = open(f'{args.input_dir}/camera/{args.target_name}.json')
    image_camera = json.load(f)['labels']
    c = torch.from_numpy(np.array([image_camera[0][1]])).float().to(device) # torch.Size([1, 25])
    print('check camera',(image_camera[0][0]==f'{args.target_name}.png'),image_camera[0][0],f'{args.target_name}.png')

    # run
    img = G.synthesis(latent[:1], c[:1])['image'] # {'image', 'image_depth', 'image_raw'} # torch.Size([1, 3, 512, 512])
    save_name = f'{args.input_dir}/eg3d_warp/{args.input_name}_{args.target_name}_warpimg.png'
    os.makedirs(f'{args.input_dir}/eg3d_warp/', exist_ok = True)
    print('save_name',save_name)
    out = to_img(img).save(save_name)

    del G

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='10036')
    parser.add_argument('--target_name', type=str, default='59641')

    parser.add_argument("--input_dir", type=str, default="../../results/preprocessed/10036_59641/")

    args = parser.parse_args()

    eg3d_newpose(args)
