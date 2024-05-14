
import argparse
import math
import os
import sys
sys.path.append('../../')
from helper import create_mask_torch,create_img_torch,to_img,get_morphed_w_code

import torch
torch.cuda.empty_cache()
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import json
import glob

from networks import lpips
import numpy as np

import pickle
sys.path.append('eg3d-main/eg3d')
import dnnlib
import legacy

import scipy.ndimage
from torch.nn import functional as F
from torchvision import transforms

def eg3d_projection(args):

    # normal optimization
    w_avg_samples = 10000
    psi = 1
    truncation_cutoff = 14
    initial_w = None
    #pti
    first_inv_lr = 5e-3

    input_name = os.path.splitext(args.input_filename)[0]
    output_name = f'{args.input_dir}/eg3d_projection/{input_name}'
    os.makedirs(f'{args.input_dir}/eg3d_projection/', exist_ok = True)

    device = torch.device('cuda')

    # load image
    print('load img')
    image = create_img_torch(Image.open(f'{args.input_dir}/crop/{args.input_filename}'),device=device,is_pil=1)
    mask = create_mask_torch(Image.open(f'{args.input_dir}/crop_mask_ignore/{input_name}.png'),device=device,resize=256,is_pil=1)

    # camera pose
    print('load camera pose')
    f = open(f'{args.input_dir}/camera/{input_name}.json')
    image_camera = json.load(f)['labels']
    c = torch.from_numpy(np.array([image_camera[0][1]])).float().to(device) # torch.Size([1, 25])
    # print('c size ([1, 25]):',c.size())
    print('check camera',(image_camera[0][0]==args.input_filename),image_camera[0][0],args.input_filename)

    batch, channel, height, width = 1,3,1024,1024
    factor = height // 256


    print('first_inv_type',args.first_inv_type)

    network_pkl = args.network

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    # G.synthesis(start_w[:1], c[:1],noise_mode='random / const / none ')

    # print('noise')
    # Setup noise inputs.
    noise_bufs = {name: buf for (name, buf) in G.backbone.synthesis.named_buffers() if 'noise_const' in name}

    print('Calculate w_avg...')
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    list_w_samples = []
    for i in range(w_avg_samples):
        w_samples = G.mapping(z=torch.from_numpy(z_samples[i:i+1,:]).to(device), c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)
        list_w_samples.append(w_samples)
        # G.mapping(torch.from_numpy(z_samples).to(device), None)  # [N, L, C]
    w_all = torch.cat(list_w_samples,dim=0)
    w_avg = w_all.mean(dim=0, keepdims=True)
    w_std = (torch.sum((w_all - w_avg) ** 2) / w_avg_samples) ** 0.5
    # torch.save(w_avg, '../../checkpoints/w_avg.pt')
    # torch.save(w_std, '../../checkpoints/w_std.pt')

            # save w_avg_samples ................................................................

    print('lpips')
    # LPIPS MODEL
    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True
    )


    print('Setup w/wp projection...')
    start_w = initial_w if initial_w is not None else w_avg

    print('init')
    # Init
    w_opt = start_w.detach().clone().to(device)
    if args.first_inv_type == 'w':
        # wp to w
        w_opt = w_opt[:,0,:]
    w_opt.requires_grad=True
    # torch.tensor(start_w, dtype=torch.float32, device=device,requires_grad=True)  # pylint: disable=not-callable
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=first_inv_lr)

    latent_in_initial = w_opt.detach().clone().to(device)
    latent_in_initial.requires_grad=False
    l_latent_loss = args.latent_loss
    l_lambda_ref = args.lambda_ref
    l_mse_ref = args.mse_ref

    initial_learning_rate=0.01
    initial_noise_factor=0.05
    lr_rampdown_length=0.25
    lr_rampup_length=0.05
    noise_ramp_length=0.75
    noise_regularize = 1e5
    # regularize_noise_weight=1e5


    num_steps = args.first_inv_steps
    i_sum = 0
    pbar = tqdm(range(num_steps), position=0, leave=True)
    for step in pbar:
        i_sum = i_sum + 1
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        if args.first_inv_type == 'w':
            ## w
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            latent_n = (w_opt + w_noise).repeat([1, 14, 1])
        else:
            ## w+
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            latent_n = (w_opt + w_noise)

        synth_images = G.synthesis(latent_n[:1], c[:1], noise_mode='const')['image']

        # # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        # # -1 to 1 -> 0 to 255
        # synth_images = (synth_images + 1) * (255 / 2)
        # if synth_images.shape[2] > 256:
        O = F.interpolate(synth_images, size=(256, 256), mode='area')

        # Noise regularization.
        n_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                n_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                n_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        n_loss = n_loss * noise_regularize

        mse_refrec_loss = (((image*mask - O*mask)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum()) * l_mse_ref
        refrec_loss = l_lambda_ref * percept(O*mask, image*mask,mask=mask) # ploy
        latent_loss = ((latent_in_initial-latent_n).pow(2).clamp(min=2e-3).mean()) * l_latent_loss

        loss = (refrec_loss + n_loss) + mse_refrec_loss + latent_loss

        pbar.set_description(
                (
                    f"i: {i_sum:4d};"
                    f"p: {refrec_loss.item():.2f};"
                    f"m: {mse_refrec_loss.item():.2f};"
                    f"la: {latent_loss.item():.2f};"
                )
            )

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                # buf *= buf.square().mean().rsqrt()
                buf *= (buf**2).mean().rsqrt()

    np.save(f'{output_name}.npy',w_opt.cpu().detach().numpy())
    synth_images = G.synthesis(latent_n[:1], c[:1], noise_mode='const')['image']
    to_img(synth_images).save(f'{output_name}.png')

    print('done w/wp projection')
    print(f'{output_name}.png')


    ##################################################################################
    w_opt.requires_grad = False
    if len(w_opt.shape) == 2:
        w_opt = w_opt.detach().repeat([1, 14, 1])

    for buf in noise_bufs.values():
        # buf[:] = torch.randn_like(buf)
        buf.requires_grad = False

    ## change gan weight
    print('start pti...')

    pti_learning_rate = 3e-4
    ## Locality regularization
    latent_ball_num_of_samples = 1
    locality_regularization_interval = 1
    use_locality_regularization = False
    regulizer_l2_lambda_ = 0.1
    regulizer_lpips_lambda_ = 0.1
    regulizer_alpha = 30
    pt_l2_lambda_ = 1
    pt_lpips_lambda_ = 1
    LPIPS_value_threshold = 0.06

    with dnnlib.util.open_url(network_pkl) as f:
        original_G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    for p in G.parameters():
        p.requires_grad = True
    optimizer = torch.optim.Adam(G.parameters(), lr=pti_learning_rate)

    num_steps = args.max_pti_steps
    pbar = tqdm(range(num_steps), position=0, leave=True)
    for step in pbar:
        i_sum = i_sum + 1

        synth_images = G.synthesis(w_opt[:1], c[:1], noise_mode='const')['image']
        O = F.interpolate(synth_images, size=(256, 256), mode='area')

        mse_refrec_loss = (((image*mask - O*mask)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum()) * pt_l2_lambda_
        refrec_loss =  percept(O*mask, image*mask,mask=mask) * pt_lpips_lambda_ # ploy

        loss = refrec_loss + mse_refrec_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pti_loss = 0.0
        z_samples = np.random.randn(latent_ball_num_of_samples, original_G.z_dim)
        w_samples = original_G.mapping(z=torch.from_numpy(z_samples).to(device), c=c, truncation_psi=psi, truncation_cutoff=truncation_cutoff)

        territory_indicator_ws = [get_morphed_w_code(w_code.unsqueeze(0), w_opt) for w_code in w_samples]

        for w_code in territory_indicator_ws:

            new_img = G.synthesis(w_code[:1], c[:1], noise_mode='none')['image']
            with torch.no_grad():
                old_img = original_G.synthesis(w_code[:1], c[:1], noise_mode='none')['image']

            if regulizer_l2_lambda_ > 0:
                l2_loss_val = (((old_img - new_img)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum()) * regulizer_l2_lambda_
                pti_loss += l2_loss_val * regulizer_l2_lambda_

            if regulizer_lpips_lambda_ > 0:
                loss_lpips = torch.squeeze(percept(old_img, new_img))
                pti_loss += loss_lpips * regulizer_lpips_lambda_


        pti_loss = pti_loss / len(territory_indicator_ws)

        loss = pti_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        pbar.set_description(( f"i: {i_sum:4d};" f"p: {refrec_loss.item():.2f};" f"m: {mse_refrec_loss.item():.2f};" f"pti: {pti_loss.item():.2f};"))
        # Step
        if refrec_loss <= LPIPS_value_threshold:
            print('refrec_loss <= LPIPS_value_threshold: break')
            break

    pickle.dump(G, open(f'{output_name}_ganweight.pkl', 'wb'))
    synth_images = G.synthesis(w_opt[:1], c[:1], noise_mode='const')['image']
    to_img(synth_images).save(f'{output_name}_ganweight.png')

    print('done pti projection')
    print(f'{output_name}_ganweight.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=str, default="../../checkpoints/ffhqrebalanced512-64.pkl")
    parser.add_argument("--input_dir", type=str, default="../../results/preprocessed/10036_59641/")
    parser.add_argument("--input_filename", type=str, default="10036.png")

    parser.add_argument("--first_inv_type", type=str, default="w")

    ## Steps
    parser.add_argument("--first_inv_steps", type=int, default=500) # pti 450, ed3g 500
    parser.add_argument("--max_pti_steps", type=int, default=500) # pti 350, ed3g 500

    ## losses
    parser.add_argument("--mse_ref", type=float, default=1.0)
    parser.add_argument("--lambda_ref", type=float, default=1.0)
    parser.add_argument("--latent_loss", type=float, default=2)

    args = parser.parse_args()



    eg3d_projection(args)
