import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import argparse
import torch
torch.cuda.empty_cache()
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from networks import lpips
from networks.style_gan_2 import Generator
import numpy as np
from torch import nn
import shutil
from helper import *

from ffhq_dataset.landmarks_detector import LandmarksDetector
from networks.BiSeNet.model import BiSeNet
import random


def salon_main(args):
    device = torch.device('cuda')
    resize = 256
    total_img = 2
    total_opt = 3
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # to (-1,1)
    ])

    face,hair = args.input_name,args.target_name
    merge_name = f'{face}_{hair}'
    main_dir = f'{args.input_dir}/{face}_{hair}/guide_images/face_pov/'
    pair_dir = f'{args.input_dir}/{face}_{hair}/guide_images/hair_pov/'
    output_dir = f'{args.output_dir}/{merge_name}/'
    os.makedirs(output_dir, exist_ok = True)
    print('main_dir',main_dir)
    print('pair_dir',pair_dir)
    print('output_dir',output_dir)

    ## loading gan
    g_ema = Generator(1024, 512, 8)
    g_ema.load_state_dict(torch.load(f'{args.checkpoints}/{args.ckpt}')["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    latent_mean_path = f'{args.checkpoints}/latent_mean_stylegan2.pt'
    latent_std_path = f'{args.checkpoints}/latent_std_stylegan2.pt'
    with torch.no_grad():
        latent_mean = torch.load(latent_mean_path)
        latent_std = torch.load(latent_std_path)

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(1, 1)

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
    for noise in noises:
        noise.requires_grad = True

    # PREDICT KEYPOINT MODEL
    landmarks_model_path = os.path.join('checkpoints','shape_predictor_68_face_landmarks.dat')
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # PREDICT MASK MODEL
    predict_mask_model_path= os.path.join('checkpoints','79999_iter.pth')
    predict_mask_model = BiSeNet(n_classes=19)
    predict_mask_model.cuda()
    predict_mask_model.load_state_dict(torch.load(predict_mask_model_path))
    predict_mask_model.eval()

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", vgg_blocks=["1", "2", "3", "4", "5"], use_gpu=True
    )

    # i_opt: 0 w/ 1 w+/ 2 pti
    for i_opt in range(0,total_opt):
        if i_opt == 0:
            print('i_opt: 0')
            ## image frame from real input
            I_guide,I_guide_s = load_guide(main_dir,pair_dir,merge_name,size=args.size_small,device=device,is_pil=1)
            M_f,M_s,M_h,M_o,M_c,M_hs,M_os = load_mask(main_dir,pair_dir,merge_name,size=args.size_small,device=device,is_pil=0)
            M_bg, M_bg_count =  get_bg_mask(main_dir,merge_name,predict_mask_model,device=device)

            list_latent_head_salons = []
            for j in range(total_img):
                list_latent_head_salons.append(latent_in.detach().clone())
                list_latent_head_salons[j].requires_grad = True

            # merge latent initial
            latent_in_initial = []
            for j in range(total_img):
                latent_in_initial.append(latent_in.detach().clone().unsqueeze(1).repeat(1, g_ema.n_latent, 1))
                latent_in_initial[j].requires_grad = False

            latent_in.requires_grad = False

            latent_loop = list_latent_head_salons[j].detach().clone()
            latent_loop.requires_grad = False
            optimizer = optim.Adam(list_latent_head_salons+ noises, lr=args.lr)

            # parameters
            step = args.step_w
            l_latent_loss = args.latent_loss
            l_lambda_hairrec = args.lambda_hairrec
            l_lambda_ref = args.lambda_ref
            l_lambda_facerec = args.lambda_facerec
            l_mse_small = args.mse_small
            l_lambda_latent_loop = args.latent_loop
            l_lambda_focus = args.focus
            l_lambda_bg = args.lambda_bg

        elif i_opt == 1:
            print('i_opt: 1')
            # update guide from i_opt results
            I_guide,I_guide_s, M_f,M_s,M_h,M_o,M_c,M_hs,M_os = update_guide_and_mask(main_dir,pair_dir,merge_name,size=args.size_small,device=device,predict_mask_model=predict_mask_model,landmarks_detector=landmarks_detector)

            alpha = 0.5
            latent_in_tail = alpha * list_latent_head_salons[0]+ (1-alpha) * list_latent_head_salons[1]

            # only one tail (share all)
            # latent_in_tail = latent_in.detach().clone()[:,args.n_layer:,:]
            latent_in_tail = latent_in_tail.detach().clone().unsqueeze(1).repeat(1, g_ema.n_latent-args.n_layer, 1)
            latent_in_tail.requires_grad = True

            latent_in_initial = []
            print('use w initial')
            for j in range(total_img):
                head_focus = list_latent_head_salons[j].detach().clone().unsqueeze(1).repeat(1, args.n_layer, 1)
                head_focus.requires_grad = False
                latent_in_initial.append(torch.cat([head_focus.detach().clone(), latent_in_tail.detach().clone()], dim=1))
                latent_in_initial[j].requires_grad = False

            for j in range(total_img):
                list_latent_head_salons[j] = list_latent_head_salons[j].detach().clone().unsqueeze(1).repeat(1, args.n_layer, 1)
                list_latent_head_salons[j].requires_grad = True

            # del head_focus
            print('len; salon {}'.format(len(list_latent_head_salons)))

            optimizer = optim.Adam(list_latent_head_salons + [latent_in_tail] + noises, lr=args.lr)

            # parameter
            step = args.step_wp
            l_latent_loss = args.latent_loss_wp
            l_lambda_hairrec = args.lambda_hairrec_wp
            l_lambda_ref = args.lambda_ref_wp
            l_lambda_facerec = args.lambda_facerec_wp
            l_mse_small = args.mse_small_wp
            l_lambda_focus = args.focus_wp
            l_lambda_latent_loop = args.latent_loop_wp
            l_lambda_bg = args.lambda_bg_wp

        elif i_opt == 2:
            print('i_opt: 2')
            # update guide
            I_guide = []
            state1_dir = f'{main_dir}/{merge_name}-iopt1.png'
            mask_pti_face,tI_guide = get_mask_pti(main_dir,merge_name,state1_dir,device=device,predict_mask_model=predict_mask_model,type='face')
            I_guide.append(tI_guide)
            state1_dir = f'{pair_dir}/{merge_name}-iopt1.png'
            mask_pti_hair,tI_guide = get_mask_pti(pair_dir,merge_name,state1_dir,device=device,predict_mask_model=predict_mask_model,type='hair')
            I_guide.append(tI_guide)
            mask_focus = [ mask_pti_face,mask_pti_hair ]

            # disable all latent
            latent_in_tail.requires_grad = False
            for list_latent_head_salon in list_latent_head_salons:
                list_latent_head_salon.requires_grad = False
            for noise in noises:
                noise.requires_grad = False
            # load original_G
            original_G = Generator(1024, 512, 8)
            original_G.load_state_dict(torch.load(f'{args.checkpoints}/{args.ckpt}')["g_ema"], strict=False)
            original_G.eval()
            original_G = original_G.to(device)

            for p in g_ema.parameters():
                p.requires_grad = True

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
            step = args.step_pti
            refrec_loss_keep = [100,100]

            optimizer = torch.optim.Adam(g_ema.parameters(), lr=pti_learning_rate)
            torch.cuda.empty_cache()


        ### for all opt
        # blur I_guide
        I_guide_resize = []
        temp = F.interpolate(I_guide[0], size=(args.size_small, args.size_small), mode='area')
        temp = F.interpolate(temp, size=(256, 256), mode='area')
        I_guide_resize.append(temp)
        temp = F.interpolate(I_guide[1], size=(args.size_small, args.size_small), mode='area')
        temp = F.interpolate(temp, size=(256, 256), mode='area')
        I_guide_resize.append(temp)

        pbar = tqdm(range(step))
        for i in pbar:
            if i_opt < 2:
                t = i / step
                lr = get_lr(t, args.lr)
                optimizer.param_groups[0]["lr"] = lr

            for j in range(total_img):
                if i_opt == 2:
                    if (refrec_loss_keep[0] <= LPIPS_value_threshold) and (refrec_loss_keep[1] <= LPIPS_value_threshold):
                        continue
                else:
                    latent_in_initial[j].requires_grad = True
                if i_opt == 0:
                    # random interpolation
                    alpha = random.random()
                    latent_in_tail = alpha * list_latent_head_salons[0]+ (1-alpha) * list_latent_head_salons[1]
                    latent_n = torch.cat([list_latent_head_salons[j].unsqueeze(1).repeat(1, args.n_layer, 1), latent_in_tail.unsqueeze(1).repeat(1, g_ema.n_latent-args.n_layer, 1)], dim=1)
                else:
                    latent_n = torch.cat([list_latent_head_salons[j], latent_in_tail], dim=1)

                if i_opt < 2:
                    noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
                    latent_n = latent_noise(latent_n, noise_strength.item())
                img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

                batch, channel, height, width = img_gen.shape
                O = F.interpolate(img_gen, size=(resize, resize), mode='area')
                OS = F.interpolate(img_gen, size=(args.size_small, args.size_small), mode='area')
                OB = F.interpolate(OS, size=(256, 256), mode='area')

                # main reconstruction loss
                if i_opt == 2:
                    refrec_loss = percept(O*mask_focus[j], I_guide[j]*mask_focus[j],mask=mask_focus[j])
                    inv_mse_refrec_loss = (((I_guide_s[j] - OS)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum())
                    loss =  refrec_loss * args.pti_p  + inv_mse_refrec_loss * args.pti_s
                    refrec_loss_keep[j] = refrec_loss
                else:
                    #salon
                    if i_opt == 0:
                        facerec_loss = l_lambda_facerec * percept(O*(1-M_s[j]), I_guide[j]*(1-M_s[j]), mask=M_f[j])
                        hairrec_loss = l_lambda_hairrec * percept(O*(1-M_hs[j])*M_o[j], I_guide[j]*(1-M_hs[j])*M_o[j], mask=M_h[j])
                    elif i_opt == 1: # blur
                        if j == 0:
                            facerec_loss = l_lambda_facerec * percept(O*(1-M_s[j]), I_guide[j]*(1-M_s[j]), mask=M_f[j])
                            hairrec_loss = l_lambda_hairrec * percept(OB*(1-M_hs[j])*M_o[j], I_guide_resize[j]*(1-M_hs[j])*M_o[j], mask=M_h[j])
                        elif j == 1:
                            facerec_loss = l_lambda_facerec * percept(OB*(1-M_s[j]), I_guide_resize[j]*(1-M_s[j]), mask=M_f[j])
                            hairrec_loss = l_lambda_hairrec * percept(O*(1-M_hs[j])*M_o[j], I_guide[j]*(1-M_hs[j])*M_o[j], mask=M_h[j])
                    if j == 0:
                        facerec_loss = facerec_loss * l_lambda_focus
                        if M_bg_count > 256*256*0.05:
                            bgrec_loss = l_lambda_bg * percept(O*M_bg, I_guide[j]*M_bg, mask=M_bg)
                            facerec_loss = facerec_loss + bgrec_loss
                    elif j == 1:
                        hairrec_loss = hairrec_loss * l_lambda_focus

                    refrec_loss = l_lambda_ref * percept(O, I_guide[j])
                    mse_small = (((I_guide_s[j] - OS)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum()) * l_mse_small
                    loss = facerec_loss + hairrec_loss + refrec_loss + mse_small

                    # latent loss
                    n_loss = noise_regularize(noises) * args.noise_regularize
                    latent_loss = ((latent_in_initial[j]-latent_n).pow(2).clamp(min=2e-3).mean()) * l_latent_loss
                    loss =  loss + latent_loss +  n_loss
                    latent_loop_loss =  l_lambda_latent_loop * ((latent_loop-list_latent_head_salons[j]).pow(2).clamp(min=2e-3).mean())
                    loss = loss + latent_loop_loss


                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if i_opt < 2:
                    noise_normalize_(noises)
                else:
                    # pti reg
                    pti_loss = 0.0
                    z_samples = torch.randn(latent_ball_num_of_samples, 512, device=device)
                    img_gen, _ = g_ema([z_samples], input_is_latent=False, noise=noises)

                    w_samples = g_ema.mapping([z_samples])[0]
                    img_gen, _ = g_ema([w_samples], input_is_latent=True, noise=noises)
                    territory_indicator_ws = [get_morphed_w_code(w_code.unsqueeze(0), latent_n) for w_code in w_samples]
                    for w_code in territory_indicator_ws:
                        new_img, _ = g_ema([w_code], input_is_latent=True, noise=noises)
                        with torch.no_grad():
                            old_img,_ = original_G([w_code], input_is_latent=True, noise=noises)

                        if regulizer_l2_lambda_ > 0:
                            l2_loss_val = (((old_img - new_img)).pow(2).mean((1, 2, 3)).clamp(min=2e-3).sum()) * regulizer_l2_lambda_
                            pti_loss += l2_loss_val * regulizer_l2_lambda_
                        if regulizer_lpips_lambda_ > 0:
                            loss_lpips = torch.squeeze(percept(old_img, new_img))
                            pti_loss += loss_lpips * regulizer_lpips_lambda_
                    pti_loss = pti_loss / len(territory_indicator_ws)

                    loss = pti_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if i_opt == 2:
                    pbar.set_description((
                        f"p: {refrec_loss.item():.2f}; pinv: {inv_mse_refrec_loss.item():.2f};"
                    ))
                else:
                    #salon
                    pbar.set_description((
                        f"f: {facerec_loss.item():.2f}; h: {hairrec_loss.item():.2f}; la: {latent_loss.item():.2f};"
                    ))
                    latent_loop = list_latent_head_salons[j].detach().clone()
                    latent_loop.requires_grad = False
                    latent_in_initial[j].requires_grad = False

        # just save image
        for j in range(total_img):
            if i_opt == 0:
                latent_n = list_latent_head_salons[j].unsqueeze(1).repeat(1, g_ema.n_latent, 1)
            else:
                latent_n = torch.cat([list_latent_head_salons[j], latent_in_tail], dim=1)
            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)
            img_ar = make_image(img_gen)
            pil_img = Image.fromarray(img_ar[0])
            if j == 0:
                pil_img.save(f'{main_dir}/{merge_name}-iopt{i_opt}.png')
            else:
                pil_img.save(f'{pair_dir}/{merge_name}-iopt{i_opt}.png')

        print('done; i_opt',i_opt)
    print(f"result: {output_dir}/{merge_name}.png")
    shutil.copy2(f'{main_dir}/{merge_name}-iopt2.png', f"{output_dir}/{merge_name}.png")
    print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='10036')
    parser.add_argument('--target_name', type=str, default='59641')

    parser.add_argument("--input_dir", type=str, default="results/preprocessed/")
    parser.add_argument("--output_dir", type=str, default="results/salon_results/")

    parser.add_argument("--checkpoints", type=str, default="/home/penguin/stylegan-salon-main/checkpoints/")
    parser.add_argument("--ckpt", type=str, default="stylegan2-ffhq-config-f.pt")

    parser.add_argument("--lr_rampup", type=float, default=0.05)
    parser.add_argument("--lr_rampdown", type=float, default=0.25)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--noise_ramp", type=float, default=0.75)
    parser.add_argument("--noise_regularize", type=float, default=1e5)

    parser.add_argument("--first_inv_type", type=str, default="w")
    parser.add_argument("--step_w", type=int, default=1000)
    parser.add_argument("--step_wp", type=int, default=500)
    parser.add_argument("--step_pti", type=int, default=500) # 350

    parser.add_argument("--size_small", type=int, default=32)

    # ## more loss
    # first state parameters
    parser.add_argument("--lambda_facerec", type=float, default=2)
    parser.add_argument("--lambda_hairrec", type=float, default=1)
    parser.add_argument("--lambda_ref", type=float, default=0)
    parser.add_argument("--mse_small", type=float, default=2)
    parser.add_argument("--latent_loss", type=float, default=4)

    parser.add_argument("--focus", type=float, default=6)
    parser.add_argument("--latent_loop", type=float, default=3)
    parser.add_argument("--lambda_bg", type=float, default=4)

    # second state parameters
    parser.add_argument("--lambda_facerec_wp", type=float, default=1)
    parser.add_argument("--lambda_hairrec_wp", type=float, default=2)
    parser.add_argument("--lambda_ref_wp", type=float, default=0)
    parser.add_argument("--mse_small_wp", type=float, default=2)
    parser.add_argument("--latent_loss_wp", type=float, default=4)

    parser.add_argument("--focus_wp", type=float, default=4)
    parser.add_argument("--latent_loop_wp", type=float, default=2)
    parser.add_argument("--lambda_bg_wp", type=float, default=4)

    #pti
    parser.add_argument("--pti_p", type=float, default=2)
    parser.add_argument("--pti_s", type=float, default=1)
    parser.add_argument("--n_layer", type=int, default=4)


    args = parser.parse_args()

    salon_main(args)
