# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import argparse
import os
import json
from preprocess import align_img,resize_n_crop_img
from PIL import Image
import numpy as np
import sys
sys.path.append('Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True)
    parser.add_argument('--imgdir', type=str, default='')
    parser.add_argument('--inv', type=int, default=0)
    args = parser.parse_args()

    lm_dir = os.path.join(args.indir, "detections")
    img_files = sorted([x for x in os.listdir(args.indir) if x.lower().endswith(".png") or x.lower().endswith(".jpg")])
    lm_files = sorted([x for x in os.listdir(lm_dir) if x.endswith(".txt")])

    lm3d_std = load_lm3d("Deep3DFaceRecon_pytorch/BFM/")

    if args.inv != 0:
        out_dir = os.path.join(args.indir, "invcrop")
    elif args.imgdir == '':
        out_dir = os.path.join(args.indir, "crop")
    else:
        out_dir = os.path.join(args.imgdir, "recrop")
    # if not os.path.exists(out_dir):
    #     print('out_dir',out_dir)
    #     os.makedirs(out_dir, exist_ok=True)
    print('out_dir',out_dir)
    os.makedirs(out_dir, exist_ok=True)
    for img_file, lm_file in zip(img_files, lm_files):
        print(img_file)
        print(lm_file)
        print('-------------')
        try:
            if args.inv != 0:
                img_path = os.path.join(args.imgdir, img_file)
            elif args.imgdir == '':
                img_path = os.path.join(args.indir, img_file)
            else:
                img_path = os.path.join(args.imgdir, img_file.split('.png')[0]+'_16_16_2Gt5_p1_m0_l2_f2_li5_nf0_ms2_pn1e-05_lp1.0s1000_1000-iopt2_salon00000.png')
                # img_path = os.path.join(args.imgdir, img_file.split('.png')[0]+'_both1000-project_p1_mse1.png')
            lm_path = os.path.join(lm_dir, lm_file)
            im = Image.open(img_path).convert('RGB')
            _,H = im.size
            lm = np.loadtxt(lm_path).astype(np.float32)
            lm = lm.reshape([-1, 2])
            lm[:, -1] = H - 1 - lm[:, -1]

            if args.inv == 0:
                print('normal')
                target_size = 1024.
                rescale_factor = 300
                center_crop_size = 700
                output_size = 512

                _, im_high, _, _, = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)

                left = int(im_high.size[0]/2 - center_crop_size/2)
                upper = int(im_high.size[1]/2 - center_crop_size/2)
                right = left + center_crop_size
                lower = upper + center_crop_size
                im_cropped = im_high.crop((left, upper, right,lower))
                im_cropped = im_cropped.resize((output_size, output_size), resample=Image.LANCZOS)
                # out_path = os.path.join(out_dir, img_file.split(".")[0] + ".png")
                out_path = os.path.join(out_dir, img_file)
                im_cropped.save(out_path)
            else:
                target_size = 1024.
                rescale_factor = 300
                center_crop_size = 700
                output_size = 512

                trans_params, im_high, _, _, = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)
                print('a')
                print(img_file.split(".")[0])
                print('trans_params',trans_params)
                print('im_high',im_high.size)

                # trans_params = np.array([w0, h0, s, t[0], t[1]])
                # out_path = os.path.join(out_dir, img_file.split(".")[0] + ".png")
                # im_cropped.save(out_path)

        except:
            pass
        # break
