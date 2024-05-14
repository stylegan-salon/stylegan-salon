import os
import numpy as np
from PIL import Image
import json

import argparse

def main(args):
    input_face = os.path.splitext(args.input_face)[0]
    input_hair = os.path.splitext(args.input_hair)[0]
    output_filename =  input_face+ '_' + input_hair

    # run eg3d projection
    os.chdir('eg3d-main/eg3d')
    print('eg3d projection')
    command = f"python eg3d_projection.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_filename {args.input_face} --network {args.eg3d_network}"
    print(command)
    os.system(command)
    command = f"python eg3d_projection.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_filename {args.input_hair} --network {args.eg3d_network}"
    print(command)
    os.system(command)

    # run eg3d new pose
    print('eg3d new pose')
    command = f"python eg3d_newpose.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_name {input_face} --target_name {input_hair}"
    print(command)
    os.system(command)
    command = f"python eg3d_newpose.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_name {input_hair} --target_name {input_face}"
    print(command)
    os.system(command)


    # run eg3d new pose with mesh
    print('eg3d new pose with mesh')
    command = f"python eg3d_newpose_from_mesh.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_name {input_face} --target_name {input_hair} --mask_type face"
    print(command)
    os.system(command)
    command = f"python eg3d_newpose_from_mesh.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_name {input_hair} --target_name {input_face} --mask_type hair"
    print(command)
    os.system(command)

    # create guide
    print('create guide')
    os.chdir('../..')
    command = f"python salon_create_guide.py --input_dir {args.output_dir}/preprocessed/{output_filename}/ --input_name {input_face} --target_name {input_hair} --checkpoints {args.checkpoints}"

    print('done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## input/output
    parser.add_argument("--input_face", type=str, default="10036.png")
    parser.add_argument("--input_hair", type=str, default="59641.png")
    parser.add_argument("--output_dir", type=str, default="/home/penguin/stylegan-salon-main/results/")

    parser.add_argument("--eg3d_network", type=str, default="/home/penguin/stylegan-salon-main/checkpoints/ffhqrebalanced512-64.pkl")
    parser.add_argument("--checkpoints", type=str, default="/home/penguin/stylegan-salon-main/checkpoints/")

    args = parser.parse_args()

    main(args)
