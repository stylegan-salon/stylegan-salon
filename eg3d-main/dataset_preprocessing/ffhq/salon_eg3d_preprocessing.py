import os
import numpy as np
from PIL import Image
import json
import scipy
import torch
import sys
sys.path.append('../../../')
# from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from networks.BiSeNet.logger import setup_logger
from networks.BiSeNet.model import BiSeNet
from helper import stylegan_preprocess,eg3d_preprocess
from helper import get_keypoint_biggest,predict_mask,create_mask_ignore
import argparse

def main(args):
    input_face = args.input_face
    input_hair = args.input_hair
    input_dir = args.input_dir
    output_dir = args.output_dir
    stylegan_crop = args.stylegan_crop

    landmarks_model_path = f'{args.checkpoints}/shape_predictor_68_face_landmarks.dat'
    predict_mask_model_path=f'{args.checkpoints}/79999_iter.pth'

    # PREDICT KEYPOINT MODEL
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # PREDICT MASK MODEL
    predict_mask_model = BiSeNet(n_classes=19)
    predict_mask_model.cuda()
    predict_mask_model.load_state_dict(torch.load(predict_mask_model_path))
    predict_mask_model.eval()

    face_image = Image.open(f'{input_dir}/{input_face}')
    hair_image = Image.open(f'{input_dir}/{input_hair}')
    output_filename = os.path.splitext(input_face)[0] + '_' + os.path.splitext(input_hair)[0]

    # stylegan crop
    if stylegan_crop != 0:
        print('stylegan crop')
        face_point = get_keypoint_biggest(landmarks_detector,f'{input_dir}/{input_face}')
        hair_point = get_keypoint_biggest(landmarks_detector,f'{input_dir}/{input_hair}')
        face_image,_,maskcrop_face_image = stylegan_preprocess(face_image,face_point,np.ones_like(np.array(face_image)))
        hair_image,_,maskcrop_hair_image = stylegan_preprocess(hair_image,hair_point,np.ones_like(np.array(hair_image)))
    else:
        maskcrop_face_image = Image.fromarray(np.ones_like(np.array(face_image)).astype(np.uint8))
        maskcrop_hair_image = Image.fromarray(np.ones_like(np.array(hair_image)).astype(np.uint8))

    # save image for eg3d preprocessing
    eg3d_input_dir = f'{output_dir}/preprocessed/{output_filename}'
    os.makedirs(eg3d_input_dir, exist_ok = True)
    face_image.save(f'{eg3d_input_dir}/{input_face}')
    hair_image.save(f'{eg3d_input_dir}/{input_hair}')

    # run eg3d preprocess
    print('eg3d preprocess')
    command = "python preprocess_in_the_wild.py --indir " + eg3d_input_dir
    print(command)
    os.system(command)

    # copy pose to eg3d projection
    print('eg3d camera pose')
    dataset_real_dir = "Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/"+output_filename+"/epoch_20_000000/dataset_real.json"
    json_output_dir = f'{output_dir}/preprocessed/{output_filename}/camera/'
    os.makedirs(json_output_dir, exist_ok = True)

    f = open(dataset_real_dir)
    data = json.load(f)
    print(len(data['labels']))

    ## split camera pose to input filenames
    for i in range(len(data['labels'])):
        list_data = []
        list_data.append(data['labels'][i])
        json_file_name = os.path.splitext(data['labels'][i][0])[0]
        json_output_name = f'{json_output_dir}/{json_file_name}.json'
        print(json_file_name)
        print(json_output_name)

        new_data = {'labels':list_data}
        with open(json_output_name, 'w') as f:
            json.dump(new_data, f)

    print('create mask')
    # crop mask eg3d
    lm_path_face = f'{eg3d_input_dir}/detections/{os.path.splitext(input_face)[0]}.txt'
    lm_path_hair = f'{eg3d_input_dir}/detections/{os.path.splitext(input_hair)[0]}.txt'

    check_inv_img_eg3d_face,_,mask_eg3d_face = eg3d_preprocess(face_image,mask=maskcrop_face_image,lm_path=lm_path_face)
    check_inv_img_eg3d_hair,_,mask_eg3d_hair = eg3d_preprocess(hair_image,mask=maskcrop_hair_image,lm_path=lm_path_hair)

    mask_eg3d_face = mask_eg3d_face.resize((256, 256), Image.NEAREST)
    mask_eg3d_hair = mask_eg3d_hair.resize((256, 256), Image.NEAREST)

    # predict mask
    face_raw_mask = np.array(predict_mask(predict_mask_model,check_inv_img_eg3d_face,256))
    hair_raw_mask = np.array(predict_mask(predict_mask_model,check_inv_img_eg3d_hair,256))
    os.makedirs(f'{eg3d_input_dir}/crop_mask/', exist_ok = True)
    Image.fromarray(face_raw_mask.astype(np.uint8)).save(f'{eg3d_input_dir}/crop_mask/{input_face}')
    Image.fromarray(hair_raw_mask.astype(np.uint8)).save(f'{eg3d_input_dir}/crop_mask/{input_hair}')


    # create mask outside eg3d/stylegan boundary
    os.makedirs(f'{eg3d_input_dir}/crop_mask_ignore/', exist_ok = True)
    mask_ignore = create_mask_ignore(mask_eg3d_face,face_raw_mask)
    Image.fromarray(mask_ignore.astype(np.uint8)).save(f'{eg3d_input_dir}/crop_mask_ignore/{input_face}')
    mask_ignore = create_mask_ignore(mask_eg3d_hair,hair_raw_mask)
    Image.fromarray(mask_ignore.astype(np.uint8)).save(f'{eg3d_input_dir}/crop_mask_ignore/{input_hair}')

    print('done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    ## input/output
    parser.add_argument("--input_dir", type=str, default="/home/penguin/stylegan-salon-main/inputs/")
    parser.add_argument("--input_face", type=str, default="10036.png")
    parser.add_argument("--input_hair", type=str, default="59641.png") # 59641 61774
    parser.add_argument("--output_dir", type=str, default="/home/penguin/stylegan-salon-main/results/")

    parser.add_argument("--checkpoints", type=str, default="/home/penguin/stylegan-salon-main/checkpoints/")
    ## preprocessing
    parser.add_argument("--stylegan_crop", type=int, default=0)

    args = parser.parse_args()

    main(args)
