import os
import argparse
import numpy as np
from PIL import Image
import cv2
# import scipy
import scipy.ndimage
import math
# from scipy import spatial
import torch
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from networks.BiSeNet.logger import setup_logger
from networks.BiSeNet.model import BiSeNet
from torchvision import transforms
from helper import *

def salon_create_guide(args):
    device = torch.device('cuda')
    original_size = 1024
    eg3d_size = 512
    output_size = 256

    landmarks_model_path = f'{args.checkpoints}/shape_predictor_68_face_landmarks.dat'
    predict_mask_model_path=f'{args.checkpoints}/79999_iter.pth'
    # PREDICT KEYPOINT MODEL
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # PREDICT MASK MODEL
    predict_mask_model = BiSeNet(n_classes=19)
    predict_mask_model.cuda()
    predict_mask_model.load_state_dict(torch.load(predict_mask_model_path))
    predict_mask_model.eval()

    lm_dir = f'{args.input_dir}/detections/'
    warp_dir = f'{args.input_dir}/eg3d_warp/'
    temp_inv_dir = f'{args.input_dir}/eg3d_warp/dummy/'
    os.makedirs(temp_inv_dir, exist_ok = True)

    main_output_dir = f'{args.input_dir}/guide_images/face_pov/'
    fixh_output_dir = f'{args.input_dir}/guide_images/hair_pov/'
    os.makedirs(main_output_dir, exist_ok = True)
    os.makedirs(fixh_output_dir, exist_ok = True)

    face = args.input_name
    hair = args.target_name
    fname = f'{face}_{hair}'
    face_name = f'{args.input_dir}/{face}.png'
    hair_name = f'{args.input_dir}/{hair}.png'
    face_img = Image.open(face_name)
    hair_img = Image.open(hair_name)

    lm_path_face = f'{lm_dir}/{face}.txt'
    lm_path_hair = f'{lm_dir}/{hair}.txt'
    lm3d_std_folder = "eg3d-main/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch/BFM"

    face_img_new_dir = f'{warp_dir}/{face}_{hair}_warpimg.png'
    hair_img_new_dir = f'{warp_dir}/{hair}_{face}_warpimg.png'
    imface = Image.open(face_img_new_dir)
    imhair = Image.open(hair_img_new_dir)

    face_mesh_new_dir = f'{warp_dir}/{face}_{hair}_maskface_render.png'
    hair_mesh_new_dir = f'{warp_dir}/{hair}_{face}_maskhair_render.png'
    mface = Image.open(face_mesh_new_dir)
    mhair = Image.open(hair_mesh_new_dir)

    face_raw_mask = np.array(predict_mask(predict_mask_model,face_img,original_size))
    hair_raw_mask = np.array(predict_mask(predict_mask_model,hair_img,original_size))

    ## inv eg3d to stylegan preprocessing (eg3d -> stylegan);
    # eg3d_params
    face_fw_inv_check_dir = f'{temp_inv_dir}/{face}_check.png'
    hair_fw_inv_check_dir = f'{temp_inv_dir}/{hair}_check.png'
    mask_ones = np.ones_like(np.array(face_img))
    mask_ones = Image.fromarray(255*mask_ones.astype(np.uint8))

    im = Image.open(face_name).convert('RGB')
    check_inv_img_eg3d,eg3d_params_face,mask_eg3d_face = eg3d_preprocess(im,mask=mask_ones,lm_path=lm_path_face,lm3d_std_folder=lm3d_std_folder)
    check_inv_img,_ = inv_eg3d_preprocess(check_inv_img_eg3d,eg3d_params_face,mask=None)
    check_inv_img.save(face_fw_inv_check_dir)
    # print('face_point')
    face_point = get_keypoint_biggest(landmarks_detector,face_fw_inv_check_dir)
    print('face_plain')
    _, face_plain = create_face_contour(face_point,face_raw_mask) # create face mask from keypoint

    im = Image.open(hair_name).convert('RGB')
    check_inv_img_eg3d,eg3d_params_hair,mask_eg3d_hair = eg3d_preprocess(im,mask=mask_ones,lm_path=lm_path_hair,lm3d_std_folder=lm3d_std_folder)
    check_inv_img,_ = inv_eg3d_preprocess(check_inv_img_eg3d,eg3d_params_hair,mask=None)
    check_inv_img.save(hair_fw_inv_check_dir)
    # print('hair_point')
    hair_point = get_keypoint_biggest(landmarks_detector,hair_fw_inv_check_dir)
    print('hair_plain')
    _, hair_plain = create_face_contour(hair_point,hair_raw_mask)

    # inv eg3d
    warp_face_name_inv = f'{temp_inv_dir}/{face}_inveg3d.png'
    warp_hair_name_inv = f'{temp_inv_dir}/{hair}_inveg3d.png'

    ## face warp img
    img_inv_eg3d,mask_inv_eg3d = inv_eg3d_preprocess(imface,eg3d_params_face,mask=mask_eg3d_face)
    face_img_new = img_inv_eg3d
    mask_inv_eg3d = np.array(mask_inv_eg3d)
    mask_inv_eg3d = mask_inv_eg3d[:,:,0]
    mask_inv_eg3d[mask_inv_eg3d!=0] = 1
    face_img_new_mask_inside = mask_inv_eg3d
    mask = np.array(predict_mask(predict_mask_model,face_img_new,original_size))
    mask = mask * face_img_new_mask_inside
    face_img_new_raw_mask = mask
    face_img_new.save(warp_face_name_inv)

    ## hair warp img
    img_inv_eg3d,mask_inv_eg3d = inv_eg3d_preprocess(imhair,eg3d_params_hair,mask=mask_eg3d_hair)
    hair_img_new = img_inv_eg3d
    mask_inv_eg3d = np.array(mask_inv_eg3d)
    mask_inv_eg3d = mask_inv_eg3d[:,:,0]
    mask_inv_eg3d[mask_inv_eg3d!=0] = 1
    hair_img_new_mask_inside = mask_inv_eg3d
    mask = np.array(predict_mask(predict_mask_model,hair_img_new,original_size))
    mask = mask * hair_img_new_mask_inside
    hair_img_new_raw_mask = mask
    hair_img_new.save(warp_hair_name_inv)

    ## create_face_contour hair
    hair_img_point = get_keypoint_biggest(landmarks_detector,warp_hair_name_inv)
    print('hair_img_new_plain')
    if hair_img_point is None:
        print('cannot predict from hair_img_point')
        hair_img_new_plain = get_mask(hair_img_new_raw_mask,'face')
    else:
        _, hair_img_new_plain = create_face_contour(hair_img_point,hair_img_new_raw_mask)

    # warp raw (face/hair) pixel by mesh
    face_mesh_new_mask_dir = f'{warp_dir}/{face}_{hair}_maskface_mask.png'
    hair_mesh_new_mask_dir = f'{warp_dir}/{hair}_{face}_maskhair_mask.png'
    warp_face_name_inv = f'{temp_inv_dir}/{face}_invmesh.png'
    warp_hair_name_inv = f'{temp_inv_dir}/{hair}_invmesh.png'

    ## warp raw face
    mask = Image.open(face_mesh_new_mask_dir).convert('RGB')
    img_inv_eg3d,mask_inv_eg3d = inv_eg3d_preprocess(mface,eg3d_params_face,mask=mask)
    face_mesh = img_inv_eg3d
    face_mesh.save(warp_face_name_inv)
    mask = np.array(mask_inv_eg3d)[:,:,0]
    mask[mask!=0] = 1
    maske = scipy.ndimage.binary_erosion(mask, iterations=20).astype(np.uint8)
    maske = scipy.ndimage.binary_dilation(mask, iterations=20).astype(np.uint8)
    face_mesh_maskmesh = mask * maske

    ## warp raw hair
    mask = Image.open(hair_mesh_new_mask_dir).convert('RGB')
    img_inv_eg3d,mask_inv_eg3d = inv_eg3d_preprocess(mhair,eg3d_params_hair,mask=mask)
    hair_mesh = img_inv_eg3d
    hair_mesh.save(warp_hair_name_inv)
    mask = np.array(mask_inv_eg3d)[:,:,0]
    mask[mask!=0] = 1
    maske = scipy.ndimage.binary_erosion(mask, iterations=20).astype(np.uint8)
    maske = scipy.ndimage.binary_dilation(mask, iterations=20).astype(np.uint8)
    hair_mesh_maskmesh = mask * maske

    ### define main
    main_face_img = np.array(face_img)
    main_face_raw_mask = face_raw_mask
    main_face_plain = face_plain
    main_face_mask_inside = np.ones_like(np.array(face_plain))

    main_hair_img = hair_img_new
    main_hair_raw_mask = hair_img_new_raw_mask
    main_hair_plain = hair_img_new_plain
    main_hair_mask_inside = hair_img_new_mask_inside

    # put from mesh into from image
    # main -> put hair mesh to hair img
    main_hair_mesh = hair_mesh
    main_hair_mask_mesh = hair_mesh_maskmesh
    main_hair_mask_mesh = main_hair_mask_mesh * get_mask(main_hair_raw_mask,'hair')
    main_hair_img = main_hair_img * (1-main_hair_mask_mesh[:,:,np.newaxis]) + main_hair_mesh * (main_hair_mask_mesh[:,:,np.newaxis])

    main_face_point = face_point
    Image.fromarray(main_hair_img.astype(np.uint8)).save(f'{main_output_dir}/{fname}_hair.png')
    print('main_hair_point')
    main_hair_point = get_keypoint_biggest(landmarks_detector,f'{main_output_dir}/{fname}_hair.png')
    if main_hair_point is None:
        print('cannot predict from warp hair, use warp img')
        main_hair_point = hair_img_point

    # new
    ### define fixh
    print('new fixh')

    fixh_face_img = face_img_new
    fixh_face_raw_mask = face_img_new_raw_mask
    fixh_face_mask_inside = face_img_new_mask_inside

    fixh_hair_img = hair_img
    fixh_hair_raw_mask = hair_raw_mask
    fixh_hair_plain = hair_plain
    fixh_hair_mask_inside = np.ones_like(np.array(hair_plain))

    # put from mesh into from image
    # fixh -> put face mesh to face img
    fixh_face_mesh = face_mesh
    fixh_face_mask_mesh = face_mesh_maskmesh
    fixh_face_mask_mesh = fixh_face_mask_mesh * get_mask(fixh_face_raw_mask,'face')
    fixh_face_img = fixh_face_img * (1-fixh_face_mask_mesh[:,:,np.newaxis]) + fixh_face_mesh * (fixh_face_mask_mesh[:,:,np.newaxis])
    Image.fromarray(fixh_face_img.astype(np.uint8)).save(f'{fixh_output_dir}/{fname}_face.png')

    fixh_face_point = get_keypoint_biggest(landmarks_detector,f'{fixh_output_dir}/{fname}_face.png')
    if fixh_face_point is None:
        print('cannot predict from pair_face_img')
        print('face_img_point')
        face_img_point = get_keypoint_biggest(landmarks_detector,f'{temp_inv_dir}/{face}_inveg3d.png')
        fixh_face_point = face_img_point
    print('fixh_face_plain')

    if fixh_face_point is None: # bad keypoint prediction
        fixh_face_plain = get_mask(face_img_new_raw_mask,'face')
    else:
        _, fixh_face_plain = create_face_contour(fixh_face_point,np.array(fixh_face_raw_mask))

    fixh_hair_point = hair_point

    ### create shift
    # main
    # a,b,ratio = 0,0,1
    if (main_face_point is None) or (main_hair_point is None):
        print('bad keypoint')
        a,b,ratio = 0,0,1
    else:
        a,b,ratio = cal_shift_parms(main_face_point,main_hair_point,ratio=None)
    a,b = int(a),int(b)
    dim = (int(ratio*original_size),int(ratio*original_size))
    print('shift_parms:',a,b,ratio)

    ## resize,shift hair
    # shift warp hair (put to raw face)
    main_hair_img = resize_shift_img_a_b(main_hair_img,a,b,dim,img_type='img',)
    main_hair_raw_mask = resize_shift_img_a_b(main_hair_raw_mask,a,b,dim,img_type='mask')
    main_hair_plain = resize_shift_img_a_b(main_hair_plain,a,b,dim,img_type='mask')
    main_hair_mask_inside = resize_shift_img_a_b(main_hair_mask_inside,a,b,dim,img_type='mask')
    # main_hair_mesh = resize_shift_img_a_b(main_hair_mesh,a,b,dim,img_type='img')
    main_hair_mask_mesh = resize_shift_img_a_b(main_hair_mask_mesh,a,b,dim,img_type='mask')

    # resize all to output size
    main_face_img = np.array(Image.fromarray(main_face_img.astype(np.uint8)).resize((output_size,output_size)))
    main_face_raw_mask = np.array(Image.fromarray(main_face_raw_mask.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_face_plain = np.array(Image.fromarray(main_face_plain.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_face_mask_inside = np.array(Image.fromarray(main_face_mask_inside.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_face_point = main_face_point.copy() // 4

    main_hair_img = np.array(Image.fromarray(main_hair_img.astype(np.uint8)).resize((output_size,output_size)))
    main_hair_raw_mask = np.array(Image.fromarray(main_hair_raw_mask.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_hair_plain = np.array(Image.fromarray(main_hair_plain.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_hair_mask_inside = np.array(Image.fromarray(main_hair_mask_inside.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))
    main_hair_mask_mesh = np.array(Image.fromarray(main_hair_mask_mesh.astype(np.uint8)).resize((output_size,output_size),Image.NEAREST))

    Image.fromarray(main_face_img.astype(np.uint8)).save(f'{main_output_dir}/{fname}_face.png')
    Image.fromarray(main_hair_mask_mesh.astype(np.uint8)*255).save(f'{main_output_dir}/{fname}_hairmesh_mask.png')

    # new
    ### define fixh
    print('new fixh')


    # cal_resize_shift_parms
    # inv from other: warp face to fit hair
    # a,b,ratio = 0,0,1
    if (fixh_hair_point is None) or (fixh_face_point is None):
        print('bad keypoint')
        a,b,ratio = 0,0,1
    else:
        a,b,ratio = cal_shift_parms(fixh_hair_point,fixh_face_point,ratio=None)
    a,b = int(a),int(b)
    dim = (int(ratio*original_size),int(ratio*original_size))
    print('shift_parms:',a,b,ratio)

    # for raw hair -> extend image boundary

    fixh_hair_mask_inside = Image.fromarray(fixh_hair_mask_inside.astype(np.uint8))
    # fixh_hair_raw_mask = resize_shift_img_a_b(fixh_hair_raw_mask,a,b,dim,img_type='mask')
    fixh_hair_raw_mask = Image.fromarray(fixh_hair_raw_mask.astype(np.uint8))
    # fixh_hair_plain = resize_shift_img_a_b(fixh_hair_plain,a,b,dim,img_type='mask')
    fixh_hair_plain = Image.fromarray(fixh_hair_plain.astype(np.uint8))

    fixh_face_img = resize_shift_img_a_b(fixh_face_img,a,b,dim,img_type='img')
    fixh_face_img = Image.fromarray(fixh_face_img.astype(np.uint8))
    fixh_face_mask_inside = resize_shift_img_a_b(fixh_face_mask_inside,a,b,dim,img_type='mask')
    fixh_face_mask_inside = Image.fromarray(fixh_face_mask_inside.astype(np.uint8))
    fixh_face_raw_mask = resize_shift_img_a_b(fixh_face_raw_mask,a,b,dim,img_type='mask')
    fixh_face_raw_mask = Image.fromarray(fixh_face_raw_mask.astype(np.uint8))
    fixh_face_mask_mesh = resize_shift_img_a_b(fixh_face_mask_mesh,a,b,dim,img_type='mask')
    Image.fromarray(fixh_face_mask_mesh.astype(np.uint8)*255).resize((output_size,output_size),Image.NEAREST).save(f'{fixh_output_dir}/{fname}_facemesh_mask.png')

    fixh_face_img.save(f'{fixh_output_dir}/{fname}_face_extend.png')
    fixh_face_point = get_keypoint_biggest(landmarks_detector,f'{fixh_output_dir}/{fname}_face_extend.png')
    if fixh_face_point is None:
        print('face_img_new',type(face_img_new))
        temp = resize_shift_img_a_b(face_img_new,a,b,dim,img_type='img')
        print('temp',type(temp))
        temp = Image.fromarray(temp.astype(np.uint8))
        temp.save(f'{fixh_output_dir}/{fname}_face_extend.png')
        print(f'{fixh_output_dir}/{fname}_face_extend.png')
        fixh_face_point = get_keypoint_biggest(landmarks_detector,f'{fixh_output_dir}/{fname}_face_extend.png')

    # resize all to output size
    if fixh_face_point is not None:
        fixh_face_point = fixh_face_point.copy() // 4
    fixh_face_img = np.array((fixh_face_img).resize((output_size,output_size)))
    fixh_face_raw_mask = np.array((fixh_face_raw_mask).resize((output_size,output_size),Image.NEAREST))
    fixh_face_mask_inside = np.array((fixh_face_mask_inside).resize((output_size,output_size),Image.NEAREST))
    fixh_face_plain = np.array((Image.fromarray(fixh_face_plain.astype(np.uint8))).resize((output_size,output_size),Image.NEAREST))

    fixh_hair_img = np.array((fixh_hair_img).resize((output_size,output_size)))
    fixh_hair_raw_mask = np.array((fixh_hair_raw_mask).resize((output_size,output_size),Image.NEAREST))
    fixh_hair_mask_inside = np.array((fixh_hair_mask_inside).resize((output_size,output_size),Image.NEAREST))
    fixh_hair_plain = np.array((fixh_hair_plain).resize((output_size,output_size),Image.NEAREST))

    ## change face background
    # use face background on face pose in all face image
    bg_color = np.mean(main_face_img[np.where(main_face_raw_mask == 0)],axis=0)
    main_color = np.zeros((output_size,output_size,3),np.uint8)
    main_color[:,:] = bg_color

    mask_background_ori = get_mask(main_face_raw_mask,get_type='background')
    mask_background = get_mask(fixh_face_raw_mask,get_type='background')
    mask = mask_background_ori * mask_background
    fixh_face_img = fixh_face_img * (1-mask_background[:,:,np.newaxis]) + main_color * mask_background[:,:,np.newaxis]
    fixh_face_img = fixh_face_img * (1-mask[:,:,np.newaxis]) + main_face_img * mask[:,:,np.newaxis]
    Image.fromarray(fixh_face_img.astype(np.uint8)).save(f'{fixh_output_dir}/{fname}_face.png')

    # create params
    main_hair_maskhair = get_mask(main_hair_raw_mask,'hair')
    main_face_maskface = get_mask(main_face_raw_mask,get_type='faceneck')
    main_face_maskhair = get_mask(main_face_raw_mask,get_type='hair')

    fixh_hair_maskhair = get_mask(fixh_hair_raw_mask,'hair')
    fixh_face_maskface = get_mask(fixh_face_raw_mask,get_type='faceneck')
    fixh_face_maskhair = get_mask(fixh_face_raw_mask,get_type='hair')

    ## create mask outside
    main_mask_outside = get_mask_outside(main_hair_mask_inside,main_hair_maskhair)
    main_mask_outside[get_mask(main_hair_raw_mask,get_type='hat')==1] = 1

    fixh_mask_outside = get_mask_outside(fixh_hair_mask_inside,fixh_hair_maskhair)
    fixh_mask_outside[get_mask(fixh_hair_raw_mask,get_type='hat')==1] = 1

    print('start save')
    ## save intial information
    Image.fromarray(main_mask_outside.astype(np.uint8)*255).save(f'{main_output_dir}/{fname}_mask_outside.png')
    Image.fromarray(main_face_img.astype(np.uint8)).save(f'{main_output_dir}/{fname}_face.png')
    Image.fromarray(main_hair_img.astype(np.uint8)).save(f'{main_output_dir}/{fname}_shift_hair.png')
    Image.fromarray(main_hair_maskhair.astype(np.uint8)*255).save(f'{main_output_dir}/{fname}_shift_hair_mask.png')

    Image.fromarray(fixh_mask_outside.astype(np.uint8)*255).save(f'{fixh_output_dir}/{fname}_mask_outside.png')
    Image.fromarray(fixh_face_img.astype(np.uint8)).save(f'{fixh_output_dir}/{fname}_face.png')
    Image.fromarray(fixh_hair_img.astype(np.uint8)).save(f'{fixh_output_dir}/{fname}_shift_hair.png')
    Image.fromarray(fixh_hair_maskhair.astype(np.uint8)*255).save(f'{fixh_output_dir}/{fname}_shift_hair_mask.png')

    ## create mask outside for face
    main_face_mask_outside = 1-main_face_mask_inside
    Image.fromarray(main_face_mask_outside.astype(np.uint8)*255).save(f'{main_output_dir}/{fname}_face_mask_outside.png')
    fixh_face_mask_outside = 1-fixh_face_mask_inside
    Image.fromarray(fixh_face_mask_outside.astype(np.uint8)*255).save(f'{fixh_output_dir}/{fname}_face_mask_outside.png')

    ## create mask and color palette
    main_color_guide = get_mask(main_hair_raw_mask,get_type='color_guide')
    main_color = np.zeros((output_size,output_size,3),np.uint8)
    main_skin_color = np.zeros((output_size,output_size),np.uint8)

    fixh_color_guide = get_mask(fixh_hair_raw_mask,get_type='color_guide')
    fixh_color = np.zeros((output_size,output_size,3),np.uint8)
    fixh_skin_color = np.zeros((output_size,output_size),np.uint8)

    ## I_guide stuff
    ## easy fill
    # make color
    bg_color = np.mean(main_face_img[np.where(main_face_raw_mask == 0)],axis=0)
    face_color = np.mean(main_face_img[np.where(main_face_raw_mask == 10)],axis=0)

    # fill bg
    main_color[:,:] = bg_color
    fixh_color[:,:] = bg_color

    # fill hair
    main_color = main_color * (1-main_hair_maskhair[:,:,np.newaxis]) + main_hair_img * main_hair_maskhair[:,:,np.newaxis]
    fixh_color = fixh_color * (1-fixh_hair_maskhair[:,:,np.newaxis]) + fixh_hair_img * fixh_hair_maskhair[:,:,np.newaxis] # fill hair

    # fill clostest color: hair and ear
    main_fill = main_color_guide.copy()
    main_fill[main_fill==7] = 1
    main_fill[main_fill!=1] = 0
    main_color,main_skin_color = fill_hair(main_color,main_skin_color,main_color_guide,main_face_point,main_hair_maskhair,main_fill,main_face_raw_mask,face_color)

    fixh_fill = fixh_color_guide.copy()
    fixh_fill[fixh_fill==7] = 1
    fixh_fill[fixh_fill!=1] = 0
    if fixh_face_point is not None:
        fixh_color,fixh_skin_color = fill_hair(fixh_color,fixh_skin_color,fixh_color_guide,fixh_face_point,fixh_hair_maskhair,fixh_fill,fixh_face_raw_mask,face_color)

    ## fill face color
    main_fill_face = main_face_plain.copy()
    fixh_fill_face = fixh_face_plain.copy()
    # fill top
    main_fill_face[:int(max(main_face_point[0][1],main_face_point[16][1])),int(main_face_point[0][0]):int(main_face_point[16][0])] = 1
    main_fill_face = main_fill_face * main_fill
    main_color[np.where(main_fill_face!=0)] = face_color
    main_skin_color[np.where(main_fill_face!=0)] = 1

    if fixh_face_point is not None:
        fixh_fill_face[:int(max(fixh_face_point[0][1],fixh_face_point[16][1])),int(fixh_face_point[0][0]):int(fixh_face_point[16][0])] = 1
    fixh_fill_face = fixh_fill_face * fixh_fill
    fixh_color[np.where(fixh_fill_face!=0)] = face_color
    fixh_skin_color[np.where(fixh_fill_face!=0)] = 1

    ## create more mask
    ## main
    # erode face higher than eyebrows
    main_face_maskface_new = main_face_maskface*(1-main_hair_maskhair)
    main_face_maskface_new_erode = scipy.ndimage.binary_erosion(main_face_maskface_new, iterations=5).astype(np.uint8)
    main_face_maskface_new_top_border = main_face_maskface_new-main_face_maskface_new_erode
    main_face_maskface_new_top_border[int(min(main_face_point[19][1],main_face_point[24][1])-5):,:] = 0 # select only higher than eyebrows
    # predict neck from keypoint
    main_plain_neck = np.zeros_like(main_color_guide)
    main_plain_neck[int(min((main_face_point[3][1],main_face_point[13][1]))):,int(main_face_point[3][0]):int(main_face_point[13][0])] = 1
    # main_plain_neck[int(min((main_face_point[4][1],main_face_point[12][1]))):,int(main_face_point[4][0]):int(main_face_point[12][0])] = 1
    main_plain_neck = main_plain_neck * (1-main_face_maskface) * (1-main_face_plain)
    # get new hairmask (remove part that may be face)
    main_mask_faceonly = get_mask(main_face_raw_mask,get_type='face')
    main_hair_mask_rm = get_new_hairmask(main_face_plain,main_mask_faceonly,main_hair_plain,main_hair_maskhair,main_mask_outside)

    ## fixh
    # erode face higher than eyebrows
    if fixh_face_point is not None:
        fixh_face_maskface_new = fixh_face_maskface*(1-main_hair_maskhair)
        fixh_face_maskface_new_erode = scipy.ndimage.binary_erosion(fixh_face_maskface_new, iterations=5).astype(np.uint8)
        fixh_face_maskface_new_top_border = fixh_face_maskface_new-fixh_face_maskface_new_erode
        fixh_face_maskface_new_top_border[int(min(fixh_face_point[19][1],fixh_face_point[24][1])-5):,:] = 0 # select only higher than eyebrows
    else:
        fixh_face_maskface_new_top_border = np.zeros_like(fixh_face_maskface)
    # predict neck from keypoint
    fixh_plain_neck = np.zeros_like(fixh_color_guide)
    if fixh_face_point is not None:
        fixh_plain_neck[int(min((fixh_face_point[3][1],fixh_face_point[13][1]))):,int(fixh_face_point[3][0]):int(fixh_face_point[13][0])] = 1
    # fixh_plain_neck[int(min((fixh_face_point[4][1],fixh_face_point[12][1]))):,int(fixh_face_point[4][0]):int(fixh_face_point[12][0])] = 1
    fixh_plain_neck = fixh_plain_neck * (1-fixh_face_maskface) * (1-fixh_face_plain)
    # get new hairmask (remove part that may be face)
    fixh_mask_faceonly = get_mask(fixh_face_raw_mask,get_type='face')
    fixh_hair_mask_rm = get_new_hairmask(fixh_face_plain,fixh_mask_faceonly,fixh_hair_plain,fixh_hair_maskhair,fixh_mask_outside)

    ## main
    ## merge I_guide
    # bg_face may be hair (long hair)
    main_bg_face = get_mask(main_hair_raw_mask,get_type='faceearneck') * (1-get_mask(main_face_raw_mask,get_type='faceearneck'))
    main_face_rebg = main_face_img * (1-main_bg_face[:,:,np.newaxis]) + main_color * (main_bg_face[:,:,np.newaxis])
    main_face_nohair = main_face_rebg * (1-main_face_maskhair[:,:,np.newaxis]) + main_color*main_face_maskhair[:,:,np.newaxis]
    # merge to guide
    main_out_i_guide = main_face_nohair * (1-main_hair_mask_rm[:,:,np.newaxis])+ main_hair_img * main_hair_mask_rm[:,:,np.newaxis]
    main_I_guide = Image.fromarray(main_out_i_guide)

    ## fixh
    # merge I_guide
    # bg_face may be hair (long hair)
    fixh_bg_face = get_mask(fixh_hair_raw_mask,get_type='faceearneck') * (1-get_mask(fixh_face_raw_mask,get_type='faceearneck'))
    fixh_face_rebg = fixh_face_img * (1-fixh_bg_face[:,:,np.newaxis]) + fixh_color * (fixh_bg_face[:,:,np.newaxis])
    fixh_face_nohair = fixh_face_rebg * (1-fixh_face_maskhair[:,:,np.newaxis]) + fixh_color*fixh_face_maskhair[:,:,np.newaxis]
    # merge to guide
    fixh_out_i_guide = fixh_face_nohair * (1-fixh_hair_mask_rm[:,:,np.newaxis])+ fixh_hair_img * fixh_hair_mask_rm[:,:,np.newaxis]
    fixh_I_guide = Image.fromarray(fixh_out_i_guide)

    ## create mask and save
    ## main
    ## M_f and M_fs
    M_f = (main_face_maskface)*(1-main_hair_maskhair)*(1-main_face_maskface_new_top_border)*(1-main_mask_outside)
    M_f = M_f * (1-get_mask(main_hair_raw_mask,get_type='hat'))
    M_f = Image.fromarray(M_f*255)
    main_mask_neck = get_mask(main_face_raw_mask,get_type='neck')
    M_fs = Image.fromarray(main_skin_color*(1-main_mask_neck)*(1-main_face_maskface*(1-main_face_maskface_new_top_border))*(1-main_hair_mask_rm)*(1-main_mask_outside)*255)

    ## M_h ans M_hs
    M_hs = get_mask(main_hair_raw_mask,get_type='faceearneck')
    M_hs = M_hs * (1-(np.array(main_hair_mask_rm)))
    M_hs[main_mask_outside>0] = 1
    main_hair_mask_rm_e = main_hair_mask_rm.copy()
    main_hair_mask_rm_e[M_hs>0] = 1

    temp_e = np.ones((266,266),dtype=main_hair_mask_rm_e.dtype)
    temp_e[5:256+5,5:256+5] = main_hair_mask_rm_e
    temp_e = scipy.ndimage.binary_erosion(temp_e, iterations=5).astype(np.uint8) # here
    main_hair_mask_rm_e = temp_e[5:256+5,5:256+5]
    M_h = Image.fromarray((main_hair_mask_rm_e*(1-(M_hs)))*255)
    M_hs = Image.fromarray(M_hs*255)
    M_o = Image.fromarray((1-main_mask_outside)*(1-main_plain_neck)*255)
    mask_face_real_bg = (1-main_face_maskface) * (1-main_hair_maskhair) * (1-main_hair_mask_rm) * (np.array(M_o)//255) * (1-np.array(M_fs)//255)
    temp_e = np.ones((266,266),dtype=mask_face_real_bg.dtype)
    temp_e[5:256+5,5:256+5] = mask_face_real_bg
    temp_e = scipy.ndimage.binary_erosion(temp_e, iterations=5).astype(np.uint8) # here
    mask_face_real_bg = temp_e[5:256+5,5:256+5]
    M_c = (np.array(M_f)//255) + (np.array(M_h)//255) + mask_face_real_bg
    # M_c = M_c * (np.array(M_o)//255)
    M_c = M_c * (1-get_mask(main_hair_raw_mask,get_type='hat'))
    M_c = Image.fromarray(M_c*255)


    main_I_guide.save(f'{main_output_dir}/{fname}_Iguide1.png')
    M_f.save(f'{main_output_dir}/{fname}_Mf.png')
    M_fs.save(f'{main_output_dir}/{fname}_Ms.png')
    M_h.save(f'{main_output_dir}/{fname}_Mh.png')
    M_o.save(f'{main_output_dir}/{fname}_Mo.png')
    M_c.save(f'{main_output_dir}/{fname}_Mc.png')
    M_hs.save(f'{main_output_dir}/{fname}_Mhs.png')

    ## fixh
    ## M_f and M_fs
    M_f = (fixh_face_maskface)*(1-fixh_hair_maskhair)*(1-fixh_face_maskface_new_top_border)*(1-fixh_mask_outside)
    M_f = M_f * (1-get_mask(fixh_hair_raw_mask,get_type='hat'))
    M_f = Image.fromarray(M_f*255)
    fixh_mask_neck = get_mask(fixh_face_raw_mask,get_type='neck')
    M_fs = Image.fromarray(fixh_skin_color*(1-fixh_mask_neck)*(1-fixh_face_maskface*(1-fixh_face_maskface_new_top_border))*(1-fixh_hair_mask_rm)*(1-fixh_mask_outside)*255)

    ## M_h ans M_hs
    M_hs = get_mask(fixh_hair_raw_mask,get_type='faceearneck')
    M_hs = M_hs * (1-(np.array(fixh_hair_mask_rm)))
    M_hs[fixh_mask_outside>0] = 1
    fixh_hair_mask_rm_e = fixh_hair_mask_rm.copy()
    fixh_hair_mask_rm_e[M_hs>0] = 1

    temp_e = np.ones((266,266),dtype=fixh_hair_mask_rm_e.dtype)
    temp_e[5:256+5,5:256+5] = fixh_hair_mask_rm_e
    temp_e = scipy.ndimage.binary_erosion(temp_e, iterations=5).astype(np.uint8) # here
    fixh_hair_mask_rm_e = temp_e[5:256+5,5:256+5]
    M_h = Image.fromarray((fixh_hair_mask_rm_e*(1-(M_hs)))*255)
    M_hs = Image.fromarray(M_hs*255)
    M_o = Image.fromarray((1-fixh_mask_outside)*(1-fixh_plain_neck)*255)
    mask_face_real_bg = (1-fixh_face_maskface) * (1-fixh_hair_maskhair) * (1-fixh_hair_mask_rm) * (np.array(M_o)//255) * (1-np.array(M_fs)//255)
    temp_e = np.ones((266,266),dtype=mask_face_real_bg.dtype)
    temp_e[5:256+5,5:256+5] = mask_face_real_bg
    temp_e = scipy.ndimage.binary_erosion(temp_e, iterations=5).astype(np.uint8) # here
    mask_face_real_bg = temp_e[5:256+5,5:256+5]
    M_c = (np.array(M_f)//255) + (np.array(M_h)//255) + mask_face_real_bg
    # M_c = M_c * (np.array(M_o)//255)
    M_c = M_c * (1-get_mask(fixh_hair_raw_mask,get_type='hat'))
    M_c = Image.fromarray(M_c*255)


    fixh_I_guide.save(f'{fixh_output_dir}/{fname}_Iguide1.png')
    M_f.save(f'{fixh_output_dir}/{fname}_Mf.png')
    M_fs.save(f'{fixh_output_dir}/{fname}_Ms.png')
    M_h.save(f'{fixh_output_dir}/{fname}_Mh.png')
    M_o.save(f'{fixh_output_dir}/{fname}_Mo.png')
    M_c.save(f'{fixh_output_dir}/{fname}_Mc.png')
    M_hs.save(f'{fixh_output_dir}/{fname}_Mhs.png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='10036')
    parser.add_argument('--target_name', type=str, default='59641')

    parser.add_argument("--input_dir", type=str, default="results/preprocessed/10036_59641/")

    parser.add_argument("--checkpoints", type=str, default="checkpoints")

    args = parser.parse_args()

    salon_create_guide(args)
