import os
import math
from PIL import Image
import numpy as np
import cv2
import scipy
import torch
from torch.nn import functional as F
import torchvision.transforms as transforms
from ffhq_dataset.face_alignment import image_align
from ffhq_dataset.landmarks_detector import LandmarksDetector
from networks.BiSeNet.logger import setup_logger
from networks.BiSeNet.model import BiSeNet

def get_morphed_w_code(new_w_code, fixed_w,regulizer_alpha = 30):
    interpolation_direction = new_w_code - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = regulizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move
    # self.morphing_regulizer_alpha * fixed_w + (1 - self.morphing_regulizer_alpha) * new_w_code
    return result_w

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2
    return loss

def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength
    return latent + noise

def get_keypoint_biggest(landmarks_detector,image_filenames):
    # landmarks_detector = LandmarksDetector(landmarks_model_path)
    # image_filenames = img_file[i]
    keep = None
    max_distant = 0
    n_key = 0
    for n_key, preds in enumerate(landmarks_detector.get_landmarks(image_filenames), start=1):
        preds = np.array([list(ele) for ele in preds]).astype(np.float32)
        distant = (preds[16] - preds[0])[0]
        if max_distant < distant:
            max_distant = distant
            keep = preds
        pass
    if n_key > 1:
        print(max_distant)
    return keep

def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

def get_mask(ref_mask,get_type='face'):
    # atts = [0:bg, 1'skin', 2'l_brow', 3'r_brow', 4'l_eye', 5'r_eye', 6'eye_g',
    #        7'l_ear',8 'r_ear', 9'ear_r',10 'nose',  11'mouth',
    #       12'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17'hair', 18 'hat']
    ref_mask = ref_mask.copy()
    if len(np.unique(ref_mask)) > 2:
        if get_type == 'hair':
            ref_mask[ref_mask!=17] = 0
            ref_mask[ref_mask==17] = 1
        elif get_type == 'face':
            ref_mask[(ref_mask==0) | (ref_mask==17) | (ref_mask==18) | (ref_mask==16) | (ref_mask==15)| (ref_mask==14)| (ref_mask==9)] = 0
            ref_mask[(ref_mask==7) | (ref_mask==8)] = 0
            ref_mask[ref_mask!=0] = 1
        elif get_type == 'faceneck':
            ref_mask[(ref_mask==0) | (ref_mask==17) | (ref_mask==18) | (ref_mask==16) | (ref_mask==15)| (ref_mask==9)] = 0
            ref_mask[(ref_mask==7) | (ref_mask==8)] = 0
            ref_mask[ref_mask!=0] = 1
        elif get_type == 'faceearneck':
            ref_mask[(ref_mask==0) | (ref_mask==17) | (ref_mask==18) | (ref_mask==16) | (ref_mask==15)| (ref_mask==9)] = 0
            ref_mask[ref_mask!=0] = 1
        elif get_type == 'color_guide':
            ref_mask[(ref_mask==2) | (ref_mask==3) | (ref_mask==4) | (ref_mask==5) | (ref_mask==6)] = 1
            ref_mask[(ref_mask==10) | (ref_mask==11) | (ref_mask==12)| (ref_mask==13)] = 1
            ref_mask[(ref_mask==15)| (ref_mask==14)] = 1
            ref_mask[(ref_mask==8)] = 7
            ref_mask[(ref_mask==18) | (ref_mask==9)] = 0
            ref_mask[(ref_mask==16)] = 0
        elif get_type == 'background':
            ref_mask[ref_mask!=0] = 1
            ref_mask = 1-ref_mask
        elif get_type == 'neck':
            ref_mask[ref_mask!=14] = 0
            ref_mask[ref_mask==14] = 1
        elif get_type == 'hat':
            ref_mask[ref_mask!=18] = 0
            ref_mask[ref_mask==18] = 1
        elif get_type == 'ear':
            ref_mask[ref_mask==7] = 8
            ref_mask[ref_mask!=8] = 0
            ref_mask[ref_mask==8] = 1
        elif get_type.isdigit():
            ref_mask[ref_mask!=int(get_type)] = 0
            ref_mask[ref_mask==int(get_type)] = 1
        else:
            print('nooooooooooooooooooooooooooooooooooooooo')
    else:
        ref_mask[ref_mask!=0] = 1
        if mask_number < 0:
            ref_mask = 1-ref_mask
    return ref_mask

def create_face_contour(point,mask):
    line = np.zeros_like(mask)
    for p in range(16):
        p1 = (int(point[p][0]), int(point[p][1]))
        p2 = (int(point[p+1][0]), int(point[p+1][1]))
        cv2.line(line, p1,p2, (1), thickness=1)
    line[:int(max(point[0][1],point[16][1])),:] = 0
    area = line.copy()
    fill_point = np.where(line!=0)
    start = fill_point[1][0]
    for p in range(1,len(fill_point[0])-1):
        if fill_point[0][p] != fill_point[0][p+1]:
            stop = fill_point[1][p]
            for q in range(start,stop+1):
                area[fill_point[0][p],q] = 1
            start = fill_point[1][p+1]
    whare_top = np.where(area!=0)
    area = scipy.ndimage.binary_erosion(area, iterations=3).astype(np.uint8)
    whare_top_e = np.where(area!=0)
    area[np.min(whare_top[0]):np.min(whare_top_e[0]),np.min(whare_top_e[1]):np.max(whare_top_e[1])] = 1
    area = area * (1-get_mask(mask,get_type='background'))
    return line,area

def fill_hair(color,skin_color,color_guide,point_face,shift_mask_hair,fill,raw_face_mask,mean_face_color):
    # color[(color_guide == 7) & (raw_face_mask == 17)] =  # fill ear
    # skin_color[(color_guide == 7) & (raw_face_mask == 17)] = 1
    # left
    tm = np.zeros_like(color_guide)
    tm[int(max(point_face[0][1],point_face[16][1])):,:int(point_face[8][0])] = 1
    fill_left = fill.copy()
    fill_left[int(max(point_face[0][1],point_face[16][1])):,int(point_face[8][0]):] = 0
    white = np.where(fill_left!=0)
    start = -1
    for i in range(len(white[0])):
        # find left color
        if start!= white[0][i]:
            start = white[0][i]
            c = color[white[0][i],white[1][i]-min(5,white[1][i])]
            ck = 0
            # ck = shift_mask_hair[white[0][i],white[1][i]-min(5,white[1][i])]
        if (color_guide[white[0][i],white[1][i]] == 7) and (raw_face_mask[white[0][i],white[1][i]] == 17):
            c = mean_face_color
            ck = 1
        color[white[0][i],white[1][i]] = c
        if ck == 1:
            skin_color[white[0][i],white[1][i]] = 1

    fill_right = fill.copy()
    fill_right[int(max(point_face[0][1],point_face[16][1])):,:int(point_face[8][0])] = 0
    white = np.where(fill_right!=0)
    start = -1
    for i in range(len(white[0])-1,-1,-1):
        if start!= white[0][i]:
            start = white[0][i]
            c = color[white[0][i],white[1][i]+min(5,255-white[1][i])]
            ck = 0
            # ck = shift_mask_hair[white[0][i],white[1][i]+min(5,255-white[1][i])]
        if (color_guide[white[0][i],white[1][i]] == 7) and (raw_face_mask[white[0][i],white[1][i]] == 17):
            c = mean_face_color
            ck = 1
        color[white[0][i],white[1][i]] = c
        if ck == 1:
            skin_color[white[0][i],white[1][i]] = 1
    return color,skin_color

# new line by line +/- 3 count
def get_new_hairmask(plain_face,mask_face,plain_face_hair,shift_mask_hair_hair,mask_outside):
    cut_point = 3
    merge_face_input = np.clip(plain_face+mask_face,0,1)
    whare_face_hair = np.where(plain_face_hair!=0)
    whare_face = np.where(merge_face_input!=0)
    plain_face_hair[:max(np.min(whare_face_hair[0]),np.min(whare_face[0])),:] = 0
    merge_face_input[:max(np.min(whare_face_hair[0]),np.min(whare_face[0])),:] = 0

    intersection_all = plain_face_hair*merge_face_input*shift_mask_hair_hair
    intersection_face = merge_face_input*shift_mask_hair_hair

    rm_hair = (intersection_face - intersection_all)*(1-mask_outside)
    hair_mask_rm = (shift_mask_hair_hair - rm_hair)*(1-mask_outside)

    rm_l = rm_hair.copy()
    rm_r = rm_hair.copy()
    rm_l[:,256//2:] = 0
    rm_r[:,:256//2] = 0
    # fill left hair_mask_rm
    whare_rm = np.where((rm_l)!=0)
    count_rm = 0
    if len(whare_rm[0]) > 1:
        row = whare_rm[1][0]
        for i in range(len(whare_rm[0])-1):
            col = whare_rm[0][i]
            if col != whare_rm[0][i+1] or i == len(whare_rm)-1 or whare_rm[1][i+1] != (whare_rm[1][i]+1):
                if shift_mask_hair_hair[whare_rm[0][i],whare_rm[1][i]+1] and shift_mask_hair_hair[whare_rm[0][i],whare_rm[1][i]+cut_point] != 0:
                    hair_mask_rm[col,row:whare_rm[1][i]+1] = 1
                    count_rm = count_rm + 1
                else:
                    if ((sum(shift_mask_hair_hair[col-1,whare_rm[1][i]:whare_rm[1][i]+cut_point]) > 1)
                        and (whare_rm[1][i]+1 - row) > cut_point):
                        hair_mask_rm[col,row:whare_rm[1][i]+1] = 1
                        count_rm = count_rm - 1
                    else:
                        hair_mask_rm[col,whare_rm[1][i]:256//2] = 0
                row = whare_rm[1][i+1]
    # fill right hair_mask_rm
    whare_rm = np.where((rm_r)!=0)
    count_rm = 0
    if len(whare_rm[0]) > 1:
        row = whare_rm[1][len(whare_rm[0])-1]
        for i in range(len(whare_rm[0])-1,0,-1):
            col = whare_rm[0][i]
            if col != whare_rm[0][i-1] or i == 1 or whare_rm[1][i-1] != (whare_rm[1][i]-1):
                if shift_mask_hair_hair[whare_rm[0][i],whare_rm[1][i]-1] and shift_mask_hair_hair[whare_rm[0][i],whare_rm[1][i]-cut_point] != 0:
                    hair_mask_rm[col,whare_rm[1][i]:row+1] = 1
                    count_rm = count_rm + 1
                else:
                    if ((sum(shift_mask_hair_hair[col-1,whare_rm[1][i]-cut_point:whare_rm[1][i]]) > 1)
                        and (row+1 - whare_rm[1][i]) > cut_point):
                        hair_mask_rm[col,whare_rm[1][i]:row+1] = 1
                        count_rm = count_rm - 1
                    else:
                        # hair_mask_rm[col,256//2:whare_rm[1][i]] = 0
                        hair_mask_rm[col,256//2:row+1] = 0
                row = whare_rm[1][i-1]

    return hair_mask_rm*shift_mask_hair_hair

def shift_img_a_b(img,a,b,shape_out):
    temp_shape_out = list(shape_out)
    temp_shape_out[1] = max(img.shape[1],shape_out[1])
    temp_img = np.zeros(temp_shape_out,np.uint8)
    shift_img = np.zeros(shape_out,np.uint8)

    if a > 0:
        temp_img[a:img.shape[0]+a,:img.shape[1]] = img[:temp_img.shape[0]-a,:temp_img.shape[1]]
    else:
        temp_img[:img.shape[0]+a,:img.shape[1]] = img[-a:temp_img.shape[0]-a,:temp_img.shape[1]]
    if b > 0:
        shift_img[:temp_img.shape[0],b:] = temp_img[:shift_img.shape[0],:shift_img.shape[1]-b]
    else:
        shift_img[:temp_img.shape[0],:temp_img.shape[1]+b] = temp_img[:shift_img.shape[0],-b:shift_img.shape[1]-b]
    return shift_img

def resize_shift_img_a_b(img,a,b,dim,img_type='img',extend=False):
    if img_type == 'mask': img_resize = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    else:
        if type(img).__module__ == np.__name__:
            img = Image.fromarray(img.astype(np.uint8))
        img_resize = img.resize(dim)
        img_resize = np.array(img_resize)

    # if extend:
    #     shape_out = list(np.array(img).shape)
    #     shape_out[0] = shape_out[0] + abs(a)
    #     shape_out[1] = shape_out[1] + abs(b)
    #     a = max(0,a)
    #     b = max(0,b)
    #     return shift_img_a_b(img_resize,a,b,shape_out)
    return shift_img_a_b(img_resize,a,b,np.array(img).shape)

def cal_resize(p_face,p_hair):
    w_face = np.mean(p_face[16:17],axis=0)[0] - np.mean(p_face[0:1],axis=0)[0]
    w_hair = np.mean(p_hair[16:17],axis=0)[0] - np.mean(p_hair[0:1],axis=0)[0]
    ratio = ((w_face/w_hair)-1)/2+1
    return ratio

def cal_shift_parms(p_face,p_hair,ratio=None):
    # ratio
    if ratio is None:
        ratio = cal_resize(p_face,p_hair)

    # reposition hair point with ratio
    p_hair = np.round((p_hair * ratio))
    # forget what is diff_size
    diff_size = 0
    p_hair = p_hair #+ diff_size

    # find diff hight
    center_x_p = (((p_face[0] + p_face[16])/2)[1] + p_face[8][1])/2
    center_x_hair = (((p_hair[0] + p_hair[16])/2)[1] + p_hair[8][1])/2
    a = int((center_x_p-center_x_hair) +diff_size)


    # find diff width (b)
    center_w_face = ((p_face[0] + p_face[16])/2)[0]
    center_w_hair = ((p_hair[0] + p_hair[16])/2)[0]
    b = int(center_w_face-center_w_hair+diff_size)

    return a,b,ratio

def get_mask_outside(mask_inside,shift_mask_hair_hair):
    mask_outside = np.zeros_like(mask_inside)
    whare_hair = np.where(shift_mask_hair_hair!=0) # where 0-h, 1-w
    whare_inside = np.where(mask_inside!=0)
    th_max = np.max(whare_inside[0])
    th_min = np.min(whare_inside[0])
    th_00 = th_01 = -1

    for p in range(len(whare_hair[0])):
        if whare_hair[0][p] == th_min:
            if th_00 == -1:
                th_00 = whare_hair[1][p]
            th_01 = whare_hair[1][p]
            mask_outside[:th_min,whare_hair[1][p]] = 1
    mask_outside[th_max+1:,:] = 1

    tw_max = np.max(whare_inside[1])
    tw_min = np.min(whare_inside[1])
    tw_00 = tw_01 = tw_10 = tw_11 = -1
    for p in range(len(whare_hair[1])):
        if whare_hair[1][p] == tw_min:
            if tw_00 == -1:
                tw_00 = whare_hair[0][p]
            tw_10 = whare_hair[0][p]
            mask_outside[whare_hair[0][p],:tw_min] = 1
        if whare_hair[1][p] == tw_max:
            if tw_01 == -1:
                tw_01 = whare_hair[0][p]
            tw_11 = whare_hair[0][p]
            mask_outside[whare_hair[0][p],tw_max+1:] = 1
    # fill corner
    if th_00 == tw_min and tw_00 == th_min:
        mask_outside[:th_min,:tw_min] = 1
    if th_01 == tw_max and tw_01 == th_min:
        mask_outside[:th_min,tw_max+1:] = 1
    return mask_outside


## create torch
def to_img(img,is_pil=True):
    batch_size, channels, img_h, img_w = img.shape
    grid_h = grid_w = 1
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    img = img.permute(1, 2, 0)
    img = img.cpu().numpy()
    if is_pil:
        return Image.fromarray(img)
    return img

def create_mask_torch(mask_path,device, resize=-1, is_pil = 1):
    if is_pil == 1:
        mask = np.array(mask_path)
    else:
        mask = np.array(Image.open(mask_path).convert("L"))

    mask[mask!=0] = 1

    if resize > 0:
        mask = np.array(Image.fromarray(mask).resize((resize,resize),Image.NEAREST))
    else:
        pass

    mask = np.expand_dims(mask,axis=0)
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask).float().to(device)
    return mask

def create_img_torch(img_path,device, size=256, is_pil = 0):
    transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # normalize the color range to [0,1].
        ]
    )
    if is_pil == 1:
        img = [transform(img_path)]
    else:
        img = [transform(Image.open(img_path).convert("RGB"))]
    img = torch.stack(img, 0).to(device)
    return img

def load_guide(main_dir,pair_dir,merge_name,size,device,is_pil=1):
    I_guide,I_guide_s = [],[]
    # load main guide
    I_guide.append(create_img_torch(Image.open(os.path.join(main_dir,merge_name+'_Iguide1.png')),device=device,is_pil=is_pil))
    I_guide_s.append(F.interpolate(I_guide[-1], size=(size, size), mode='area'))

    # load pair guide
    I_guide.append(create_img_torch(Image.open(os.path.join(pair_dir,merge_name+'_Iguide1.png')),device=device,is_pil=is_pil))
    I_guide_s.append(F.interpolate(I_guide[-1], size=(size, size), mode='area'))

    return I_guide,I_guide_s

def load_mask(main_dir,pair_dir,merge_name,size,device,is_pil=0):
    M_f,M_s,M_h,M_o,M_c,M_hs,M_os = [],[],[],[],[],[],[]
    # load main guide
    M_f.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Mf.png'),device=device,is_pil=is_pil))
    M_s.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Ms.png'),device=device,is_pil=is_pil))
    M_h.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Mh.png'),device=device,is_pil=is_pil))
    M_o.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Mo.png'),device=device,is_pil=is_pil))
    M_c.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Mc.png'),device=device,is_pil=is_pil))
    M_hs.append(create_mask_torch(os.path.join(main_dir,merge_name+'_Mhs.png'),device=device,is_pil=is_pil))
    M_os.append(F.interpolate(M_o[-1], size=(size, size), mode='nearest'))

    # load pair guide
    M_f.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Mf.png'),device=device,is_pil=is_pil))
    M_s.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Ms.png'),device=device,is_pil=is_pil))
    M_h.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Mh.png'),device=device,is_pil=is_pil))
    M_o.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Mo.png'),device=device,is_pil=is_pil))
    M_c.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Mc.png'),device=device,is_pil=is_pil))
    M_hs.append(create_mask_torch(os.path.join(pair_dir,merge_name+'_Mhs.png'),device=device,is_pil=is_pil))
    M_os.append(F.interpolate(M_o[-1], size=(size, size), mode='nearest'))

    return M_f,M_s,M_h,M_o,M_c,M_hs,M_os

## predict_mask
def mask_to_image(mask,size):
    return Image.fromarray((mask).astype(np.uint8)).resize(size,Image.NEAREST)


# atts = [0:bg, 1'skin', 2'l_brow', 3'r_brow', 4'l_eye', 5'r_eye', 6'eye_g',
#        7'l_ear',8 'r_ear', 9'ear_r',10 'nose',  11'mouth',
#       12'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17'hair', 18 'hat']
def predict_mask(net,image,size=256,is_torch=False):

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        image = image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.cuda()
        out = net(img)[0]
        if is_torch:
            return out
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    return mask_to_image(parsing,(size,size))

def create_mask_ignore(eg3d_mask,raw_mask):
    mask_ignore = raw_mask.copy()
    mask_ignore[(mask_ignore==15) | (mask_ignore==9)] = 18
    mask_ignore[mask_ignore!=18] = 0
    mask_ignore[mask_ignore==18] = 1

    mask_ignore = scipy.ndimage.binary_dilation(mask_ignore, iterations=5).astype(np.uint8)
    mask_ignore = 1-mask_ignore

    mask_ignore = mask_ignore * np.array(eg3d_mask)[:,:,0]

    return mask_ignore

def update_guide(main_dir,name,state1_dir,device,predict_mask_model,landmarks_detector,n_guide=0):
    Iguide1_dir = os.path.join(main_dir,'{}_Iguide1.png'.format(name))
    Mf_dir = os.path.join(main_dir,'{}_Mf.png'.format(name))
    Mh_dir = os.path.join(main_dir,'{}_Mh.png'.format(name))
    Ms_dir = os.path.join(main_dir,'{}_Ms.png'.format(name))
    Mo_dir = os.path.join(main_dir,'{}_Mo.png'.format(name))
    Mhn_dir = os.path.join(main_dir,'{}_shift_hair_mask.png'.format(name))
    Mc_dir = os.path.join(main_dir,'{}_Mc.png'.format(name))
    Moutside_dir = os.path.join(main_dir,'{}_mask_outside.png'.format(name))

    Mhs_dir = os.path.join(main_dir,'{}_Mhs.png'.format(name))
    new_M_hs = np.array(Image.open(Mhs_dir)) // 255
    ## state1_dir = main_dir + '{}/{}addMhsMomseb0wf3wph3r1synth999_last0.png'.format(name,name)

    Moutside = np.array(Image.open(Moutside_dir)) // 255

    if n_guide == 0:
    	Mhmesh_dir = os.path.join(main_dir,'{}_hairmesh_mask.png'.format(name))
    	M_hmesh = np.array(Image.open(Mhmesh_dir)) // 255

    else:
        Mfmesh_dir = os.path.join(main_dir,'{}_facemesh_mask.png'.format(name))
        M_fmesh = np.array(Image.open(Mfmesh_dir)) // 255

    I_f_name = os.path.join(main_dir,'{}_face.png'.format(name))
    # I_f_name = dataset_dir+'/{}.png'.format(face)
    I_f = Image.open(I_f_name)
    mask_I_f = predict_mask(predict_mask_model,I_f,256)
    # I_h_name = dataset_dir+'/{}.png'.format(hair)
    I_h_name = os.path.join(main_dir,'{}_shift_hair.png'.format(name))

    kpoint_face = get_keypoint_biggest(landmarks_detector,I_f_name) #// 4

    I_h = Image.open(I_h_name)
    mask_I_h = predict_mask(predict_mask_model,I_h,256)
    mask_I_h *= (1-Moutside)
    O_1 = Image.open(state1_dir)
    mask_O_1 = predict_mask(predict_mask_model,O_1,256)

    if kpoint_face is not None:
        mask_target,mask_remask = fix_mask_face(np.array(mask_I_f),np.array(mask_O_1),I_f_name,kpoint_face,return_mask_remask=True)
    else:
        mask_remask = np.zeros_like(mask_I_f)

    I_guide = np.array(Image.open(Iguide1_dir).resize((256,256)))
    I_f = np.array(I_f.resize((256,256)))
    I_h = np.array(I_h.resize((256,256)))
    O_1 = np.array(O_1.resize((256,256)))
    M_h = np.array(Image.open(Mh_dir)) // 255
    M_f = np.array(Image.open(Mf_dir)) // 255
    M_s = np.array(Image.open(Ms_dir)) // 255
    M_o = np.array(Image.open(Mo_dir)) // 255
    M_hn = np.array(Image.open(Mhn_dir)) // 255
    M_c = np.array(Image.open(Mc_dir)) // 255
    M_f = M_f* (1-get_mask(np.array(mask_I_f),get_type='neck'))

    # face
    img_out = np.array(I_f)
    mask_gan_hair = get_mask(np.array(mask_O_1),get_type='hair')
    mask_If_hair = get_mask(np.array(mask_I_f),get_type='hair')
    mask_If_hair_d = scipy.ndimage.binary_dilation(mask_If_hair, iterations=5).astype(np.uint8)
    img_out = img_out * (1-mask_If_hair_d[:,:,np.newaxis]) + I_guide * (mask_If_hair_d[:,:,np.newaxis])
    mask_If_hair_d *= (1-mask_gan_hair)
    img_out = img_out * (1-mask_If_hair_d[:,:,np.newaxis]) + O_1 * (mask_If_hair_d[:,:,np.newaxis])

    M_s_d = scipy.ndimage.binary_dilation(M_s, iterations=5).astype(np.uint8)
    img_out = img_out * (1-M_s_d[:,:,np.newaxis]) + O_1 * (M_s_d[:,:,np.newaxis])

    # put possible to be hair
    mask_Ih_hair_d = scipy.ndimage.binary_dilation(M_hn, iterations=5).astype(np.uint8)

    # mask_Ih_hair_d = M_hn
    img_out = img_out * (1-mask_Ih_hair_d[:,:,np.newaxis]) + O_1 * (mask_Ih_hair_d[:,:,np.newaxis])
    # top hair
    mask_top_hairgan = mask_gan_hair.copy()
    if kpoint_face is not None:
        mask_top_hairgan[int(max(kpoint_face[0][1],kpoint_face[16][1])):,:] = 0
    mask_top_hairgan = scipy.ndimage.binary_dilation(mask_top_hairgan, iterations=5).astype(np.uint8)

    img_out = img_out * (1-mask_top_hairgan[:,:,np.newaxis]) + O_1 * (mask_top_hairgan[:,:,np.newaxis])

    # hair in hat
    mask_bg = np.array(mask_O_1.copy())
    mask_bg[mask_bg==18] = 0
    mask_bg[mask_bg==16] = 0
    mask_bg[mask_bg==15] = 0
    mask_bg[mask_bg==9] = 0
    mask_bg[mask_bg!=0] = 1
    mask_morehair = get_mask(np.array(mask_I_h),get_type='hair') * mask_bg
    img_out = img_out * (1-mask_morehair[:,:,np.newaxis]) + O_1 * (mask_morehair[:,:,np.newaxis])

    temp = (1-M_o) * mask_gan_hair
    img_out = img_out * (1-temp[:,:,np.newaxis]) + O_1 * (temp[:,:,np.newaxis])

    mask_bg = np.array(mask_I_f.copy())
    mask_bg[mask_bg==18] = 0
    mask_bg[mask_bg==16] = 0
    mask_bg[mask_bg==15] = 0
    mask_bg[mask_bg==9] = 0
    mask_bg[mask_bg!=0] = 1
    temp = (1-M_o) * mask_bg
    img_out = img_out * (1-temp[:,:,np.newaxis]) + O_1 * (temp[:,:,np.newaxis])

    # put ear
    mask_ear = get_mask(np.array(mask_O_1),get_type='ear')
    mask_ear *= get_mask(np.array(mask_I_f),get_type='hair')
    img_out = img_out * (1-mask_ear[:,:,np.newaxis]) + O_1 * (mask_ear[:,:,np.newaxis])

    # put real hair
    if n_guide == 0:
        img_out = img_out * (1-M_hmesh[:,:,np.newaxis]) + I_guide * (M_hmesh[:,:,np.newaxis])
    else:
    	img_out = img_out * (1-M_h[:,:,np.newaxis]) + I_guide * (M_h[:,:,np.newaxis])

    # fix mask
    mask_remask2 = mask_remask * (1-mask_If_hair)
    fill = fill_remaskmask(mask_remask2,img_out)
    img_out = img_out * (1-mask_remask2[:,:,np.newaxis]) + fill * (mask_remask2[:,:,np.newaxis])

    new_M_h = M_h.copy()
    mask_hat_d = scipy.ndimage.binary_dilation(get_mask(np.array(mask_I_h),get_type='hair'), iterations=5).astype(np.uint8)
    new_M_h *= (1-mask_hat_d)
    new_M_f = M_f * (1-M_s_d)
    new_M_f *= (1-mask_hat_d)

    if n_guide == 0:
    	new_M_o = (1-(M_hn*(1-M_h)))*(1-M_s)*(1-mask_ear)*M_o * (1-M_hmesh)
    else:
    	new_M_o = (1-(M_hn*(1-M_h)))*(1-M_s)*(1-mask_ear)*M_o
    mask_neck = mask_If_hair_d * get_mask(np.array(mask_I_f),get_type='neck')
    new_M_o *= (1-mask_neck) * (1-mask_remask2)* (1-mask_top_hairgan)
    new_M_o[new_M_h==1] = 1
    new_M_o = scipy.ndimage.binary_erosion(new_M_o, iterations=5).astype(np.uint8)
    new_M_s = M_s
    new_M_s[mask_hat_d==1] = 1

    # create mask bg
    if n_guide == 0:
        mask_h_real = M_hn
        mask_h_real =  scipy.ndimage.binary_dilation(mask_h_real, iterations=5).astype(np.uint8)
        mask_out = 1-mask_h_real
        mask_h_gan = get_mask(np.array(mask_O_1),get_type='hair')
        mask_h_gan =  scipy.ndimage.binary_dilation(mask_h_gan, iterations=10).astype(np.uint8)
        mask_out = mask_out * (1-mask_h_gan)

        mask_f = np.array(mask_O_1.copy())
        mask_f[(mask_f==1)|(mask_f==2)|(mask_f==3)|(mask_f==4)|(mask_f==5)|(mask_f==6)] = 1
        mask_f[(mask_f==10)|(mask_f==11)|(mask_f==12)|(mask_f==13)|(mask_f==14)] = 1
        mask_f[mask_f!=1] = 0
        mask_bg = np.array(mask_I_f.copy())
        mask_bg[mask_bg==18] = 0
        mask_bg[mask_bg==16] = 0
        mask_bg[mask_bg==15] = 0
        mask_bg[mask_bg==9] = 0
        mask_bg[mask_bg!=0] = 1
        mask_f = mask_f * mask_bg
        mask_out = mask_out * (1-mask_f)
        out_M_bg = create_mask_torch(Image.fromarray(mask_out.astype(np.uint8)*255),device=device, is_pil = 1)
        Image.fromarray(mask_out.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_bg.png'.format(name)))
    else:
        out_M_bg = None

    out_guide = create_img_torch(Image.fromarray(img_out.astype(np.uint8)),device=device, is_pil = 1)
    out_M_f = create_mask_torch(Image.fromarray(new_M_f.astype(np.uint8)*255),device=device, is_pil = 1)
    out_M_h = create_mask_torch(Image.fromarray(new_M_h.astype(np.uint8)*255),device=device, is_pil = 1)
    out_M_s = create_mask_torch(Image.fromarray(new_M_s.astype(np.uint8)*255),device=device, is_pil = 1)
    out_M_hs = create_mask_torch(Image.fromarray(new_M_hs.astype(np.uint8)*255),device=device, is_pil = 1)
    out_M_o = create_mask_torch(Image.fromarray(new_M_o.astype(np.uint8)*255),device=device, is_pil = 1)

    Image.fromarray(img_out.astype(np.uint8)).save(os.path.join(main_dir,'{}_Iguide2.png'.format(name)))
    Image.fromarray(new_M_f.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_f.png'.format(name)))
    Image.fromarray(new_M_h.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_h.png'.format(name)))
    Image.fromarray(new_M_s.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_s.png'.format(name)))
    Image.fromarray(new_M_hs.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_hs.png'.format(name)))
    Image.fromarray(new_M_o.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_out_M_o.png'.format(name)))

    return out_guide,out_M_f,out_M_h,out_M_s,out_M_hs,out_M_o, out_M_bg

def update_guide_and_mask(main_dir,pair_dir,merge_name,size,device,predict_mask_model,landmarks_detector):
    I_guide,I_guide_s = [],[]
    M_f,M_s,M_h,M_o,M_c,M_hs,M_os = [],[],[],[],[],[],[]

    # main input
    output_opt0 = f'{main_dir}/{merge_name}-iopt0.png'

    tI_guide,tM_f,tM_h,tM_s,tM_hs,tM_o,M_bg = update_guide(main_dir,merge_name,output_opt0,device=device,predict_mask_model=predict_mask_model,landmarks_detector=landmarks_detector,n_guide=0)

    tM_c = tM_o
    tI_guide_s = F.interpolate(tI_guide, size=(size,size), mode='area')
    tM_os = F.interpolate(tM_o, size=(size,size), mode='nearest')

    tM_os[tM_os!=1] = 1
    I_guide.append(tI_guide)
    I_guide_s.append(tI_guide_s)
    M_f.append(tM_f)
    M_h.append(tM_h)
    M_s.append(tM_s)
    M_o.append(tM_o)
    M_hs.append(tM_hs)
    M_c.append(tM_c)
    M_os.append(tM_os)
    # pair
    output_opt0 = f'{pair_dir}/{merge_name}-iopt0.png'
    tI_guide,tM_f,tM_h,tM_s,tM_hs,tM_o,_ = update_guide(pair_dir,merge_name,output_opt0,device=device,predict_mask_model=predict_mask_model,landmarks_detector=landmarks_detector,n_guide=1)

    tM_c = tM_o
    tI_guide_s = F.interpolate(tI_guide, size=(size,size), mode='area')
    tM_os = F.interpolate(tM_o, size=(size,size), mode='nearest')

    tM_os[tM_os!=1] = 1
    I_guide.append(tI_guide)
    I_guide_s.append(tI_guide_s)
    M_f.append(tM_f)
    M_h.append(tM_h)
    M_s.append(tM_s)
    M_o.append(tM_o)
    M_hs.append(tM_hs)
    M_c.append(tM_c)
    M_os.append(tM_os)

    return I_guide,I_guide_s, M_f,M_s,M_h,M_o,M_c,M_hs,M_os

def get_mask_pti(main_dir,name,state1_dir,device,predict_mask_model,type='face'):
    if type == 'hair':
        I_gan = Image.open(state1_dir).resize((256,256))
        mask_I_gan = np.array(predict_mask(predict_mask_model,I_gan,256))

        face_dir = os.path.join(main_dir,'{}_shift_hair.png'.format(name))
        I_f = Image.open(face_dir).resize((256,256))
        mask_dir = os.path.join(main_dir,'{}_shift_hair_mask.png'.format(name))
        mask_out = np.array(Image.open(mask_dir)) // 255
        mask_out = scipy.ndimage.binary_erosion(mask_out, iterations=10).astype(np.uint8)

        mask_h_gan = get_mask(np.array(mask_I_gan),get_type='hair')

        mask_out = mask_out * mask_h_gan

    else:
        I_gan = Image.open(state1_dir).resize((256,256))
        mask_I_gan = np.array(predict_mask(predict_mask_model,I_gan,256))

        face_dir = os.path.join(main_dir,'{}_face.png'.format(name))
        I_f = Image.open(face_dir)
        mask_I_f =np.array(predict_mask(predict_mask_model,I_f,256))

        mask_h_real = get_mask(np.array(mask_I_f),get_type='hair')
        mask_h_real =  scipy.ndimage.binary_dilation(mask_h_real, iterations=5).astype(np.uint8)

        mask_out = 1-mask_h_real

        mask_h_gan = get_mask(np.array(mask_I_gan),get_type='hair')
        mask_h_gan =  scipy.ndimage.binary_dilation(mask_h_gan, iterations=10).astype(np.uint8)

        mask_out = mask_out * (1-mask_h_gan)

        mask_f = mask_I_gan.copy()
        mask_f[(mask_f==1)|(mask_f==2)|(mask_f==3)|(mask_f==4)|(mask_f==5)|(mask_f==6)] = 1
        mask_f[(mask_f==10)|(mask_f==11)|(mask_f==12)|(mask_f==13)|(mask_f==14)] = 1
        mask_f[mask_f!=1] = 0
        mask_bg = mask_I_f.copy()
        # mask_bg[mask_bg==18] = 0
        mask_bg[mask_bg==16] = 0
        mask_bg[mask_bg==15] = 0
        mask_bg[mask_bg==9] = 0
        mask_bg[mask_bg!=0] = 1
        mask_f = mask_f * mask_bg

        mask_out = mask_out * (1-mask_f)

        mask_f_real = mask_I_f.copy()
        mask_f_real[(mask_f_real==1)|(mask_f_real==2)|(mask_f_real==3)|(mask_f_real==4)|(mask_f_real==5)|(mask_f_real==6)] = 1
        mask_f_real[(mask_f_real==10)|(mask_f_real==11)|(mask_f_real==12)|(mask_f_real==13)] = 1
        mask_f_real[mask_f_real!=1] = 0

        mask = mask_f_real.copy()
        mask *= (1-mask_h_real)
        mask *= (1-mask_h_gan)
        mask = scipy.ndimage.binary_erosion(mask, iterations=10).astype(np.uint8)

        mask_out[mask==1] = 1
        # mask = np.array(Image.open(mask_dir)) // 255

    img_out = np.array(I_f) * mask_out[:,:,np.newaxis] + np.array(I_gan) * ( 1-mask_out[:,:,np.newaxis])

    out = create_mask_torch(Image.fromarray(mask_out.astype(np.uint8)*255),device=device, is_pil = 1)
    Image.fromarray(mask_out.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_M_focus_pti.png'.format(name)))
    out_guide = create_img_torch(Image.fromarray(img_out.astype(np.uint8)),device=device, is_pil = 1)

    return out,out_guide

def fill_mask(fill,maskgan,maskreal):
    # left
    fill_out = fill.copy()
    fill_left = fill.copy()
    fill_left[:,fill.shape[0]//2:] = 0
    white = np.where(fill_left!=0)
    start = -1
    for i in range(len(white[0])):
        # find left color
        if start!= white[0][i]:
            start = white[0][i]
            c = maskgan[white[0][i],white[1][i]-min(1,white[1][i])]
        if c == 17:
            fill_out[white[0][i],white[1][i]] = c
        elif maskreal[white[0][i],white[1][i]] == 17:
            fill_out[white[0][i],white[1][i]] = 0
        else:
            fill_out[white[0][i],white[1][i]] = maskreal[white[0][i],white[1][i]]
    # right
    fill_right = fill.copy()
    fill_right[:,:fill.shape[0]//2] = 0
    white = np.where(fill_right!=0)
    start = -1
    for i in range(len(white[0])-1,-1,-1):
        if start!= white[0][i]:
            start = white[0][i]
            c = maskgan[white[0][i],white[1][i]+min(1,255-white[1][i])]
        if c == 17:
            fill_out[white[0][i],white[1][i]] = c
        elif maskreal[white[0][i],white[1][i]] == 17:
            fill_out[white[0][i],white[1][i]] = 0
        else:
            fill_out[white[0][i],white[1][i]] = maskreal[white[0][i],white[1][i]]
    return fill_out

def fix_mask_face(mask_input,mask_target,img_input_name,point_face,return_mask_remask=False):
    _, plain_face = create_face_contour(point_face,mask_input)

    mask_noface = get_mask(np.array(mask_target),get_type='face')
    mask_noface[:int(max(point_face[0][1],point_face[16][1])),:] = 0
    mask_noface[plain_face==1] = 0
    mask_noface[get_mask(np.array(mask_input),get_type='faceneck')==1] = 0

    mask_remask = mask_noface.copy()
    fill_out = fill_mask(mask_remask,mask_target,mask_input)
    mask_output = mask_target.copy()
    mask_output = mask_output * (1-mask_remask) + fill_out * mask_remask

    if return_mask_remask:
        return mask_output,mask_remask#,mask_noface
    else:
        return mask_output

def fill_remaskmask(fill,maskgan):
    # left
    fill_out = maskgan.copy()
    fill_left = fill.copy()
    fill_left[:,fill.shape[0]//2:] = 0
    white = np.where(fill_left!=0)
    start = -1
    for i in range(len(white[0])):
        # find left color
        if start!= white[0][i]:
            start = white[0][i]
            c = maskgan[white[0][i],max(0,white[1][i]-1)]
        fill_out[white[0][i],white[1][i]] = c
    fill_right = fill.copy()
    fill_right[:,:fill.shape[0]//2] = 0
    white = np.where(fill_right!=0)
    start = -1
    for i in range(len(white[0])-1,-1,-1):
        if start!= white[0][i]:
            start = white[0][i]
            c = maskgan[white[0][i],min(maskgan.shape[0]-1,white[1][i]+1)]
        fill_out[white[0][i],white[1][i]] = c
    return fill_out

def get_bg_mask(main_dir,name,predict_mask_model,device):
    Mo_dir = os.path.join(main_dir,'{}_mask_outside.png'.format(name))
    M_o = np.array(Image.open(Mo_dir)) // 255

    Mh_dir = os.path.join(main_dir,'{}_Mh.png'.format(name))
    M_h= np.array(Image.open(Mh_dir)) // 255

    I_f_name = os.path.join(main_dir,'{}_face.png'.format(name))
    I_f = Image.open(I_f_name).resize((256,256))
    mask_I_f = predict_mask(predict_mask_model,I_f,256)

    mask_bg = np.array(mask_I_f.copy())
    mask_bg[mask_bg==18] = 0
    mask_bg[mask_bg==16] = 0
    mask_bg[mask_bg==15] = 0
    mask_bg[mask_bg==9] = 0
    mask_bg[mask_bg!=0] = 1
    mask_bg[M_h!=0] = 0
    mask_out = (1-mask_bg) * (1-M_o) * (1-M_h)
    mask_out = 1 - scipy.ndimage.binary_dilation((1-mask_out), iterations=5).astype(np.uint8)

    bg_count = np.sum(mask_out)
    out_Mbg = create_mask_torch(Image.fromarray(mask_out.astype(np.uint8)*255),device=device, is_pil = 1)
    Image.fromarray(mask_out.astype(np.uint8)*255).save(os.path.join(main_dir,'{}_Mbg.png'.format(name)))

    return out_Mbg,bg_count

## stylegan preprocessing
def cal_position(face_landmarks):
    lm = np.array(face_landmarks)
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2
    return quad,qsize

def find_coefs(original_coords, warped_coords):
        matrix = []
        for p1, p2 in zip(original_coords, warped_coords):
            matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
            matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

        A = np.matrix(matrix, dtype=np.float)
        B = np.array(warped_coords).reshape(8)

        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res).reshape(8)

# change mask to list
def stylegan_preprocess(img,face_landmarks,mask=None):
    output_size=1024
    transform_size=4096
    enable_padding=True

    if mask is not None:
        mask = Image.fromarray(mask.astype(np.uint8))

    stylegan_params = []

    quad,qsize = cal_position(face_landmarks)

    stylegan_params.append(img.size) # 0 raw_size
    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, Image.ANTIALIAS)
        if mask is not None:
            mask = mask.resize(rsize, Image.NEAREST)
        quad /= shrink
        qsize /= shrink

    stylegan_params.append(img.size) # 1 shrink_size


    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    stylegan_params.append(border) # 2 border

    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
    #     left, top, right, bottom
        img = img.crop(crop)
        if mask is not None:
            mask = mask.crop(crop)
        quad -= crop[0:2]
    stylegan_params.append(crop) # 3 crop

    # Pad.
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        if mask is not None:
            if len(np.array(mask).shape) == 3:
                mask = np.pad(np.float32(mask), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), mode='constant')
            else: mask = np.pad(np.float32(mask), ((pad[1], pad[3]), (pad[0], pad[2])), mode='constant')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask_pad = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask_pad * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask_pad, 0.0, 1.0)
        img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        # don't pad value on mask
        if mask is not None:
            mask = Image.fromarray(np.uint8(np.clip(np.rint(mask), 0, 255)))
        quad += pad[:2]

    stylegan_params.append(pad) # 4 pad
    stylegan_params.append(img.size) # 5 pad_size
    pad_quad = (quad + 0.5).flatten()
    stylegan_params.append(pad_quad) # 6 pad_quad

    # Transform.
    img = img.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.BILINEAR)
    if mask is not None:
        mask = mask.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(), Image.NEAREST)

    # (x0, y0, x1, y1, x2, y2, x3, y3)
    # upper left, lower left, lower right, and upper right corner
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.ANTIALIAS)
        if mask is not None:
            mask = mask.resize((output_size, output_size), Image.NEAREST)

    # if mask is not None:
    #     mask = np.array(mask)[:,:,0]
    return img, stylegan_params, mask

# stylegan_params 0 raw_size, 1 shrink_size, 2 border, 3 crop, 4pad, 5 pad_size, 6 pad_quad
def inv_stylegan_preprocess(inv_img,stylegan_params,mask=None):
    output_size=1024
    transform_size=4096
    enable_padding=True

    if output_size < transform_size:
        inv_img = inv_img.resize((transform_size, transform_size), Image.ANTIALIAS)
        if mask is not None: mask = mask.resize((transform_size, transform_size), Image.NEAREST)
    # Transform.
    pad_quad = stylegan_params[6]
    coeffs = find_coefs(
      [(pad_quad[0],pad_quad[1]), (pad_quad[2], pad_quad[3]), (pad_quad[4], pad_quad[5]), (pad_quad[6], pad_quad[7])],
      [(0, 0), (0,transform_size), (transform_size, transform_size), (transform_size,0)]
    )
    inv_img = inv_img.transform(stylegan_params[5], Image.PERSPECTIVE, coeffs, Image.BILINEAR)
    if mask is not None: mask = mask.transform(stylegan_params[5], Image.PERSPECTIVE, coeffs, Image.NEAREST)
    # Pad.
    pad = stylegan_params[4]
    if enable_padding and max(pad) > stylegan_params[2] - 4:
        inv_img = Image.fromarray(np.array(inv_img)[pad[1]:-pad[3],pad[0]:-pad[2],:])
        if mask is not None: mask = Image.fromarray(np.array(mask)[pad[1]:-pad[3],pad[0]:-pad[2],:])

    # Crop.
    crop = stylegan_params[3]
    shrink_size = stylegan_params[1]
    if crop[2] - crop[0] < shrink_size[0] or crop[3] - crop[1] < shrink_size[1]:
        inv_img = Image.fromarray(np.pad(np.array(inv_img), ((crop[1], shrink_size[1]-crop[3]), (crop[0], shrink_size[0]-crop[2]),(0,0)), mode='constant'))
        if mask is not None:
            if len(np.array(mask).shape) == 3:
                mask = Image.fromarray(np.pad(np.array(mask), ((crop[1], shrink_size[1]-crop[3]), (crop[0], shrink_size[0]-crop[2]),(0,0)), mode='constant'))
            else:
                mask = Image.fromarray(np.pad(np.array(mask), ((crop[1], shrink_size[1]-crop[3]), (crop[0], shrink_size[0]-crop[2])), mode='constant'))

    inv_img = inv_img.resize(stylegan_params[0], Image.ANTIALIAS)
    if mask is not None:
        mask = mask.resize(stylegan_params[0], Image.NEAREST)
        mask = np.array(mask)
        mask[mask!=255] = 0
        mask = Image.fromarray(mask.astype(np.uint8))

    return inv_img, mask


from scipy.io import loadmat
import os.path as osp
# load landmarks for standard face, which is used for image preprocessing
def load_lm3d(bfm_folder):

    Lm3D = loadmat(osp.join(bfm_folder, 'similarity_Lm3D_all.mat'))
    Lm3D = Lm3D['lm']

    # calculate 5 facial landmarks using 68 landmarks
    lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
    Lm3D = np.stack([Lm3D[lm_idx[0], :], np.mean(Lm3D[lm_idx[[1, 2]], :], 0), np.mean(
        Lm3D[lm_idx[[3, 4]], :], 0), Lm3D[lm_idx[5], :], Lm3D[lm_idx[6], :]], axis=0)
    Lm3D = Lm3D[[1, 2, 0, 3, 4], :]

    return Lm3D

# calculating least square problem for image alignment
def POS(xp, x):
    npts = xp.shape[1]

    A = np.zeros([2*npts, 8])

    A[0:2*npts-1:2, 0:3] = x.transpose()
    A[0:2*npts-1:2, 3] = 1

    A[1:2*npts:2, 4:7] = x.transpose()
    A[1:2*npts:2, 7] = 1

    b = np.reshape(xp.transpose(), [2*npts, 1])

    k, _, _, _ = np.linalg.lstsq(A, b)

    R1 = k[0:3]
    R2 = k[4:7]
    sTx = k[3]
    sTy = k[7]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
    t = np.stack([sTx, sTy], axis=0)

    return t, s

# resize and crop images for face reconstruction
def resize_n_crop_img(img, lm, t, s, target_size=224., mask=None):
    w0, h0 = img.size
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
    right = left + target_size
    up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
    below = up + target_size

    img = img.resize((w, h), resample=Image.BICUBIC)
    img = img.crop((left, up, right, below))

    if mask is not None:
        mask = mask.resize((w, h), resample=Image.NEAREST)
        mask = mask.crop((left, up, right, below))

    lm = np.stack([lm[:, 0] - t[0] + w0/2, lm[:, 1] -
                  t[1] + h0/2], axis=1)*s
    lm = lm - np.reshape(
            np.array([(w/2 - target_size/2), (h/2-target_size/2)]), [1, 2])

    return img, lm, mask

def align_img(img, lm, lm3D, mask=None, target_size=224., rescale_factor=102.):
    """
    Return:
        transparams        --numpy.array  (raw_W, raw_H, scale, tx, ty)
        img_new            --PIL.Image  (target_size, target_size, 3)
        lm_new             --numpy.array  (68, 2), y direction is opposite to v direction
        mask_new           --PIL.Image  (target_size, target_size)

    Parameters:
        img                --PIL.Image  (raw_H, raw_W, 3)
        lm                 --numpy.array  (68, 2), y direction is opposite to v direction
        lm3D               --numpy.array  (5, 3)
        mask               --PIL.Image  (raw_H, raw_W, 3)
    """

    w0, h0 = img.size
    if lm.shape[0] != 5:
        lm5p = extract_5p(lm)
    else:
        lm5p = lm

    # calculate translation and scale factors using 5 facial landmarks and standard landmarks of a 3D face
    t, s = POS(lm5p.transpose(), lm3D.transpose())
    s = rescale_factor/s

    # processing the image
    img_new, lm_new, mask_new = resize_n_crop_img(img, lm, t, s, target_size=target_size, mask=mask)
    trans_params = np.array([w0, h0, s, t[0], t[1]])

    return trans_params, img_new, lm_new, mask_new

def eg3d_preprocess(im,mask=None,lm_path='',lm3d_std_folder="Deep3DFaceRecon_pytorch/BFM"):
    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size_eg3d = 512

    eg3d_params = []
    _,H = im.size
    lm = np.loadtxt(lm_path).astype(np.float32)
    lm = lm.reshape([-1, 2])
    lm[:, -1] = H - 1 - lm[:, -1]
    eg3d_params.append(im.size) # 0

    lm3d_std = load_lm3d(lm3d_std_folder)
    trans_params, im_high, _, mask, = align_img(im, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor,mask=mask)
    eg3d_params.append(trans_params) # 1
    eg3d_params.append(im_high.size) # 2

    left = int(im_high.size[0]/2 - center_crop_size/2)
    upper = int(im_high.size[1]/2 - center_crop_size/2)
    right = left + center_crop_size
    lower = upper + center_crop_size
    eg3d_params.append((left, upper, right,lower)) # 3

    im_cropped = im_high.crop((left, upper, right,lower))
    eg3d_params.append(im_cropped.size) # 4

    im_cropped = im_cropped.resize((output_size_eg3d, output_size_eg3d), resample=Image.LANCZOS)

    if mask is not None:
        mask = mask.crop((left, upper, right,lower))
        mask = mask.resize((output_size_eg3d, output_size_eg3d), resample=Image.NEAREST)

    return im_cropped,eg3d_params,mask

# eg3d_params; 0 im.size, 1 trans_params, 2 im_high, 3 (left, upper, right,lower), 4 im_cropped
def inv_eg3d_preprocess(inv_img,eg3d_params,mask=None):
    target_size = 1024.

    inv_img = inv_img.resize(eg3d_params[4], resample=Image.BICUBIC) #BICUBIC

    left, upper, right,lower = eg3d_params[3]
    inv_img = Image.fromarray(np.pad(np.array(inv_img), ((upper, eg3d_params[2][1]-lower), (left, eg3d_params[2][0]-right),(0,0)), mode='constant'))
    if mask is not None:
        mask = mask.resize(eg3d_params[4], resample=Image.NEAREST)
        mask = Image.fromarray(np.pad(np.array(mask), ((upper, eg3d_params[2][1]-lower), (left, eg3d_params[2][0]-right),(0,0)), mode='constant'))

    w0, h0, s, t_0, t_1 = eg3d_params[1]
    w0, h0 = eg3d_params[0]
    w = (w0*s).astype(np.int32)
    h = (h0*s).astype(np.int32)
    left = (w/2 - target_size/2 + float((t_0 - w0/2)*s)).astype(np.int32)
    up = (h/2 - target_size/2 + float((h0/2 - t_1)*s)).astype(np.int32)

    inv_img = inv_img.crop((-left, -up, w-left, h-up))
    inv_img = inv_img.resize(eg3d_params[0], resample=Image.LANCZOS)
    if mask is not None:
        mask = mask.crop((-left, -up, w-left, h-up))
        mask = mask.resize(eg3d_params[0], resample=Image.NEAREST)

    return inv_img,mask





if __name__ == "__main__":
    landmarks_model_path = 'checkpoints/shape_predictor_68_face_landmarks.dat'
    predict_mask_model_path='checkpoints/79999_iter.pth'

    # PREDICT KEYPOINT MODEL
    landmarks_detector = LandmarksDetector(landmarks_model_path)
    # PREDICT MASK MODEL
    predict_mask_model = BiSeNet(n_classes=19)
    predict_mask_model.cuda()
    predict_mask_model.load_state_dict(torch.load(predict_mask_model_path))
    predict_mask_model.eval()
