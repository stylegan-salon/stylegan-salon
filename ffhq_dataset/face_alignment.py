import numpy as np
import scipy.ndimage
import os
import PIL.Image


def image_align(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

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

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)

        # print('1img',img.size)
        # print('quad',quad)
        # print('qsize',qsize)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            quad /= shrink
            qsize /= shrink
            # print('2img',img.size)
        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            quad -= crop[0:2]
            # print('3img',img.size)

        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
            h, w, _ = img.shape
            y, x, _ = np.ogrid[:h, :w, :1]
            mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            blur = qsize * 0.02
            img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            quad += pad[:2]
            # print('4img',img.size)

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            # print('5img',img.size)
            # print('quad',quad)
        # Save aligned image.
        if dst_file != '':
            img.save(dst_file, 'PNG')
        return img

def image_align_noblur(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True,pair_file=None,pair_out=None):
        # Align function from FFHQ dataset pre-processing step
        # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py

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

        # Load in-the-wild image.
        if not os.path.isfile(src_file):
            print('\nCannot find source image. Please run "--wilds" before "--align".')
            return
        img = PIL.Image.open(src_file)
        pair_img = PIL.Image.open(pair_file)

        # Shrink.
        shrink = int(np.floor(qsize / output_size * 0.5))
        if shrink > 1:
            rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
            img = img.resize(rsize, PIL.Image.ANTIALIAS)
            pair_img = pair_img.resize(rsize, PIL.Image.NEAREST)
            quad /= shrink
            qsize /= shrink

        # Crop.
        border = max(int(np.rint(qsize * 0.1)), 3)
        crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
        if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
            img = img.crop(crop)
            pair_img = pair_img.crop(crop)
            quad -= crop[0:2]

        mask_in = PIL.Image.fromarray(255*np.ones_like(np.array(pair_img)).astype(np.uint8))
        # Pad.
        pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
        pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
        if enable_padding and max(pad) > border - 4:
            pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
            img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'constant', constant_values=(0))
            pair_img = np.pad(np.float32(pair_img), ((pad[1], pad[3]), (pad[0], pad[2])), 'constant', constant_values=(0))
            mask_in = np.pad(np.float32(mask_in), ((pad[1], pad[3]), (pad[0], pad[2])), 'constant', constant_values=(0))

            # blur stuff
            # h, w, _ = img.shape
            # y, x, _ = np.ogrid[:h, :w, :1]
            # mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
            # blur = qsize * 0.02
            # img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
            # img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
            img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
            # mask_in[mask_in<0] = 0
            # mask_in[mask_in!=0] = 255
            # mask_in = PIL.Image.fromarray(mask_in.astype(np.uint8))
            mask_in = PIL.Image.fromarray(np.uint8(np.clip(np.rint(mask_in), 0, 255)))
            pair_img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(pair_img), 0, 255)))
            quad += pad[:2]

        # Transform.
        img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
        mask_in = mask_in.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.NEAREST)
        pair_img = pair_img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.NEAREST)


        if output_size < transform_size:
            img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
            mask_in = mask_in.resize((output_size, output_size), PIL.Image.NEAREST)
            pair_img = pair_img.resize((output_size, output_size), PIL.Image.NEAREST)

        # Save aligned image.
        if dst_file != '':
            img.save(dst_file, 'PNG')
            mask_in.save(dst_file[:-4]+'_maskin.png', 'PNG')
            pair_img.save(pair_out, 'PNG')
        return img
# def image_align_inv(src_file, dst_file, face_landmarks, output_size=1024, transform_size=4096, enable_padding=True):
#         # Align function from FFHQ dataset pre-processing step
#         # https://github.com/NVlabs/ffhq-dataset/blob/master/download_ffhq.py
#
#         lm = np.array(face_landmarks)
#         lm_chin          = lm[0  : 17]  # left-right
#         lm_eyebrow_left  = lm[17 : 22]  # left-right
#         lm_eyebrow_right = lm[22 : 27]  # left-right
#         lm_nose          = lm[27 : 31]  # top-down
#         lm_nostrils      = lm[31 : 36]  # top-down
#         lm_eye_left      = lm[36 : 42]  # left-clockwise
#         lm_eye_right     = lm[42 : 48]  # left-clockwise
#         lm_mouth_outer   = lm[48 : 60]  # left-clockwise
#         lm_mouth_inner   = lm[60 : 68]  # left-clockwise
#
#         # Calculate auxiliary vectors.
#         eye_left     = np.mean(lm_eye_left, axis=0)
#         eye_right    = np.mean(lm_eye_right, axis=0)
#         eye_avg      = (eye_left + eye_right) * 0.5
#         eye_to_eye   = eye_right - eye_left
#         mouth_left   = lm_mouth_outer[0]
#         mouth_right  = lm_mouth_outer[6]
#         mouth_avg    = (mouth_left + mouth_right) * 0.5
#         eye_to_mouth = mouth_avg - eye_avg
#
#         # Choose oriented crop rectangle.
#         x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
#         x /= np.hypot(*x)
#         x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
#         y = np.flipud(x) * [-1, 1]
#         c = eye_avg + eye_to_mouth * 0.1
#         quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
#         qsize = np.hypot(*x) * 2
#
#         # Load in-the-wild image.
#         if not os.path.isfile(src_file):
#             print('\nCannot find source image. Please run "--wilds" before "--align".')
#             return
#         img = PIL.Image.open(src_file)
#         print('quad',quad)
#         print('qsize',qsize)
#         print('img',img.size)
#         print('0 start ------------')
#         # Shrink.
#         shrink = int(np.floor(qsize / output_size * 0.5))
#         if shrink > 1:
#             rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
#             img = img.resize(rsize, PIL.Image.ANTIALIAS)
#             quad /= shrink
#             qsize /= shrink
#             print('rsize',rsize)
#         if dst_file != '':
#             img.save(dst_file.split('.png')[0]+'_1Shrink1.png', 'PNG')
#         print('quad',quad)
#         print('qsize',qsize)
#         print('img',img.size)
#         print('1 Shrink1 ------------')
#         # Crop.
#         border = max(int(np.rint(qsize * 0.1)), 3)
#         crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
#         crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
#         if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
#             img = img.crop(crop)
#             quad -= crop[0:2]
#         if dst_file != '':
#             img.save(dst_file.split('.png')[0]+'_2crop2.png', 'PNG')
#         print('quad',quad)
#         print('qsize',qsize)
#         print('img',img.size)
#         print('2 crop ------------')
#
#         # # Pad.
#         # pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
#         # pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
#         # if enable_padding and max(pad) > border - 4:
#         #     pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
#         #     img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
#         #     h, w, _ = img.shape
#         #     y, x, _ = np.ogrid[:h, :w, :1]
#         #     mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
#         #     blur = qsize * 0.02
#         #     img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
#         #     img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
#         #     img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
#         #     quad += pad[:2]
#         # if dst_file != '':
#         #     img.save(dst_file.split('.png')[0]+'_3Pad3.png', 'PNG')
#         # print('quad',quad)
#         # print('qsize',qsize)
#         # print('img',img.size)
#         # print('3 Pad ------------')
#
#         # Transform.
#         img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
#         if dst_file != '':
#             img.save(dst_file.split('.png')[0]+'_4Transform4.png', 'PNG')
#         print('quad',quad)
#         print('qsize',qsize)
#         print('img',img.size)
#         print('4 Transform ------------')
#
#         if output_size < transform_size:
#             img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
#
#         # Save aligned image.
#         if dst_file != '':
#             img.save(dst_file, 'PNG')
#         print('=========================================')
#         return img
