# stylegan-salon

## Preparing datasets
Please download and add these models to 'checkpoints' folder:
face detection model from face-parsing.PyTorch-master
https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view

ffhqrebalanced512-64.pkl from eg3d
https://github.com/NVlabs/eg3d/blob/main/docs/models.md

stylegan2-ffhq-config-f.pt from stylegan2
https://github.com/rosinality/stylegan2-pytorch/blob/master/README.md

we use eg3d preprocessing in our preprocessing step, please follow "Preparing datasets" installation in:
https://github.com/NVlabs/eg3d.
Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) submodule is properly initialized.

## Getting Started
Please add your input images to 'inputs' folder.
Run the following commands
1. Eg3d preprocessing
```.bash
cd eg3d-main/dataset_preprocessing/ffhq
python salon_eg3d_preprocessing.py
```
2. StyleganSalon preprocessing
```.bash
cd ../../..
python salon_eg3d_preprocessing.py
```
3. Run hairstyle transfer
```.bash
python salon_main.py
```

## Acknowledgments
This code borrows heavily from
stylegan2: https://github.com/rosinality/stylegan2-pytorch
eg3d: https://github.com/NVlabs/eg3d


## BibTeX
```
@inproceedings{Khwanmuang2023StyleGANSalon,
    author = {Khwanmuang, Sasikarn and Phongthawee, Pakkapon and Sangkloy, Patsorn and Suwajanakorn, Supasorn},
    title = {StyleGAN Salon: Multi-View Latent Optimization for Pose-Invariant Hairstyle Transfer},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2023},
  }
```
