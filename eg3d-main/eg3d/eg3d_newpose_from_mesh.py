
import argparse
# import math
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import json
import pickle
import numpy as np
from PIL import Image
import sys
sys.path.append('../../')
from helper import create_img_torch,get_mask
import torch

import re
from typing import List, Optional, Tuple, Union
import click
import dnnlib
import legacy
import imageio
import scipy.interpolate
import mrcfile
from camera_utils import LookAtPoseSampler
# from torch_utils import misc

import pytorch3d
import skimage
from plyfile import PlyData, PlyElement, PlyProperty
from tqdm.auto import tqdm
import shutil
import time
import kornia

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer.blending import hard_rgb_blend, BlendParams

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    OpenGLPerspectiveCameras,
    PointLights,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# # add path for demo utils functions
# import sys
# sys.path.append(os.path.abspath(''))

## from mesh
def create_samples(N=256, voxel_origin=[0, 0, 0], cube_length=2.0):
    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = np.array(voxel_origin) - cube_length/2
    voxel_size = cube_length / (N - 1)

    overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(N ** 3, 3)

    # transform first 3 columns
    # to be the x, y, z index
    samples[:, 2] = overall_index % N
    samples[:, 1] = (overall_index.float() / N) % N
    samples[:, 0] = ((overall_index.float() / N) / N) % N

    # transform first 3 columns
    # to be the x, y, z coordinate
    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    num_samples = N ** 3

    return samples.unsqueeze(0), voxel_origin, voxel_size

def clean_junk_file(outdir,gpu_id):
    os.remove(f"{outdir}/barebone{gpu_id}.ply")
    os.remove(f"{outdir}/rgb{gpu_id}.npy")
    os.remove(f"{outdir}/rgb{gpu_id}_intrinsic.npy")
    os.remove(f"{outdir}/rgb{gpu_id}_pose.npy")

def gen_mesh_from_latent(G, outdir: str,name, latent, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1), num_keyframes=None, wraps=2, psi=1, truncation_cutoff=14, cfg='FFHQ', image_mode='image', gen_shapes=False, camera=None,gpu_id=0,rgb_dir='', device=torch.device('cuda'), **video_kwargs):
    os.makedirs(outdir, exist_ok=True)

    generate_mesh(G, latent, outdir=outdir, outname=f"barebone{gpu_id}", device=device)

    # raw / style_gan_2
    generate_rgbmap(G, latent, outdir=outdir, outname=f"rgb{gpu_id}", device=device, camera=camera,from_latent=0,rgb_dir=rgb_dir)

    project_mesh_color(outdir, f"barebone{gpu_id}.ply", f"rgb{gpu_id}.npy", f"rgb{gpu_id}_intrinsic.npy", f"rgb{gpu_id}_pose.npy", "{}.ply".format(name))

    # project_mesh_color(outdir, "barebone.ply", "rgb.npy", "rgb_intrinsic.npy", "rgb_pose.npy", "mesh.ply")
    clean_junk_file(outdir,gpu_id)

def generate_mesh(G, latent, outdir="", outname="", device="cuda"):
    # PURE: cleaner version of mesh generation step
    print("Creating barebone mesh...")
    # configuration
    MAX_BATCH = 100000
    VOXEL_RESOLUTION = 512
    PSI = 1

    samples, voxel_origin, voxel_size = create_samples(N=VOXEL_RESOLUTION, voxel_origin=[0, 0, 0], cube_length=G.rendering_kwargs['box_warp'])
    samples = samples.to(device)
    sigmas = torch.zeros((samples.shape[0], samples.shape[1], 1), device=device)
    transformed_ray_directions_expanded = torch.zeros((samples.shape[0], MAX_BATCH, 3), device=device)
    transformed_ray_directions_expanded[..., -1] = -1
    head = 0
    with tqdm(total = samples.shape[1]) as pbar:
        with torch.no_grad():
            while head < samples.shape[1]:
                torch.manual_seed(0)
                sigma = G.sample_mixed(samples[:, head:head+MAX_BATCH], transformed_ray_directions_expanded[:, :samples.shape[1]-head], latent , truncation_psi=PSI, noise_mode='const')['sigma']
                sigmas[:, head:head+MAX_BATCH] = sigma
                head += MAX_BATCH
                pbar.update(MAX_BATCH)

    sigmas = sigmas.reshape((VOXEL_RESOLUTION, VOXEL_RESOLUTION, VOXEL_RESOLUTION)).cpu().numpy()
    sigmas = np.flip(sigmas, 0)

    pad = int(30 * VOXEL_RESOLUTION / 256)
    pad_top = int(38 * VOXEL_RESOLUTION / 256)
    sigmas[:pad] = 0
    sigmas[-pad:] = 0
    sigmas[:, :pad] = 0
    sigmas[:, -pad_top:] = 0
    sigmas[:, :, :pad] = 0
    sigmas[:, :, -pad:] = 0

    # generate non-color ply file
    from shape_utils import convert_sdf_samples_to_ply
    convert_sdf_samples_to_ply(np.transpose(sigmas, (2, 1, 0)), [0, 0, 0], 1, os.path.join(outdir, f'{outname}.ply'), level=10)


def generate_rgbmap(G, latent, outdir="", outname="", device="cuda",camera=None,from_latent=1,rgb_dir=''):
    # Configuration
    #APPLY TO FFHQ DATASET ONLY
    print("Creating rgb_map...")
    INTRINSIC = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=device)
    RENDER_RADIUS = 2.7
    IMAGE_MODE='image'
    LOOKAT_POINT = torch.tensor([0, 0, 0.2], device=device)

    ### camera pose
    if camera is None:
        # ## gen
        cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, LOOKAT_POINT, radius=RENDER_RADIUS, device=device)
        c = torch.cat([cam2world_pose.reshape(-1, 16), INTRINSIC.reshape(-1, 9)], 1)
    else:
        c = torch.from_numpy(np.array([camera[0][1]])).float().to(device) # torch.Size([1, 25])
        cam2world_pose = c[:,:16].reshape(-1, 4,4)
    ### rgb
    if from_latent == 1:
        ## gen
        print('gen')
        img = G.synthesis(ws=latent, c=c[0:1], noise_mode='const')[IMAGE_MODE][0]
    else:
        print('input_filename',rgb_dir)
        img = create_img_torch(Image.open(rgb_dir),is_pil=1,size = 512,device=device)
        img = img.squeeze()

    # save output
    rgb_img = img.detach().cpu().permute(1,2,0).numpy()

    np.save(outdir + f'/{outname}.npy', rgb_img)
    np.save(outdir + f'/{outname}_pose.npy', cam2world_pose.cpu().numpy())
    np.save(outdir + f'/{outname}_intrinsic.npy', INTRINSIC.cpu().numpy())


def pose2opengl(poses):
    cvt = np.array([
        [0, 0, -1],
        [0, 1, 0],
        [1, 0, 0]]
    ).astype(np.float32)
    poses = (cvt @ poses.T).T
    return poses

def project_mesh_color(working_dir, mesh_ply, color_npy, intrinsic_npy, pose_npy, output_ply):
    # Configuration
    # from plyfile import PlyData, PlyElement, PlyProperty
    MESH_BOUND = 512
    print("Projecting rgbmap to barebone mesh...")

    #load numpy
    rgbs = np.load(f"{working_dir}/{color_npy}")
    intrinsic = np.load(f"{working_dir}/{intrinsic_npy}")
    pose = np.load(f"{working_dir}/{pose_npy}")

    # read mesh file
    plydata = PlyData.read(f"{working_dir}/{mesh_ply}")
    num_vertex = len(plydata['vertex'].data)
    vertices = np.zeros((num_vertex, 3), dtype=np.float32)
    for i,k in enumerate(['x','y','z']):
       vertices[:,i] = plydata['vertex'][k].astype(np.float32)

    # rescale from [-0.5, 0.5] (FFHQ Setting)
    vertices = vertices / MESH_BOUND  #scale to [0,1]
    vertices = vertices - 0.5 # scale to -0.5, 0.5
    vertices = pose2opengl(vertices)


    # convert to pytorch format
    rgbs = torch.from_numpy(rgbs).permute(2,0,1)[None] # NCHW
    vertices = torch.from_numpy(vertices) #[vertices,3]
    pose = torch.from_numpy(pose).squeeze() #4,4
    pose = torch.linalg.inv(pose) #convert from cam2world to world2cam
    intrinsic = torch.from_numpy(intrinsic).squeeze() #3,3

    # move point in 3d space to camera coordinate
    vertices_cam = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1) #Ray,4
    vertices_cam = pose @ vertices_cam.permute(1,0)
    vertices_cam = vertices_cam.permute(1,0)
    vertices_cam = vertices_cam[..., :3] / vertices_cam[...,3:4].expand(-1,3)

    # project 3d to 2d pixel
    vertices_pixel = intrinsic @ vertices_cam.permute(1,0)
    vertices_pixel = vertices_pixel.permute(1,0)
    vertices_pixel = vertices_pixel[...,:2] / vertices_pixel[...,2:3].expand(-1,2) #scale [0-1]
    vertices_pixel = (vertices_pixel * 2.0) - 1.0 #scale [-1.0, 1.0]


    # sample color from all image
    sampled_color = torch.nn.functional.grid_sample(rgbs, vertices_pixel[None,None], align_corners=True) #[1, 3, 121, 1037122, 1]
    sampled_color = sampled_color.squeeze().permute(1,0) #[1037122,  3]

    #rescale color from [-1,1] to [0,255]
    sampled_color = torch.clamp(sampled_color, -1.0, 1.0)
    sampled_color = ((sampled_color + 1.0) / 2.0) * 255.0

    # back from pytorch to numpy
    vertex = vertices.numpy()
    color = sampled_color.numpy().astype(np.uint8)

    # save ply file
    cmb_vertex = np.empty(len(vertex), dtype=[
        ('x', 'f4'),
        ('y', 'f4'),
        ('z', 'f4'),
        ('red', 'u1'),
        ('green', 'u1'),
        ('blue', 'u1'),
    ])
    cmb_vertex['x'] = vertex[:,0]
    cmb_vertex['y'] = vertex[:,1]
    cmb_vertex['z'] = vertex[:,2]
    cmb_vertex['red'] = color[:,0]
    cmb_vertex['green'] = color[:,1]
    cmb_vertex['blue'] = color[:,2]

    output_plydata = PlyData(
        [
            PlyElement.describe(cmb_vertex, 'vertex'),
            plydata['face']
        ] #, text=True
    )
    output_plydata.write(f"{working_dir}/{output_ply}")
    print(f"saved ply file to: {working_dir}/{output_ply}" )

    # np.save(working_dir+'/'+output_ply[:-4] + '_vertex.npy', vertex)
    # np.save(working_dir+'/'+output_ply[:-4] + '_color.npy', color)
    # print('vertex',vertex.shape)
    # print('color',color.shape)
    print(working_dir+'/'+output_ply[:-4] + '_vertex.npy')


#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.
    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    if m := re.match(r'^(\d+)[x,](\d+)$', s):
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------
## from obj

class SimpleShader(torch.nn.Module):
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = blend_params if blend_params is not None else BlendParams()

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        texels = meshes.sample_textures(fragments)
        images = hard_rgb_blend(texels, fragments, blend_params)
        return images  # (N, H, W, 3) RGBA image

def ed3g2torch3d(c2w):
    w2c = torch.linalg.inv(c2w)
    R = w2c[:,:3,:3]
    T = w2c[:,:3,3]
    R = R.transpose(2,1)
    R[:,:,0:2] *= -1
    T[:,0:2] *=-1
    return R,T

def get_vanila_mesh(vertices, faces):
    """
    Create vanila mesh (mesh without texture) to view
    """
    color = torch.ones_like(vertices) * 0.5
    tex = TexturesVertex(verts_features=color)
    mesh = Meshes(verts=vertices, faces=faces, textures = tex).to(vertices.device)
    return mesh

def load_ply(ply_path):
    plydata = PlyData.read(ply_path)
    num_vertex = len(plydata['vertex'].data)
    vertices = np.zeros((num_vertex, 3), dtype=np.float32)
    for i,k in enumerate(['x','y','z']):
       vertices[:,i] = plydata['vertex'][k].astype(np.float32)
    faces = plydata['face'].data['vertex_indices']
    faces = np.concatenate(faces)
    #convert to pytorch
    vertices = torch.from_numpy(vertices)[None]
    faces = torch.from_numpy(faces).view(-1,3)[None]
    return vertices, faces

def get_visible_faces(mesh, R, T, fov, image_size, mask=None):
    cameras = FoVPerspectiveCameras(device=mesh.device, R=R, T=T, fov=fov, degrees=False)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    fragments = rasterizer(mesh)
    pix_to_face = fragments.pix_to_face
    visible_faces = pix_to_face
    if mask is not None:
        negative = torch.ones_like(visible_faces) * -1
        visible_faces = visible_faces * mask + (1.0 - mask) * negative
    return visible_faces

def render_mesh(mesh, R,T, fov, image_size):
    device = mesh.device
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov, degrees=False)
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    )
    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=SimpleShader(
            device=device,
        )
    )
    return renderer(mesh)

def compute_uv(vertices, intrinsic, c2w, image_size=512):
    num_vertex = vertices.shape[0]
    # prepare variable
    w2c = torch.linalg.inv(c2w).squeeze() #stand for world to coordinate
    intrinsic = intrinsic.squeeze()
    vertices = vertices.squeeze()

    # change from world coordinate to camera coordinate
    vertices_pad = torch.cat([vertices, torch.ones_like(vertices[:,:1])] , dim=-1)
    vertices_cam = (w2c[None].expand(num_vertex, -1, -1) @ vertices_pad[...,None]).squeeze()
    vertices_cam = vertices_cam[..., :3] / vertices_cam[..., 3:].expand(-1,3)

    # projected to 2d coordinate
    vertices_uv = (intrinsic[None].expand(num_vertex, -1, -1) @ vertices_cam[...,None]).squeeze()
    vertices_uv = vertices_uv[..., :2] / vertices_uv[..., 2:].expand(-1,2)

    # shift by half pixel to match align border
    vertices_uv = (vertices_uv * (image_size / (image_size - 1))) - (0.5 / image_size)

    # if lower than 0 or more than 1, we clamp to border
    vertices_uv = torch.clamp(vertices_uv, 0.0, 1.0)
    # make y up instead of y down to match OpenGL obj format
    vertices_uv[:,1] = 1.0 - vertices_uv[:,1]
    return vertices_uv


def create_obj(output_dir, vertices, faces, intrinsic, c2w,file_name):
    """
    Create texture-map obj to
    """
    vertices_uv = compute_uv(vertices, intrinsic, c2w)
    vertices = vertices.squeeze()
    faces = faces.squeeze()
    with open(f"{output_dir}/{file_name}.obj", "w") as f:
        f.write(f"\n\nmtllib {file_name}.mtl\n\n")
        print("writing vertices...")
        for v in tqdm(vertices):
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        print("writing uv...")
        for uv in tqdm(vertices_uv):
            f.write(f"vt {uv[0]} {uv[1]}\n")
        f.write("usemtl material_face\n")
        print("writing faces...")
        for face in tqdm(faces):
            a, b, c = (face+1)
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

def create_mtl(output_dir,file_name):
    with open(f"{output_dir}/{file_name}.mtl","w") as f:
        f.write("newmtl material_face\n")
        f.write(f"map_Kd {file_name}.png\n")
        f.write("Ka 1.000 1.000 1.000 # white\n")
        f.write("Kd 1.000 1.000 1.000  # white\n")
        f.write("Ks 0.000 0.000 0.000  # black\n")
        f.write("Ns 10.0\n")


def parse_pose(pose,input_type='str'):
    """
    Parse string pose to fov/R/T
    """
    if input_type == 'str':
        pose = pose.split(',')
    if len(pose) != 25:
        raise Exception("Require pose to have 25 input paramters")
    poses = []
    for p in pose:
        poses.append(float(p))
    # c2w: camera2world extrainsic
    c2w = torch.tensor(poses[:16]).view(1,4,4)
    K = torch.tensor(poses[16:]).view(3,3)
    # fov = 2 * np.arctan(1 / (2 * K[0,0]))
    fov = 2 * np.arctan(1 / (2 * (K[0,0])))
    R,T = ed3g2torch3d(c2w)

    assert fov > 0.0 and fov < np.pi  # expected fov in this range
    assert len(R.shape) == 3 and R.shape[0] == 1 and R.shape[1] == 3 and R.shape[2] == 3
    assert len(T.shape) == 2 and T.shape[0] == 1 and T.shape[1] == 3
    return fov, R, T, K, c2w



def eg3d_newpose_from_mesh(args):
    shuffle_seed=None
    truncation_psi = 0.7
    truncation_cutoff=14
    grid=(1,1)
    num_keyframes=None
    w_frames=120

    reload_modules=False
    cfg='FFHQ'
    image_mode='image'
    sampling_multiplier=2
    nrr=None
    shapes=False
    interpolate=True
    device = torch.device('cuda')
    gpu_id = 0

    start_all_time = time.time()

    outdir_mesh =  f'{args.input_dir}/eg3d_warp/dummy/{args.input_name}-mesh/'
    os.makedirs(outdir_mesh, exist_ok=True)
    outdir_obj =  f'{args.input_dir}/eg3d_warp/dummy/{args.input_name}-bigfile/'
    os.makedirs(outdir_obj, exist_ok=True)

    print('>> create .ply')

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
    G.requires_grad=False
    G.rendering_kwargs['depth_resolution'] = int(G.rendering_kwargs['depth_resolution'] * sampling_multiplier)
    G.rendering_kwargs['depth_resolution_importance'] = int(G.rendering_kwargs['depth_resolution_importance'] * sampling_multiplier)
    if nrr is not None: G.neural_rendering_resolution = nrr

    if truncation_cutoff == 0:
        truncation_psi = 1.0 # truncation cutoff of 0 means no truncation anyways
    if truncation_psi == 1.0:
        truncation_cutoff = 14 # no truncation so doesn't matter where we cutoff

    rgb_png_path = f'{args.input_dir}/crop/{args.input_name}.png'
    f = open(f'{args.input_dir}/camera/{args.input_name}.json')
    input_camera = json.load(f)['labels']

    gen_mesh_from_latent(G=G, latent=latent,camera=input_camera,gpu_id=gpu_id, outdir=outdir_mesh,name=args.input_name,rgb_dir=rgb_png_path, bitrate='10M', grid_dims=(1,1), num_keyframes=num_keyframes, w_frames=w_frames, shuffle_seed=shuffle_seed, psi=truncation_psi, truncation_cutoff=truncation_cutoff, cfg=cfg, image_mode=image_mode,device = device)

    print('>> create .obj')
    print("load ply...")
    rgb_pose_input = np.array(input_camera[0][1])
    vertices, faces = load_ply(f"{outdir_mesh}/{args.input_name}.ply")
    fov, R,T, gl_k, gl_c2w = parse_pose(rgb_pose_input,input_type='list')

    print("create obj file...")
    create_mtl(f"{outdir_obj}",args.input_name)
    create_obj(f"{outdir_obj}", vertices, faces, gl_k, gl_c2w,args.input_name)
    shutil.copy2(f"{rgb_png_path}", f"{outdir_obj}/{args.input_name}.png")


    print('>> create warp')
    f = open(f'{args.input_dir}/camera/{args.target_name}.json')
    target_camera = json.load(f)['labels']
    rgb_pose_target = np.array(target_camera[0][1])
    vertices = vertices.to(device)
    faces = faces.to(device)

    print("loading vanila mesh...")
    start_time = time.time()
    # get vanila mesh without texture to get
    vanila_mesh = get_vanila_mesh(vertices, faces)

    # get mask
    mask = skimage.io.imread(f'{args.input_dir}/crop_mask/{args.input_name}.png') # mask 0-255
    mask = get_mask(mask,args.mask_type) * 255
    mask = skimage.img_as_float32(mask) # mask 0-1 (float)
    mask = np.squeeze(mask)
    if len(mask.shape) > 2:
        mask = skimage.color.rgb2gray(mask[...,:3])
        mask = np.squeeze(mask)
    mask = skimage.transform.resize(mask, (8192, 8192), anti_aliasing=False)
    mask = torch.from_numpy(mask).to(device)
    mask = torch.round(mask)
    mask = mask[None,:,:,None]

    print("getting viisble faces...")
    visible_faces = get_visible_faces(vanila_mesh, R, T, fov, image_size=8192, mask=mask).unique()

    print(f"visible faces in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    print("load texturemap mesh...")
    mesh = load_objs_as_meshes([f"{outdir_obj}/{args.input_name}.obj"]).to(device)
    print(f"load texturemap mesh in {time.time() - start_time:.2f} seconds")


    start_time = time.time()
    fov, R,T, gl_k, gl_c2w = parse_pose(rgb_pose_target,input_type='list')
    image_output = render_mesh(mesh, R,T, fov, image_size=512)
    image_visible = get_visible_faces(vanila_mesh, R, T, fov, image_size=512) #render vanila face instead because mesh and vanila mesh has differnt face ids.
    mask = torch.isin(image_visible,visible_faces)
    not_surface = image_output[...,3:4].bool()
    mask = torch.logical_and(mask, not_surface)
    # remove small noise
    opening_kernel = torch.ones(5,5).to(device)
    mask = mask[...,0][None]
    mask = kornia.morphology.closing(mask.float(), opening_kernel)
    mask = mask[0][...,None]

    skimage.io.imsave(f'{args.input_dir}/eg3d_warp/{args.input_name}_{args.target_name}_mask{args.mask_type}_mask.png', (mask.squeeze().cpu().numpy()*255).astype(np.uint8))
    skimage.io.imsave(f'{args.input_dir}/eg3d_warp/{args.input_name}_{args.target_name}_mask{args.mask_type}_render.png', (image_output[0,:,:,:3].cpu().numpy()*255).astype(np.uint8))

    print(f"render mesh in {time.time() - start_time:.2f} seconds")
    print(f"total rendering time {time.time() - start_all_time:.2f} seconds")

    print('done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='10036')
    parser.add_argument('--target_name', type=str, default='59641')

    parser.add_argument('--mask_type', type=str, default='face') # 'face/hair'


    parser.add_argument("--input_dir", type=str, default="../../results/preprocessed/10036_59641/")

    args = parser.parse_args()

    eg3d_newpose_from_mesh(args)
