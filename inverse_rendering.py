import argparse
import copy
from os import path
from pathlib import Path

import torch

torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
# import ssl          # enable if downloading models gives CERTIFICATE_VERIFY_FAILED error
# ssl._create_default_https_context = ssl._create_unverified_context
import os
import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config
from graf.utils import count_trainable_parameters, to_phi, to_theta, polar_to_cartesian, look_at

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)
from torchvision.transforms import *
import numpy as np
from PIL import Image
import math


def create_gif(frames, output_dir):
    output_name = 'inverse_render.gif'
    img, *imgs = frames
    img.save(fp=os.path.join(f'{output_dir}', output_name), format='GIF', append_images=imgs,
             save_all=True, duration=45, loop=0, interlace=False)


def get_rays(generator, pose):
    return generator.val_ray_sampler(generator.H, generator.W, generator.focal, pose)[0]


def create_samples(generator, z, poses):
    generator.eval()

    N_samples = len(z)
    device = generator.device
    rays = torch.stack([get_rays(generator, poses[i].to(device)) for i in range(N_samples)])

    rays = rays.permute(1, 0, 2, 3).flatten(1, 2)
    rgb, _, _, _ = generator(z, rays=rays)
    reshape = lambda x: x.view(N_samples, generator.H, generator.W, x.shape[1]).permute(0, 3, 1, 2)
    rgb = reshape(rgb)
    return rgb


def tensor_to_PIL(img):
    img = img.squeeze() * 0.5 + 0.5
    return Image.fromarray(img.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy())


def get_array_with_single_pose(azimuth, elevation, render_radius=10):
    # Get a single pose
    phi = to_phi(azimuth)
    theta = to_theta(elevation)


    loc = polar_to_cartesian(render_radius, phi, theta, deg=True)
    R = look_at(loc)[0]
    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    poses = torch.from_numpy(np.stack([RT]))

    return poses


def get_ground_truth_images(img_size, image_folder, n_input_views, device):
    img_names = ['000000.png', '000001.png', '000002.png', '000003.png'][:n_input_views]
    img_paths = [os.path.join(image_folder, "rgb", name) for name in img_names]

    transforms = Compose([
        Resize(img_size),
        ToTensor(),
        Lambda(lambda x: x * 2 - 1),
    ])

    gt_images = []
    for path in img_paths:
        img = Image.open(path).convert('RGB')
        img = transforms(img).to(device).unsqueeze(0)
        gt_images.append(img)

    return torch.cat(gt_images, dim=0)


def inverse_rendering(image_path, output_dir, num_frames, n_input_views, img_size, generator, device):
    # Load one or multiple images
    gt_images = get_ground_truth_images(img_size, image_path, n_input_views, device).to(generator.device)
    tensor_to_PIL(gt_images[0]).show()
    # Inference poses:
    img_yaws = np.asarray([math.pi / 2, math.pi, -math.pi/2, -math.pi/2])
    img_yaws = list(((img_yaws + math.pi) / (math.pi * 2)) - .25)
    img_yaws = [yaw if yaw > 0 else yaw + 1. for yaw in img_yaws]
    img_pitches = [.2132118, .2132118, .2132118, .0019027]


    z = torch.zeros((1, 256)).to(generator.device)
    z_offsets = torch.zeros_like(z).to(generator.device)
    z_offsets.requires_grad_()

    optimizer = torch.optim.Adam([z_offsets], lr=1e-2, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100, gamma=0.5)

    frames = []

    for idx in range(n_input_views):
        save_image(gt_images[idx], f"{output_dir}/gt{idx}.jpg", normalize=True)

    n_iterations = 200

    for i in range(n_iterations):
        noise_z = (0.05 * torch.randn_like(z) * (n_iterations - i) / n_iterations).to(generator.device)

        all_frames = []
        for idx in range(n_input_views):
            poses = get_array_with_single_pose(azimuth=img_yaws[idx], elevation=img_pitches[idx])
            rgbs = create_samples(
                generator=generator,
                z=z + noise_z + z_offsets,
                poses=poses
            )
            all_frames.append(rgbs)
        all_frames = torch.cat(all_frames, dim=0)
        loss = torch.nn.L1Loss()(all_frames, gt_images)
        # loss = loss.mean()
        print(f"{i + 1}/{n_iterations}: {loss.item()} {scheduler.get_lr()}")

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 25 == 0:
            save_image(all_frames[0], f"{output_dir}/{i}.jpg", normalize=True)

        del rgbs
        del all_frames

        # Every 25 rounds make images of all possible gt_poses
        if i % 25 == 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    for idx in range(len(img_pitches)):
                        poses = get_array_with_single_pose(azimuth=img_yaws[idx], elevation=img_pitches[idx])
                        rgbs = create_samples(
                            generator=generator,
                            z=z + z_offsets,
                            poses=poses
                        )
                        save_image(rgbs, f"{output_dir}/{i}_{idx}.jpg", normalize=True)

        # Every round add the first gt pose to the frames
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                poses = get_array_with_single_pose(azimuth=img_yaws[0], elevation=img_pitches[0])
                rgbs = create_samples(
                    generator=generator,
                    z=z + z_offsets,
                    poses=poses
                )
                frames.append(tensor_to_PIL(rgbs))

        scheduler.step()
    create_gif(frames, output_dir)


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--image_path', type=str)
    parser.add_argument('--output_dir', type=str, default='inverse_images')
    parser.add_argument('--num_frames', type=int, default=128)
    parser.add_argument('--n_input_views', type=int, default=1)
    parser.add_argument('--use_view_lock_for_optimization', action='store_true')

    args, unknown = parser.parse_known_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    checkpoint_dir = path.join(out_dir, 'chkpts')
    eval_dir = os.path.join(out_dir, 'eval')
    os.makedirs(eval_dir, exist_ok=True)

    config['training']['nworkers'] = 0

    # Logger
    checkpoint_io = CheckpointIO(
        checkpoint_dir=checkpoint_dir
    )

    device = torch.device("cuda:0")

    # Dataset
    train_dataset, hwfr, render_poses = get_data(config)
    # in case of orthographic projection replace focal length by far-near
    if config['data']['orthographic']:
        hw_ortho = (config['data']['far'] - config['data']['near'], config['data']['far'] - config['data']['near'])
        hwfr[2] = hw_ortho

    config['data']['hwfr'] = hwfr  # add for building generator
    print(train_dataset, hwfr, render_poses.shape)

    # Create models
    generator, _ = build_models(config, disc=False)
    print('Generator params: %d' % count_trainable_parameters(generator))

    # Put models on gpu if needed
    generator = generator.to(device)

    # Register modules to checkpoint
    checkpoint_io.register_modules(
        **generator.module_dict  # treat NeRF specially
    )
    print(config['expname'])
    model_file = os.path.join(config['training']['outdir'], config['expname'], 'chkpts/model_best.pt')

    # Distributions
    ydist = get_ydist(1, device=device)  # Dummy to keep GAN training structure in tact
    y = torch.zeros(batch_size)  # Dummy to keep GAN training structure in tact
    zdist = get_zdist(config['z_dist']['type'], config['z_dist']['dim'],
                      device=device)

    # Test generator, use model average
    generator_test = copy.deepcopy(generator)
    generator_test.parameters = lambda: generator_test._parameters
    generator_test.named_parameters = lambda: generator_test._named_parameters
    checkpoint_io.register_modules(**{k + '_test': v for k, v in generator_test.module_dict.items()})

    # Evaluator
    evaluator = Evaluator(0, generator_test, zdist, ydist,
                          batch_size=batch_size, device=device)

    # Load checkpoint
    print('load %s' % os.path.join(checkpoint_dir, model_file))
    load_dict = checkpoint_io.load(model_file)

    # Get the object camera distance
    render_radius = config['data']['radius']
    if isinstance(render_radius, str):  # use maximum radius
        render_radius = float(render_radius.split(',')[1])

    # My code:
    inverse_rendering(args.image_path, args.output_dir, args.num_frames, args.n_input_views, config['data']['imsize'], generator_test, device)


if __name__ == '__main__':
    main()
