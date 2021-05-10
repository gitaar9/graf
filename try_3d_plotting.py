import argparse
import os
from os import path
import numpy as np
import time
import copy
import csv
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
from torchvision.utils import save_image
import torch.nn.functional as F
from functools import partial

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
# import ssl          # enable if downloading models gives CERTIFICATE_VERIFY_FAILED error
# ssl._create_default_https_context = ssl._create_unverified_context

import sys
sys.path.append('submodules')        # needed to make imports work in GAN_stability

from graf.gan_training import Evaluator as Evaluator
from graf.config import get_data, build_models, update_config, get_render_poses
from graf.utils import count_trainable_parameters, to_phi, to_theta, get_nsamples, polar_to_cartesian, look_at
from graf.transforms import ImgToPatch

from submodules.GAN_stability.gan_training.checkpoints import CheckpointIO
from submodules.GAN_stability.gan_training.distributions import get_ydist, get_zdist
from submodules.GAN_stability.gan_training.config import (
    load_config,
)

from external.colmap.filter_points import filter_ply


def uniformly_sample_3d_points(sample_per_axis, show_points=False):
    import itertools
    x_ = np.linspace(-3., 3., sample_per_axis)
    y_ = np.linspace(-3., 3., sample_per_axis)
    z_ = np.linspace(-3., 3., sample_per_axis)

    # x_ = np.linspace(-2., 2., sample_per_axis)
    # y_ = np.linspace(-3., 3., sample_per_axis)
    # z_ = np.linspace(-1., 3., sample_per_axis)

    coordinates_list = list(itertools.product(x_, y_, z_))

    if show_points:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(*list(zip(*coordinates_list)), marker='o')
        plt.show()

    return coordinates_list


def get_array_with_single_pose(render_radius):
    # Get a single pose
    # Car facing right:
    theta = to_theta(.45642212862617093 / 2)  # +- 45 degree angle
    phi = to_phi(1 / 2)
    # Car facing left
    theta = to_theta(.45642212862617093 / 2)  # +- 45 degree angle
    phi = to_phi(1)
    # car facing us, directly from front no angle for theta
    theta = to_theta(.45642212862617093)
    phi = to_phi(.75)

    loc = polar_to_cartesian(render_radius, phi, theta, deg=True)
    R = look_at(loc)[0]
    RT = np.concatenate([R, loc.reshape(3, 1)], axis=1)
    poses = torch.from_numpy(np.stack([RT]))

    return poses


def query_points_from_nerf(generator, coordinates_list, single_z, view_dir, device, amount_of_points):
    # Duplicate the z for all the points we will query later
    z = single_z.cpu().detach().numpy()
    z = np.asarray([list(z[0])] * amount_of_points)
    z = torch.from_numpy(z).float().to(device)

    # For every point copy the same viewing direction as a shortcut
    viewdirs = torch.from_numpy(np.asarray([view_dir] * len(coordinates_list))).float().to(device)
    # Create tensors from the sampled points
    coordinates = torch.from_numpy(np.asarray([[c] for c in coordinates_list])).float().to(device)

    network_query_fn = generator.render_kwargs_test['network_query_fn']
    network_fn = generator.render_kwargs_test['network_fn']
    raw = network_query_fn(coordinates, viewdirs=viewdirs, network_fn=network_fn, features=z)
    # This is here to show what network_query_fn actually is:
    # network_query_fn = lambda inputs, viewdirs, network_fn, features: run_network(inputs, viewdirs, network_fn,
    #                                                                               features=features,
    #                                                                               embed_fn=embed_fn,
    #                                                                               embeddirs_fn=embeddirs_fn,
    #                                                                               netchunk=args.netchunk,
    #                                                                               feat_dim_appearance=args.
    #                                                                               feat_dim_appearance)

    # Go from the output tensor to RGB and alpha values:
    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3] for me that means: [1, n_points, 3]
    relu = partial(F.relu, inplace=True)  # saves a lot of memory
    alpha = 1. - torch.exp(-relu(raw[..., 3]))

    return raw, rgb, alpha


def plot_points(coordinates_list, rgb, alpha):
    # need to plot the points using their alpha/rgb
    rgb = rgb.cpu().detach().numpy().squeeze()
    alpha = alpha.cpu().detach().numpy().squeeze()
    print("RGB array shape: ", rgb.shape, " Min and max: ", np.min(rgb), np.max(rgb))
    print("Alpha array shape: ", alpha.shape, " Min and max: ", np.min(alpha), np.max(alpha))
    rgba = np.concatenate((rgb, np.expand_dims(alpha, axis=1)), axis=1) # Combine rgb and alpha

    # Filter out points with to little alpha or completely white points
    zipped = [(c, (r, g, b, a)) for c, (b, g, r, a) in list(zip(coordinates_list, rgba)) if a > 0.01 and ((b + g + r) < 2.95)]
    coordinates_list, rgba = zip(*zipped)
    print("Nr of points left after filtering: ", len(coordinates_list))

    # Plot the points in 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.view_init(elev=90, azim=90)
    ax.scatter(*list(zip(*coordinates_list)), marker='.', c=list(rgba))

    limit = 3
    ax.set_xlim3d(-limit, limit)
    ax.set_ylim3d(-limit, limit)
    ax.set_zlim3d(-limit, limit)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def main():
    # Arguments
    parser = argparse.ArgumentParser(
        description='Train a GAN with different regularization strategies.'
    )
    parser.add_argument('config', type=str, help='Path to config file.')
    parser.add_argument('--pretrained', action='store_true', help='Load pretrained model.')

    args, unknown = parser.parse_known_args()
    config = load_config(args.config, 'configs/default.yaml')
    config['data']['fov'] = float(config['data']['fov'])
    config = update_config(config, unknown)

    # Short hands
    batch_size = config['training']['batch_size']
    out_dir = os.path.join(config['training']['outdir'], config['expname'])
    if args.pretrained:
        config['expname'] = '%s_%s' % (config['data']['type'], config['data']['imsize'])
        out_dir = os.path.join(config['training']['outdir'], config['expname'] + '_from_pretrained')
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

    # Get model file
    if args.pretrained:
        # config_pretrained = load_config('configs/pretrained_models.yaml', 'configs/pretrained_models.yaml')
        # print(config_pretrained['training']['outdir'])
        # exit()
        model_file = config['training']['outdir'] + "/shapenetships_128_new_plan/chkpts/model_best.pt"
        # For CARLA:
        model_file = 'https://s3.eu-central-1.amazonaws.com/avg-projects/graf/models/carla/carla_128.pt'
    else:
        model_file = 'model_best.pt'

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
    prompt = ''
    while not prompt:
        print("Building new object...")
        poses = get_array_with_single_pose(render_radius)

        # Sample a single latent space z
        single_z = zdist.sample((1,))  # shape([1, 256])

        # Create output folder and a reference image
        outpath = os.path.join('output/single_frame/')
        os.makedirs(outpath, exist_ok=True)
        evaluator.make_video(outpath, single_z, poses, as_gif=False)

        # Create some points in 3D and a ray direction for the NERF model is input
        samples_per_axis = 65
        coordinates_list = uniformly_sample_3d_points(samples_per_axis)  # Uniformly sample 3d points
        view_dir = [0,  .9, -.3]  # Pick a ray direction

        # Query the points directly to the NERF model
        raw, rgb, alpha = query_points_from_nerf(
            generator=generator_test,
            coordinates_list=coordinates_list,
            single_z=single_z,
            view_dir=view_dir,
            device=device,
            amount_of_points=samples_per_axis**3
        )

        plot_points(coordinates_list, rgb, alpha)  # Scatter the points in 3D plot
        del raw, rgb, alpha
        torch.cuda.empty_cache()

        prompt = input('Press ENTER without typing anything to generate a new model or type quit and ENTER to quit:\n')


if __name__ == '__main__':
    main()
    # If we need a single set z use this:
    # z = torch.from_numpy(np.asarray([[0.6473, -0.6994, 1.3732, 0.9141, 0.6754, 2.5580, 0.0292, 1.2643,
    #          0.3498, -0.9836, 1.0502, -0.7577, 0.6887, -0.3215, 0.3779, 0.7225,
    #          1.4827, 0.5803, 0.2477, 1.1816, -0.6831, 1.3893, -0.9371, -1.6761,
    #          -0.2751, -1.1555, -0.0146, -0.0985, -0.1656, 1.5738, 0.8722, 0.0876,
    #          0.0334, -0.6630, 0.8795, 0.9439, -0.7742, -0.8840, -0.2141, -0.2051,
    #          -1.3055, 0.4132, 1.5080, 0.0146, 0.9725, 1.3789, -0.1919, 0.4754,
    #          0.2574, -0.0748, -0.5291, -0.2026, -1.3655, -0.1510, 1.4367, 0.4001,
    #          0.3484, -0.8528, -0.8202, 0.3092, -0.1040, -0.2242, 1.0259, -2.5109,
    #          0.2410, -0.2397, 1.0717, -0.9201, -0.4242, 0.4183, 1.1752, -0.2251,
    #          1.1756, -1.1862, -1.3131, -0.4298, 1.2245, -2.0602, 1.7400, -0.7422,
    #          1.7924, -0.4478, 1.2036, -0.6995, -0.3870, -0.7308, 1.3977, -1.3814,
    #          1.3799, 0.2244, 1.2534, -0.2502, 1.2443, 1.0667, 1.2183, -0.2516,
    #          0.3078, 0.4481, 1.1296, 0.2433, -0.1642, 0.1139, -1.7637, -0.0813,
    #          -0.5017, 2.3660, -0.0286, 0.1010, -0.6676, -1.0064, 2.0285, -0.9095,
    #          1.6474, -0.2595, -0.2735, 1.0353, -1.1805, -0.1019, 0.8218, 2.5521,
    #          0.2640, 0.3857, 0.4281, -0.7893, -1.9387, -0.0162, -0.7568, 1.5606,
    #          0.4136, 0.2299, -1.4850, -1.5502, -1.2177, -0.0410, 0.2524, 1.5629,
    #          0.3325, -1.4577, -0.0415, 0.4440, -1.1150, -0.1878, -0.9375, -0.4563,
    #          0.4534, 0.8871, -0.6633, -0.7545, -1.2928, 0.0648, 0.4324, -0.0198,
    #          -1.1646, -0.3557, -1.0400, -0.0158, 1.3280, -1.5007, -1.6266, 0.0803,
    #          -0.1404, 0.7192, 1.3333, 1.4521, 0.5018, -0.0782, 0.3346, 0.2395,
    #          -0.2833, 0.0956, -0.9275, 0.9586, 0.2396, 1.3134, 0.0260, -0.2533,
    #          -0.9337, 0.0276, -1.3675, 1.4202, -0.9930, -0.2948, -0.6257, 0.9259,
    #          0.3337, 0.2854, -0.0324, 0.5147, -0.0588, -1.5435, -0.9469, -0.5754,
    #          -1.2698, -0.4550, 1.5144, 0.9072, -1.3557, 0.4247, 1.1686, -0.2056,
    #          -0.3484, -0.9065, 0.3156, -0.3749, 1.6489, -0.6705, -0.3112, 0.7619,
    #          0.3001, 0.5082, 0.1201, -0.7158, 0.1729, -0.5704, 0.3301, 0.4497,
    #          -1.6424, 0.0290, -2.0277, 0.3778, -1.4347, -0.7366, 0.1425, -0.1132,
    #          -1.6011, -1.9934, 0.3446, -1.1522, 1.4772, -0.2376, 0.3583, 1.1927,
    #          0.0438, -0.0784, 0.5175, 1.8108, -0.8586, -1.8894, 0.4743, 0.1858,
    #          -0.8160, 0.1594, 1.4548, 0.8908, -0.2225, 1.2897, 1.5729, 0.5333,
    #          -2.5373, -0.5130, -0.1839, -0.2978, -0.7178, -0.9236, -0.2874, 1.0573]] * 27000)).float().to(device) # 27000
