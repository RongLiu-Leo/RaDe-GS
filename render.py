#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import shutil
import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image

def render_set(dataset, name, iteration, views, gaussians, pipeline, background, kernel_size):
    model_path = dataset.model_path
    source_path = dataset.source_path

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(os.path.join(render_path, "rgb"), exist_ok=True)
    makedirs(os.path.join(render_path, "depth"), exist_ok=True)
    makedirs(os.path.join(gts_path, "rgb"), exist_ok=True)
    makedirs(os.path.join(gts_path, "depth"), exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_package = render(view, gaussians, pipeline, background, kernel_size=kernel_size)
        rendering = render_package['render']
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, "rgb", view.image_name + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, "rgb", view.image_name + ".png"))

        depth = render_package['median_depth'].squeeze(0)
        gt_depth = np.load(os.path.join(source_path, "synthetic", "raw_depths", view.image_name + "_depth.npy"))
        print( depth.shape, gt_depth.shape)
        gt_depth = np.array(
            Image.fromarray(gt_depth).resize((depth.shape[1], depth.shape[0])),
            dtype=np.float32
        )
        print( depth.shape, gt_depth.shape)
        np.save(os.path.join(gts_path, "depth", view.image_name + "_depth.npy"), gt_depth)
        np.save(os.path.join(render_path, "depth", view.image_name + "_depth.npy"), depth.cpu().numpy())
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.kernel_size)

        if not skip_test:
             render_set(dataset, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, dataset.kernel_size)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)