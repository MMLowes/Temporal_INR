import argparse
import json
import os
import imageio
import numpy as np
import matplotlib.pyplot as plt

from models import models
from utils import general


def temporal_interpolation(reg_model, image_stack, ref, save_folder, temporal_resolutions=[10, 20, 50, 100], slice_idx=[1,250], exists_ok=False, save_slices_only=False):
    
    for t_res in temporal_resolutions:
        gif_path = os.path.join(save_folder, f"temporal_interpolation_{t_res}.gif")
        if os.path.isfile(gif_path) and exists_ok:
            print(f"Skipping existing {gif_path}")
            continue
        
        print(f"Generating temporal interpolation at resolution {t_res}, saving to {gif_path}")
        temporal_image_stack = reg_model.temporal_resulution_transform(image_stack, t_res, method="direct", save_slices_only=save_slices_only)

        frame_folder = os.path.join(save_folder, f"temporal_frames_{t_res}")
        os.makedirs(frame_folder, exist_ok=True)

        spacing = ref.GetSpacing()
        size = ref.GetSize()
        xmin = 0
        xmax = spacing[0] * size[0]
        ymin = 0
        ymax = spacing[2] * size[2]
        new_min, new_max = 0.05, 1.3

        # Loop over the temporal stack and save each frame as an image
        frames = []
        for i in range(temporal_image_stack.shape[0]):
            if save_slices_only:
                frame = temporal_image_stack[i].cpu().numpy()
            else:  
                frame = temporal_image_stack[i].select(slice_idx[0],slice_idx[1]).cpu().numpy()#.transpose(1, 2, 0)
            old_min, old_max = np.min(frame), np.max(frame)
            frame = (frame - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
            frame = np.clip(frame, 0, 1)
            frame = frame ** 1.2

            frame_path = os.path.join(frame_folder, f"frame_{i:03d}.png")
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            ax.imshow(frame.T, cmap="gray", origin="lower", vmin=0, vmax=1, extent=[xmin, xmax, ymin, ymax], aspect='auto')
            ax.axis("off")  # Turn off axes for better aesthetics
            fig.patch.set_facecolor('white') # Add a white border around the plot for cleaner visuals
            fig.tight_layout()
            plt.savefig(frame_path, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            frames.append(frame_path)

        # Create a GIF from the saved frames
        
        
        frames = [imageio.imread(frame_path) for frame_path in frames]
        # frames = np.repeat(frames, 100//t_res, axis=0)
        frames = [f for f in frames]
        fps = t_res/5
        if os.path.isfile(gif_path):
            os.remove(gif_path)
        imageio.mimsave(gif_path, frames, fps=fps, loop=0)
        print(f"GIF saved at {gif_path}")

def dirlab_interpolation(dir_folder, save_folder, v, temporal_resolutions = [10, 20, 50, 100], exists_ok=False, save_slices_only=True):
    
    image_stack, mask, ref = general.load_stack_DIRLab(dir_folder, v)

    slice_idx = [1, 250 if image_stack.shape[2]==512 else 150]

    with open(os.path.join(save_folder, f"variation_{v:02d}","kwargs.json"), "r") as f:
        kwargs = json.load(f)

    reg_model = models.TemporalImplicitRegistrator(image_stack, mask, **kwargs)
    reg_model.load_network()
    save_folder_v = os.path.join(save_folder, f"variation_{v:02d}", "temporal_interpolation")
    temporal_interpolation(reg_model, image_stack, ref, save_folder_v, temporal_resolutions, slice_idx, exists_ok, save_slices_only=save_slices_only)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run temporal interpolation on DIR-LAB dataset")
    parser.add_argument("--path", type=str, required=True, help="Path to the DIR-LAB dataset")
    parser.add_argument("--save_folder", type=str, default="./results", help="Folder to save results")
    parser.add_argument("--exists_ok", action="store_true", help="Skip existing results")
    parser.add_argument("--temporal_resolutions", type=int, nargs="+", default=[20, 50, 100], help="Temporal resolutions for interpolation")
    parser.add_argument("--save_slices_only", action="store_true", help="Save only the slices instead of full volumes")

    args = parser.parse_args()
    dir_folder = args.path
    save_folder = args.save_folder
    os.makedirs(save_folder, exist_ok=True)
    
    for v in range(1, 11):
        dirlab_interpolation(dir_folder, save_folder, 8, temporal_resolutions=args.temporal_resolutions, exists_ok=args.exists_ok, save_slices_only=args.save_slices_only)


    