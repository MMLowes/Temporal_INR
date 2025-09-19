import argparse
from utils import general
from models import models

import numpy as np
import os
import json
import numpy as np
from copy import deepcopy


def main(dir_folder, variation, exist_ok=True, **kwargs):

    use_CT  = kwargs["sdf_alpha"] < 1.0
    use_sdf = kwargs["sdf_alpha"] > 0.0

    image_stack, mask, reference = general.load_stack_DIRLab(dir_folder, variation, use_CT, use_sdf)
    use_mask = kwargs.pop("mask", True)
    mask = mask if use_mask else None
    reg_model = models.TemporalImplicitRegistrator(image_stack, mask, **kwargs)
    
    if exist_ok and os.path.isfile(reg_model.opts.network_name):
        reg_model.load_network()
    else:
        reg_model.fit()
        reg_model.save_network()
    
    # plot losses
    general.plot_loss(reg_model, **reg_model.args)
    
    lm0, lm5, lms = general.load_DIRLab_landmarks(dir_folder, variation, reference)
        
    transform_methods = ["direct", "sequential", "taylor", "combined"]
    transform_means = {}
    for method in transform_methods:
        lm_scaled = general.scale_landmarks_to_1_1(lm0, reference.GetSize())
        coords_moved_f = reg_model.transform_points_over_cycle(lm_scaled, method=method)
        lm_moved = general.scale_landmarks_to_original(coords_moved_f[5], reference.GetSize())
        acc0 = general.compute_landmark_accuracy(lm_moved, lm5, reference.GetSpacing())
        general.save_landmarks(lm_moved, os.path.join(kwargs["save_folder"], "moved_landmarks", f"landmarks_moved_from_0_to_5_{method}.txt"))
        
        lm_scaled = general.scale_landmarks_to_1_1(lm5, reference.GetSize())
        coords_moved = reg_model.transform_points_over_cycle(lm_scaled, start_time=5, method=method)
        lm_moved = general.scale_landmarks_to_original(coords_moved[5], reference.GetSize())
        acc5 = general.compute_landmark_accuracy(lm_moved, lm0, reference.GetSpacing())
        transform_means[method+"_f"] = acc0[0][0]
        transform_means[method+"_b"] = acc5[0][0]
    
    lm_scaled = general.scale_landmarks_to_1_1(lms[0], reference.GetSize())
    coords_moved = reg_model.transform_points_over_cycle(lm_scaled, method="taylor")
    lm_moved = general.scale_landmarks_to_original(coords_moved, reference.GetSize())
    accs = [general.compute_landmark_accuracy(lm_moved[i], lms[i], reference.GetSpacing(), return_std=False)[0] for i in range(6)]

    return accs, transform_means

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Run temporal registration on DIR-LAB dataset")
    parser.add_argument("--path", type=str, required=True, help="Path to the DIR-LAB dataset")
    parser.add_argument("--save_folder", type=str, default="./results", help="Folder to save results")
    parser.add_argument("--sdf_alpha", type=float, default=0.05, help="SDF alpha value")
    parser.add_argument("--point_alpha", type=float, default=0.01, help="Point alpha value")
    parser.add_argument("--n_encoding_features", type=int, default=6, help="Number of encoding features")
    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--exist_ok", action="store_true", default=True, help="Overwrite existing files")

    args = parser.parse_args()
    dir_folder = args.path
    folder = args.save_folder
    os.makedirs(folder, exist_ok=True)
    exist_ok = args.exist_ok

    kwargs_orig = {
            "sdf_alpha": args.sdf_alpha,
            "point_alpha": args.point_alpha,
            "n_encoding_features": args.n_encoding_features,
            "epochs": args.n_epochs,
            "lr": args.lr,
            "cycle_length": 10,
            "scheduler": "step",
            "time_weights": [6,1,1,1,1,6,1,1,1,1]
    }
    
    t_means = {}
    print("\n")
    for variation in range(1, 11):
        kwargs = deepcopy(kwargs_orig)
        kwargs["save_folder"] = os.path.join(folder, f"variation_{variation:02d}")
        print(f"Running test - Variation {variation:02d}")
        os.makedirs(kwargs["save_folder"], exist_ok=True)
        _, transform_means = main(dir_folder, variation, exist_ok=exist_ok, **kwargs)
        t_means[variation] = transform_means
    
    # print("\n\nResults:")
    # for key, value in output.items():
    #     print(f"Variation {key}: {value[0][0][0]} ({value[1][0][0]})")
    print(f"\n\nResults from insp to exp")
    print(" "*12+", ".join([f"{method:>13}" for method in list(t_means.values())[0].keys()]))
    for key, value_mean in t_means.items():
        print(f"Subject {key:02d}:", end=" ")
        print(", ".join([f"{value:>13}" for key, value in value_mean.items()]))
        
    print(f"Means:     ", end=" ")
    print(", ".join([f"{np.mean([value[key] for value in t_means.values()]).round(2):>10}" for key in list(t_means.values())[0].keys()]))

    # the value taylor_f is the one used in the paper
    with open(os.path.join(folder, f"results_taylor_{np.mean([value['taylor_f'] for value in t_means.values()]).round(2)}.json"), "w") as f:
                json.dump(t_means, f)