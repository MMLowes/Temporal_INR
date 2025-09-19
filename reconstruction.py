import argparse
from utils import general
from models import models

import numpy as np
import os
import json
import torch
import numpy as np
from collections import defaultdict
from copy import deepcopy

def main(dir_folder, variation, exist_ok=True, **kwargs):
    
   
    use_CT  = kwargs["sdf_alpha"] < 1.0
    use_sdf = kwargs["sdf_alpha"] > 0.0
    
    gt_stack, mask, reference = general.load_stack_DIRLab(dir_folder, variation, use_CT, use_sdf)
    
    # ensure no data spillage
    image_stack = gt_stack.clone()
    recon_ids = torch.argwhere(torch.tensor(kwargs["time_weights"])==0).flatten()
    for idx in recon_ids:
        image_stack[idx] = torch.zeros_like(image_stack[idx])
    
    use_mask = kwargs.pop("mask", True)
    mask = mask if use_mask else None
    reg_model = models.TemporalImplicitRegistrator(image_stack, mask, **kwargs)
    # print(kwargs)
    
    if exist_ok and os.path.isfile(reg_model.opts.network_name):
        reg_model.load_network()
    else:
        reg_model.fit()
        reg_model.save_network()
    
    # plot losses
    general.plot_loss(reg_model, **reg_model.args)
    
    recon_metrics = reg_model.temporal_reconstruction(image_stack, gt_stack, reference=reference)
    
    # pprint.pprint(recon_metrics)

    return recon_metrics

if __name__ == "__main__":
  
    parser = argparse.ArgumentParser(description="Run temporal interpolation on DIR-LAB dataset")
    parser.add_argument("--path", type=str, required=True, help="Path to the DIR-LAB dataset")
    parser.add_argument("--save_folder", type=str, default="./results", help="Folder to save results")
    parser.add_argument("--exists_ok", action="store_true", help="Skip existing results")
    parser.add_argument("--sdf_alpha", type=float, default=0.05, help="SDF alpha value")
    parser.add_argument("--point_alpha", type=float, default=0.01, help="Point alpha value")
    parser.add_argument("--n_encoding_features", type=int, default=6, help="Number of encoding features")
    parser.add_argument("--n_epochs", type=int, default=10000, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--test_name", type=str, default="second", help="Type of test: hold_out, second, 0_3_5_8, extreme_phases")

    args = parser.parse_args()
    dir_folder = args.path
    folder = args.save_folder
    os.makedirs(folder, exist_ok=True)

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
    test_name = args.test_name
    assert test_name in ["hold_out", "second", "0_3_5_8", "extreme_phases"], "Invalid test name. Choose from: hold_out, second, 0_3_5_8, extreme_phases"
    n_repeats = 10 if test_name=="hold_out" else 2 if test_name=="second" else 1
    
    output = defaultdict(dict)
    t_means = {}
    print("\n")
    
    for variation in range(1, 11):
        for h in range(n_repeats):
            kwargs = deepcopy(kwargs_orig)
            if test_name == "hold_out":
                time_weights = np.ones(10)
                time_weights[h] = 0
            elif test_name == "second":
                time_weights = np.ones(10)
                time_weights[h::2] = 0
            elif test_name == "0_3_5_8":
                time_weights = np.array([1,0,0,1,0,1,0,0,1,0])
            elif test_name == "extreme_phases":
                time_weights = np.array([1,0,0,0,0,1,0,0,0,0])
            kwargs["time_weights"] = time_weights.tolist()
            kwargs["save_folder"] = os.path.join(folder, f"variation_{variation:02d}", f"reconstruction_{test_name}", 
                                                 f"hold_out_{h:02d}" if test_name == "hold_out" else "second_{h:02d}" if test_name == "second" else test_name)
            
            print(f"Running test: {test_name} - Variation {variation:02d} - {h+1:02d}/{n_repeats:02d}")
            os.makedirs(kwargs["save_folder"], exist_ok=True)
            
            recon_metrics = main(dir_folder, variation, exist_ok=args.exists_ok, **kwargs)
            
            # combine dicts
            for key, value in recon_metrics.items():
                output[variation][key] = value
                
        with open(os.path.join(folder, f"variation_{variation:02d}", f"reconstruction_{test_name}", f"results_{test_name}.json"), "w") as f:
            json.dump(output[variation], f, indent=4)

    with open(os.path.join(folder, f"reconstruction_metrics", f"results_{test_name}.json"), "w") as f:
        json.dump(output, f, indent=4)
    print(f"\n\nResults from insp to exp - {test_name}")
    print(" "*12+", ".join([f"{method:>13}" for method in list(output.values())[0].keys()]))
    for key, value_mean in output.items():
        print(f"Subject {key:02d}:", end=" ")
        print(", ".join([f"{np.mean(value):>13.2f}" for key, value in value_mean.items()]))