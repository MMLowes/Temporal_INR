import glob
import os
import numpy as np
import SimpleITK as sitk
import argparse

def preprocess_dirlab(path, calculate_masks=True):

    dtype = np.dtype(np.int16)

    image_sizes = [
        [94, 256, 256],
        [112, 256, 256],
        [104, 256, 256],
        [99, 256, 256],
        [106, 256, 256],
        [128, 512, 512],
        [136, 512, 512],
        [128, 512, 512],
        [128, 512, 512],
        [120, 512, 512],
    ]

    # Scale of data, per image pair
    voxel_sizes = [
        [2.5, 0.97, 0.97],
        [2.5, 1.16, 1.16],
        [2.5, 1.15, 1.15],
        [2.5, 1.13, 1.13],
        [2.5, 1.1, 1.1],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
        [2.5, 0.97, 0.97],
    ]

    for i in range(10):
        case_num = i + 1

        shape = image_sizes[i]
        res = voxel_sizes[i]
        files = glob.glob(os.path.join(path, f"Case{case_num}Pack/Images/case{case_num}_T*.img"))
        files.sort()
        for j, file in enumerate(files):
            with open(file, "rb") as f:
                data = np.fromfile(f, dtype)
            image_exp = data.reshape(shape)
            image_exp = np.flip(image_exp, axis=0)
            # image_exp = np.flip(image_exp, axis=1)
            image_exp -= 1024
            image = sitk.GetImageFromArray(image_exp)
            image.SetSpacing(res[::-1])
            # set direction matrix
            direction = np.array(
                [
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                ]
            ).flatten()
            image.SetDirection(direction.tolist())
            sitk.WriteImage(image, os.path.join(path, f"Case{case_num}Pack/Images/case{case_num}_T{j}0.nii.gz"))

            os.remove(file)
            
            
    if not calculate_masks:
        return
    
    from lungmask import LMInferer
    inferer = LMInferer()

    for i in range(10):
        case_num = i + 1

        shape = image_sizes[case_num]
        res = voxel_sizes[case_num]
        os.makedirs(f"C:/phd/DIR-LAB/Case{case_num}Pack/Masks", exist_ok=True)

        for j in range(10):
            input_image = sitk.ReadImage(os.path.join(path, f"Case{case_num}Pack/Images/case{case_num}_T{j}0.nii.gz"))
            inp = sitk.GetArrayFromImage(input_image)
            # rescale to hu values
            
            seg = inferer.apply(inp)  
            segmentation = sitk.GetImageFromArray(seg)
            segmentation.CopyInformation(input_image)
            sitk.WriteImage(segmentation, os.path.join(path, f"Case{case_num}Pack/Masks/case{case_num}_T{j}0_seg.nii.gz"))
            
            assert seg.sum() > 0, f"Segmentation failed for Case {case_num}, Time {j}"
    
    
if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run temporal registration on DIR-LAB dataset")
    parser.add_argument("--path", type=str, required=True, help="Path to the DIR-LAB dataset")
    parser.add_argument("--calculate_masks", action="store_true", default=True, help="Calculate lung masks using lungmask package")
    args = parser.parse_args()
    
    dir_folder = args.path
    calculate_masks = args.calculate_masks
    preprocess_dirlab(dir_folder, calculate_masks)