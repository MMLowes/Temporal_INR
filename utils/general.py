import numpy as np
import os
import torch
import glob
import SimpleITK as sitk
import edt
import cupy as cp
import cupyx.scipy.ndimage as ndi
import skimage.morphology as morphology
import vtk
import matplotlib.pyplot as plt
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, normalized_root_mse


def compute_landmark_accuracy(landmarks_pred, landmarks_gt, voxel_size, return_std=True):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)

    means = np.append(np.mean(difference), means)
    stds = np.append(np.std(difference), stds)

    means = np.round(means, 2)
    stds = np.round(stds, 2)

    # means = means[::-1]
    # stds = stds[::-1]

    return (means, stds) if return_std else means

def compute_landmark_tre_per_point(landmarks_pred, landmarks_gt, voxel_size, return_std=True):
    landmarks_pred = np.round(landmarks_pred)
    landmarks_gt = np.round(landmarks_gt)

    difference = landmarks_pred - landmarks_gt
    difference = np.abs(difference)
    difference = difference * voxel_size

    means = np.mean(difference, 0)
    stds = np.std(difference, 0)

    difference = np.square(difference)
    difference = np.sum(difference, 1)
    difference = np.sqrt(difference)
    return difference

def compute_landmarks(network, landmarks_pre, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]

    coordinate_tensor = torch.FloatTensor(landmarks_pre / (scale_of_axes)) - 1.0

    output = network(coordinate_tensor.cuda())

    delta = output.cpu().detach().numpy() * (scale_of_axes)

    return landmarks_pre + delta, delta

def scale_landmarks_to_1_1(landmarks, image_size):
    scale_of_axes = [(0.5 * s) for s in image_size]
    return (landmarks / scale_of_axes) - 1.0

def scale_landmarks_to_original(landmarks, image_size):
    scale_of_axes = np.array([(0.5 * s) for s in image_size])
    return (landmarks + 1.0) * scale_of_axes

def transform_landmarks(reg_model, landmarks, start_time, end_time, reference):
    
    coords = scale_landmarks_to_1_1(landmarks, reference.GetSize())
    
    # Transform the coordinates using the registration model
    moved_coords = reg_model.transform_points(coords, start_time, end_time, do_encoding=True)
    
    moved_landmarks = scale_landmarks_to_original(moved_coords, reference.GetSize())
    
    return moved_landmarks

def save_landmarks(landmarks, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    np.savetxt(filename, landmarks, fmt='%.2f')

def save_torch_volume_as_sitk(volume, filename, reference, convert_to_HU=True):
    volume = volume.cpu().numpy().transpose(2, 1, 0)
    if convert_to_HU:
        s_min, s_max = np.quantile(sitk.GetArrayFromImage(reference), [0.001, 0.999])
        volume = (volume + 1) / 2 * (s_max - s_min) + s_min
    image = sitk.GetImageFromArray(volume)
    image.CopyInformation(reference)
    sitk.WriteImage(image, filename)
    
def load_DIRLAB_ref(folder, variation, id_num):
    image_ref = sitk.ReadImage(os.path.join(folder, f"Images/case{variation}_T{id_num*10:02d}.nii.gz"))
    return image_ref

def load_single_DIRLab(folder, variation, id_num, sdf_max=50):

    
    image_ref = sitk.ReadImage(os.path.join(folder, f"Images/case{variation}_T{id_num*10:02d}.nii.gz"))
    im = sitk.GetArrayFromImage(image_ref).astype(np.float32).transpose(2,1,0)
    s_min, s_max = np.quantile(im, [0.001, 0.999])
    # print(s_min, s_max)
    im = (im - s_min) / (s_max - s_min) * 2 - 1
    im = np.clip(im, -1, 1)
    image = torch.FloatTensor(im)
    
    mask_ref = sitk.ReadImage(os.path.join(folder, f"Masks/case{variation}_T{id_num*10:02d}_seg.nii.gz"))
    mask = sitk.GetArrayFromImage(mask_ref).astype(np.uint8).transpose(2,1,0)
    mask = mask > 0

    sdf = -edt.sdf(mask, 
                    anisotropy=mask_ref.GetSpacing(), 
                    parallel=8 # number of threads, <= 0 sets to num cpu
                    )
    
    sdf = torch.FloatTensor(sdf)
    
    sdf[sdf > sdf_max] = sdf_max
    sdf[sdf > 0] /= sdf_max
    sdf[sdf < 0] /= abs(sdf.min())
    
    return image, sdf, mask, image_ref

def load_stack_DIRLab(dir_folder, variation, use_CT=True, use_sdf=True, sdf_max=50):
    
    folder = os.path.join(dir_folder, f"Case{variation}Pack/")
    
    image_stack = []
    references = []
    
    total_mask = None
    
    for i in range(10):
        
        im, sdf, mask, ref = load_single_DIRLab(folder, variation, i, sdf_max=sdf_max)
        
        if use_CT and use_sdf:
            image = torch.stack((im, sdf),-1)
        elif use_CT:
            image = im
        elif use_sdf:
            image = sdf
        
        image_stack.append(image)
        references.append(ref) 
        
        total_mask = mask if total_mask is None else np.logical_or(total_mask, mask)
        
    # dilate the mask

    total_mask = morphology.binary_dilation(total_mask, morphology.ball(5))
    
    assert check_references_are_equal(references), "References are not equal"
    
    reference = references[0]
        
    return torch.stack(image_stack, 0), total_mask, reference
    
def load_empty_stack_DIRLab(dir_folder, variation, use_CT=True, use_sdf=True, sdf_max=50):
    
    folder = os.path.join(dir_folder, f"Case{variation}Pack/")

    im, sdf, mask, ref = load_single_DIRLab(folder, variation, 0, sdf_max=sdf_max)
    
    if use_CT and use_sdf:
        image = torch.stack((im, sdf),-1)
    elif use_CT:
        image = im
    elif use_sdf:
        image = sdf
        
    image_stack = [torch.zeros_like(image) for i in range(10)]
        
    return torch.stack(image_stack, 0), None, ref

def load_DIRLab_landmarks(dir_folder, variation, ref):
    
    size = np.array(ref.GetSize())
    
    folder = os.path.join(dir_folder, f"Case{variation}Pack/")
    files = glob.glob(os.path.join(folder, "*xtremePhases", f"*.txt"))
    files.sort()
    
    landmarks_insp = np.loadtxt(files[0])
    landmarks_insp[:,2] = size[2] - landmarks_insp[:,2]
    
    landmarks_exp = np.loadtxt(files[1])
    landmarks_exp[:,2] = size[2] - landmarks_exp[:,2]
    
    files = glob.glob(os.path.join(folder, "Sampled4D", f"case{variation}*.txt"))
    files.sort()
    
    landmarks_sampled = np.array([np.loadtxt(file) for file in files])
    for i in range(len(landmarks_sampled)):
        landmarks_sampled[i,:,2] = size[2] - landmarks_sampled[i,:,2]
            
    return landmarks_insp, landmarks_exp, landmarks_sampled


def encode_time(t, T=20):
    """
    Encode the time frame t using sinusoidal functions for a cyclic representation.
    
    :param t: The current time frame (integer).
    :param T: The total number of time frames in the cycle.
    :return: A tuple (sin(t/T * 2π), cos(t/T * 2π)).
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t).float()
    theta = 2 * torch.pi * t / T
    return torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1) 

    
def fti(input_array, indices):
    x_indices, y_indices, z_indices = indices.split(1, dim=1)
    return fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices)

def fast_trilinear_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1] * x * (1 - y) * z
        + input_array[x0, y1, z1] * (1 - x) * y * z
        + input_array[x1, y1, z0] * x * y * (1 - z)
        + input_array[x1, y1, z1] * x * y * z
    )
    return output

def fast_trilinear_interpolation_4D(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[0] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[1] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[2] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[0] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[1] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[2] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[x0, y0, z0].T * (1 - x) * (1 - y) * (1 - z)
        + input_array[x1, y0, z0].T * x * (1 - y) * (1 - z)
        + input_array[x0, y1, z0].T * (1 - x) * y * (1 - z)
        + input_array[x0, y0, z1].T * (1 - x) * (1 - y) * z
        + input_array[x1, y0, z1].T * x * (1 - y) * z
        + input_array[x0, y1, z1].T * (1 - x) * y * z
        + input_array[x1, y1, z0].T * x * y * (1 - z)
        + input_array[x1, y1, z1].T * x * y * z
    )
    return output.T


def fast_trilinear_temporal_interpolation(input_array, t_indices, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[3] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)

    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    output = (
        input_array[t_indices, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[t_indices, x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[t_indices, x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[t_indices, x1, y1, z0] * x * y * (1 - z)
        + input_array[t_indices, x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[t_indices, x1, y0, z1] * x * (1 - y) * z
        + input_array[t_indices, x0, y1, z1] * (1 - x) * y * z
        + input_array[t_indices, x1, y1, z1] * x * y * z
    )
    return output

def temporal_interpolation(input_array, t_indices, indices, interpolation="trilinear"):
    x_indices, y_indices, z_indices = indices.split(1, dim=1)
    if interpolation.find("nearest") != -1:
        return fast_nearest_neighbor_interpolation_4D(input_array, t_indices, x_indices, y_indices, z_indices)
    elif interpolation.find("linear") != -1:
        return fast_trilinear_temporal_interpolation_4D(input_array, t_indices, x_indices, y_indices, z_indices)
    else:
        raise ValueError(f"Interpolation method {interpolation} not recognized.")

def fast_nearest_neighbor_interpolation_4D(input_array, t_indices, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[3] - 1) * 0.5

    x_indices = torch.round(x_indices).to(torch.long)
    y_indices = torch.round(y_indices).to(torch.long)
    z_indices = torch.round(z_indices).to(torch.long)

    x_indices = torch.clamp(x_indices, 0, input_array.shape[1] - 1)
    y_indices = torch.clamp(y_indices, 0, input_array.shape[2] - 1)
    z_indices = torch.clamp(z_indices, 0, input_array.shape[3] - 1)

    output = input_array[t_indices, x_indices, y_indices, z_indices]
    return output


def fast_trilinear_temporal_interpolation_4D(input_array, t_indices, x_indices, y_indices, z_indices):
    """
    Performs fast trilinear temporal interpolation on a 4D input array.

    Args:
        input_array (torch.Tensor): The input array of shape (T, X, Y, Z) or (T, X, Y, Z, C).
        t_indices (torch.Tensor): The temporal indices for interpolation, in range [0, T-1]
        x_indices (torch.Tensor): The x-axis indices for interpolation, in range [-1, 1].
        y_indices (torch.Tensor): The y-axis indices for interpolation, in range [-1, 1].
        z_indices (torch.Tensor): The z-axis indices for interpolation, in range [-1, 1].

    Returns:
        torch.Tensor: The interpolated output array.

    """
    x_indices = (x_indices + 1) * (input_array.shape[1] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[2] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[3] - 1) * 0.5

    x0 = torch.floor(x_indices.detach()).to(torch.long)
    y0 = torch.floor(y_indices.detach()).to(torch.long)
    z0 = torch.floor(z_indices.detach()).to(torch.long)
    x1 = x0 + 1
    y1 = y0 + 1
    z1 = z0 + 1

    x0 = torch.clamp(x0, 0, input_array.shape[1] - 1)
    y0 = torch.clamp(y0, 0, input_array.shape[2] - 1)
    z0 = torch.clamp(z0, 0, input_array.shape[3] - 1)
    x1 = torch.clamp(x1, 0, input_array.shape[1] - 1)
    y1 = torch.clamp(y1, 0, input_array.shape[2] - 1)
    z1 = torch.clamp(z1, 0, input_array.shape[3] - 1)
    
    x = x_indices - x0
    y = y_indices - y0
    z = z_indices - z0

    t_indices = t_indices.detach().to(torch.long) if isinstance(t_indices, torch.Tensor) else t_indices
    # if input_array.ndim == 5:
    x0, y0, z0 = x0.squeeze(), y0.squeeze(), z0.squeeze()
    x1, y1, z1 = x1.squeeze(), y1.squeeze(), z1.squeeze()
    t_indices = t_indices if isinstance(t_indices, int) else t_indices.squeeze()
    
    if input_array.ndim == 4:
       x, y, z = x.squeeze(), y.squeeze(), z.squeeze()
    
    
        
    output = (
          input_array[t_indices, x0, y0, z0] * (1 - x) * (1 - y) * (1 - z)
        + input_array[t_indices, x1, y0, z0] * x * (1 - y) * (1 - z)
        + input_array[t_indices, x0, y1, z0] * (1 - x) * y * (1 - z)
        + input_array[t_indices, x1, y1, z0] * x * y * (1 - z)
        + input_array[t_indices, x0, y0, z1] * (1 - x) * (1 - y) * z
        + input_array[t_indices, x1, y0, z1] * x * (1 - y) * z
        + input_array[t_indices, x0, y1, z1] * (1 - x) * y * z
        + input_array[t_indices, x1, y1, z1] * x * y * z
    )
    return output

def fast_nearest_neighbor_interpolation(input_array, x_indices, y_indices, z_indices):
    x_indices = (x_indices + 1) * (input_array.shape[0] - 1) * 0.5
    y_indices = (y_indices + 1) * (input_array.shape[1] - 1) * 0.5
    z_indices = (z_indices + 1) * (input_array.shape[2] - 1) * 0.5

    x_indices = torch.round(x_indices).to(torch.long)
    y_indices = torch.round(y_indices).to(torch.long)
    z_indices = torch.round(z_indices).to(torch.long)

    x_indices = torch.clamp(x_indices, 0, input_array.shape[0] - 1)
    y_indices = torch.clamp(y_indices, 0, input_array.shape[1] - 1)
    z_indices = torch.clamp(z_indices, 0, input_array.shape[2] - 1)

    output = input_array[x_indices, y_indices, z_indices]
    return output

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def make_masked_coordinate_tensor(dims, mask=None):
    """Make a coordinate tensor, either masked or not."""
    coordinate_tensor = [torch.linspace(-1, 1, dims[i]) for i in range(3)]
    coordinate_tensor = torch.meshgrid(*coordinate_tensor, indexing="ij")
    coordinate_tensor = torch.stack(coordinate_tensor, dim=3)
    coordinate_tensor = coordinate_tensor.view([np.prod(dims), 3])
    
    if mask is not None:
        assert mask.shape == dims
        coordinate_tensor = coordinate_tensor[mask.flatten() > 0, :]

    coordinate_tensor = coordinate_tensor.cuda()

    return coordinate_tensor

# def compute_ncc(x1,x2,e=1e-10):
#     cc = np.mean((x1 - np.mean(x1)) * (x2 - np.mean(x2)))
#     # Compute stable standard deviations
#     std1 = np.std(x1)
#     std2 = np.std(x2)
    
#     # Compute NCC with epsilon for numerical stability
#     return cc / (std1 * std2 + e)

def compute_ncc(volume1, volume2):
    """
    Compute the Normalized Cross-Correlation (NCC) between two 3D volumes.
    
    Parameters:
    - volume1: numpy array, reference volume (ground truth)
    - volume2: numpy array, transformed/moving volume
    
    Returns:
    - ncc: float, the NCC value (-1 to 1)
    """
    assert volume1.shape == volume2.shape, "Volumes must have the same shape"

    # Flatten volumes into 1D arrays
    X = volume1.flatten()
    Y = volume2.flatten()

    # Compute mean values
    X_mean = np.mean(X)
    Y_mean = np.mean(Y)

    # Compute NCC
    numerator = np.sum((X - X_mean) * (Y - Y_mean))
    denominator = np.sqrt(np.sum((X - X_mean) ** 2) * np.sum((Y - Y_mean) ** 2))
    
    ncc = numerator / (denominator + 1e-10)  # Add small value to avoid division by zero

    return ncc

def compute_nmse(volume1, volume2):
    """
    Compute the Normalized Mean Squared Error (NMSE) between two 3D volumes.
    
    Parameters:
    - volume1: numpy array, reference volume (ground truth)
    - volume2: numpy array, transformed/moving volume
    
    Returns:
    - nmse: float, the NMSE value
    """
    # Ensure the volumes have the same shape
    assert volume1.shape == volume2.shape, "Volumes must have the same shape"
    
    # Compute NMSE
    nmse = np.sum((volume1 - volume2) ** 2) / np.sum(volume1 ** 2)
    
    return nmse

def compute_reconstruction_metrics(vol_gt, vol_pred, reference=None, convert_to_HU=False):
    if isinstance(vol_gt, torch.Tensor):
        vol_gt = vol_gt.detach().cpu().numpy()
    if isinstance(vol_pred, torch.Tensor):
        vol_pred = vol_pred.detach().cpu().numpy()
    # scale from [-1, 1] to [s_min, s_max]
    if convert_to_HU:
        if reference is not None:
            s_min, s_max = np.quantile(sitk.GetArrayFromImage(reference), [0.001, 0.999])
        vol_gt = (vol_gt + 1) / 2 * (s_max - s_min) + s_min
        vol_pred = (vol_pred + 1) / 2 * (s_max - s_min) + s_min
        data_range = s_max - s_min
    else:
        data_range = 2
    
    
    mse = compute_nmse(vol_gt, vol_pred)
    ssim = structural_similarity(vol_gt, vol_pred, data_range=data_range)
    psnr = peak_signal_noise_ratio(vol_gt, vol_pred, data_range=data_range)
    ncc = compute_ncc(vol_gt, vol_pred)
    return [float(psnr), float(ssim), float(ncc), float(mse)]

def plot_loss(reg_model, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for key, value in reg_model.losses.items():
        ax.plot(value, label=key, alpha=0.8)
    ax.set_title(f"lr: {kwargs['lr']}, batch_size: {kwargs['batch_size']}, epochs: {kwargs['epochs']}, sdf_alpha: {kwargs['sdf_alpha']}, point_alpha: {kwargs['point_alpha']}, n_encoding_features: {kwargs['n_encoding_features']}")
    ax.legend(fontsize=6)
    ax.set_xlabel("Epoch", fontsize=8)
    ax.set_ylabel("Loss", fontsize=8)
    ax.set_ylim(0, 0.2)
    os.makedirs(os.path.join(kwargs["save_folder"], "plots"), exist_ok=True)
    plt.savefig(os.path.join(kwargs["save_folder"], "plots", "losses.png"))
    plt.close()

######################################
import re
import pandas as pd
from scipy.ndimage import binary_dilation
from utils import utils


def get_name(split, case_id, sequential=False):
    name = f"{split.pseudonymized_id.iloc[case_id-1 if sequential else 0].strip()}_to_{split.pseudonymized_id.iloc[case_id].strip()}"
    return name

def load_single_segmentation(split, folder, case_id=0):
    
    
    path = os.path.join(folder, split.pseudonymized_id.iloc[case_id], "segmentations/total_seg/total_seg.nii.gz")
    image = sitk.ReadImage(path)
    im = sitk.GetArrayFromImage(image).astype(np.uint8).transpose(2,1,0)
    return torch.tensor(im), image

def load_segmentation_stack(split, folder):
        
        image_stack = []
        references = []
        
        for i in range(len(split)):
            
            im, ref = load_single_segmentation(split, folder, i)
            
            image_stack.append(im)
            references.append(ref) 
            
        assert check_references_are_equal(references), "References are not equal"
        reference = references[0]
            
        return torch.stack(image_stack, 0), reference
    
def load_single_CT_scan(split, root, case_id):
    
    path = os.path.join(root, "NIFTI", split.filename.iloc[case_id])
    
    image = sitk.ReadImage(path)
    
    # convert values to [-1,1]
    im = sitk.GetArrayFromImage(image).astype(np.float32).transpose(2,1,0)
    s_min, s_max = np.quantile(im, [0.001, 0.999])
    im = (im - s_min) / (s_max - s_min) * 2 - 1
    im = np.clip(im, -1, 1)
    im = torch.FloatTensor(im)
    
    return im, image

def load_single_sdf(split, folder, case_id, sdf_max=20, lvm_id=5):  
    
    path = os.path.join(folder, split.pseudonymized_id.iloc[case_id], "segmentations/total_seg/total_seg.nii.gz")
    image = sitk.ReadImage(path)

    im = sitk.GetArrayFromImage(image).transpose(2,1,0)
    sdf = -edt.sdf(im==lvm_id, 
                    anisotropy=image.GetSpacing(), 
                    parallel=8 # number of threads, <= 0 sets to num cpu
                    )
    sdf = torch.FloatTensor(sdf)
    sdf[sdf > sdf_max] = sdf_max
    sdf[sdf > 0] /= sdf_max
    sdf[sdf < 0] /= abs(sdf.min())
    
    # import time
    # start = time.time()
    
    im_cp = cp.array(im)
    mask = ndi.binary_dilation(im_cp > 0, structure=cp.ones((5,5,5)))
    # mask = binary_dilation(im > 0, structure=np.ones((5,5,5)))
    mask = torch.tensor(mask.get())
    # print(time.time()-start)
    return sdf, mask

def check_references_are_equal(references):
    equal_size = len(set([ref.GetSize() for ref in references])) <= 1
    equal_spacing = len(set([ref.GetSpacing() for ref in references])) <= 1
    equal_origin = len(set([ref.GetOrigin() for ref in references])) <= 1
    equal_direction = len(set([ref.GetDirection() for ref in references])) <= 1
    
    return equal_size and equal_spacing and equal_origin and equal_direction

def load_empty_image_stack(split, root, folder, use_mask=True, use_SDF=True):
    im, ref = load_single_CT_scan(split, root, 0) 
    image = torch.zeros((len(split), *im.shape, 2)) if use_SDF else torch.zeros((len(split), *im.shape))
    return image, None, ref

def load_image_stack(split, root, folder, use_mask=True, use_CT=True, use_sdf=True, sdf_max=20):
    
    assert use_CT or use_sdf, "CT and/or SDF must be used"
    
    image_stack = []
    references = []
    total_mask = None
    
    for i in range(len(split)):
        
        im, ref = load_single_CT_scan(split, root, i)
        sdf, mask = load_single_sdf(split, folder, i, sdf_max=sdf_max)
        
        if use_CT and use_sdf:
            image = torch.stack((im, sdf),-1)
        elif use_CT:
            image = im
        elif use_sdf:
            image = sdf
        
            
        image_stack.append(image)
        references.append(ref) 
        
        if use_mask:
            total_mask = np.logical_or(total_mask, mask) if total_mask is not None else mask
    
    assert check_references_are_equal(references), "References are not equal"
    reference = references[0]
        
    return torch.stack(image_stack, 0), total_mask, reference

def save_image_stack(image_stack, reference, save_folder, name, save_lvm=False):
    
    for i in range(image_stack.shape[0]):
        im = image_stack[i].cpu().numpy().transpose(2,1,0)
        image = sitk.GetImageFromArray(im)
        image.CopyInformation(reference)
        sitk.WriteImage(image, os.path.join(save_folder, f"{name}_{i*5:02d}.nii.gz"))
        
        if save_lvm:
            image_sampled = utils.resample_to_smallest_spacing(image)
            lvm_seg = image_sampled==5 
            sitk.WriteImage(lvm_seg, f"{save_folder}/{name}_lvm_seg_{i*5:02d}.nii.gz")
            utils.convert_label_map_to_surface(f"{save_folder}/{name}_lvm_seg_{i*5:02d}.nii.gz",
                                    f"{save_folder}/{name}_mesh_{i*5:02d}.vtk",
                                    segment_id=1)
    
def dice_score(vol1, vol2, labels=None, return_labels=[]):
    
    vol1 = vol1.cpu().numpy() if isinstance(vol1, torch.Tensor) else vol1
    vol2 = vol2.cpu().numpy() if isinstance(vol2, torch.Tensor) else vol2
    
    if labels is None:
        labels = np.unique(np.concatenate((vol1, vol2)))
        labels = np.delete(labels, np.where(labels == 0))  # remove background
    elif isinstance(labels, int):
        labels = [labels]
        
    dicem = np.zeros(len(labels))
    for idx, lab in enumerate(labels):
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2 * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = top / bottom

    return np.mean(dicem), dicem[return_labels]
    
def dice_stack(stack1, stack2, labels=None, return_labels=[], mean_reduce=False):
        
        assert stack1.shape == stack2.shape, "Stacks must have the same shape"
        
        if isinstance(return_labels, int):
            return_labels = [return_labels]
        
        dice_scores = []
        for i in range(stack1.shape[0]):
            d_mean, d_labels = dice_score(stack1[i], stack2[i], labels, return_labels)
            dice_scores.append([d_mean, *d_labels])
        
        return np.array(dice_scores).mean(0) if mean_reduce else np.array(dice_scores)
    
def get_bounding_box(reference):
    # Get the origin, size and spacing of the image
    origin = reference.GetOrigin()
    size = reference.GetSize()
    spacing = reference.GetSpacing()
    direction = reference.GetDirection()[::4]
    
    # Calculate the physical size of the image
    physical_size = [size[i]*spacing[i] for i in range(len(size))]
    
    # Calculate the bounding box
    direction = reference.GetDirection()[0::4]
    physical_size = [sz*sp for sz, sp in zip(size, spacing)]
    bounding_box = np.array([(o, o + d*s) for o, d, s in zip(origin, direction, physical_size)])
    bounding_box.sort(1)
    
    return bounding_box

def scale_points_from_reference_to_1_1(points, reference):

    # bounding_box = get_bounding_box(reference)
    
    # # Scale the points to the range [-1, 1]
    # min_vals = bounding_box[:, 0]
    # max_vals = bounding_box[:, 1]
    # scaled_points = 2 * ((points - min_vals) / (max_vals - min_vals)) - 1
    if reference is None:
        return points
    
    if isinstance(points, vtk.vtkPoints):
        points = vtk_to_numpy(points.GetData())
    if points.ndim == 1:
        points = points.reshape(-1, 3)
    elif points.ndim > 2:
        scaled_points = np.array([scale_points_from_reference_to_1_1(p, reference) for p in points])
        return scaled_points
        
    scaled_points = np.zeros_like(points)
    
    for i in range(len(points)):
        point = np.array(reference.TransformPhysicalPointToContinuousIndex(points[i].tolist()))
        scaled_points[i] = point
        
    scaled_points = 2*scaled_points/(np.array(reference.GetSize())-1)-1
    
    return scaled_points

def scale_points_from_1_1_to_reference(scaled_points, reference):
    
    # bounding_box = get_bounding_box(reference)
    
    # # Scale the points back to the physical space
    # min_vals = bounding_box[:, 0]
    # max_vals = bounding_box[:, 1]
    # points = ((scaled_points + 1) / 2) * (max_vals - min_vals) + min_vals
    if reference is None:
        return scaled_points
    
    if scaled_points.ndim == 1:
        scaled_points = scaled_points.reshape(-1, 3)
    elif scaled_points.ndim > 2:
        points = np.array([scale_points_from_1_1_to_reference(sp, reference) for sp in scaled_points])
        return points
    
    points = (scaled_points + 1) * (np.array(reference.GetSize()) - 1) * 0.5
    
    for i in range(len(points)):
        point = reference.TransformContinuousIndexToPhysicalPoint(points[i].tolist())
        points[i] = np.array(point)
    
    return points
