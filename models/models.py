import torch
import torch.nn as nn
import torch.optim as optim
import os
import tqdm
import numpy as np
from collections import defaultdict
from types import SimpleNamespace
from copy import deepcopy
import json


from utils import general
from networks import networks
from objectives import ncc, tripleLoss
from objectives import regularizers


class TemporalImplicitRegistrator:
    """This is a class for implicitly registrating a cycle of images.
    With the option to incorperate multiple image modalities e.g. HU-values and signed distance field óf a shape of interest."""


    def __init__(self, image_stack, mask=None, **kwargs):
        """Initialize the learning model.
        
        Args:
            image_stack (torch.Tensor): The image stack, shape (n, x, y, z) or (n, x, y, z, 2).
            mask (torch.Tensor): A mask to sample coordinates for training within (same mask for all images), shape (x, y, z)
            **kwargs: Additional keyword arguments.
        """

        # Set all default arguments in a dict: self.args
        
        self.set_default_arguments()
        self.parse_arguments(**kwargs)
        self.initialize_network_optimizer_criterion()
        os.makedirs(self.opts.save_folder, exist_ok=True)
        self.save_kwargs(**kwargs)
        self.opts.network_name = os.path.join(self.opts.save_folder, "network.pth")

        assert image_stack.shape[0] == self.opts.cycle_length, "The number of images in the stack must be equal to the cycle length."

        self.use_both_sdf_and_hu = image_stack.ndim == 5
        # Initialization
        self.image_stack = image_stack.cuda()
        self.shape = self.image_stack.shape[1:-1] if self.use_both_sdf_and_hu else self.image_stack.shape[1:]
        self.possible_coordinate_tensor = general.make_masked_coordinate_tensor(self.shape, mask)
        self.time_steps = torch.arange(0, self.opts.cycle_length, 1).cuda()
        self.encoded_time_steps = general.encode_time(self.time_steps, self.opts.cycle_length)
        
        self.time_weights = torch.tensor(self.opts.time_weights, dtype=torch.float32) if self.opts.time_weights\
                        else torch.ones(self.opts.cycle_length, dtype=torch.float32)

    def save_kwargs(self, **kwargs):
        """Save the keyword arguments."""
        with open(os.path.join(kwargs["save_folder"], "kwargs.json"), "w") as f:
            json.dump(kwargs, f)
        
    def set_default_arguments(self):
        """Set default arguments."""
        filename = 'models/default_arguments.json'
        if not os.path.exists(filename):
            filename = '../models/default_arguments.json'
            
        with open(filename, 'r') as file:
            
            self.args = json.load(file)
            
    def parse_arguments(self, **kwargs):
        """Parse arguments from kwargs."""
        assert all(kwarg in self.args.keys() for kwarg in kwargs)
        
        self.args.update(kwargs)
                
        self.args = {k: v.lower() if isinstance(v, str) else v for k,v in self.args.items()}
        self.opts = SimpleNamespace(**self.args)
        
    def initialize_network_optimizer_criterion(self):
        """Initialize the network, optimizer and criterion."""

        # Set seed
        torch.manual_seed(self.opts.seed)
    
        # Init network
        if self.opts.network_type == "mlp":
            self.network = networks.MLP(self.opts.layers)
        else:
            self.network = networks.Siren(self.opts.layers,
                                          self.opts.weight_init,
                                          self.opts.omega,
                                          self.opts.encoding_type,
                                          self.opts.n_encoding_features,
                                          self.opts.skip_connection,
                                          self.opts.include_original_coords)
    
        self.network.cuda()

        # Choose the optimizer
        if self.opts.optimizer == "sgd":
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.opts.lr, momentum=self.opts.momentum)
        elif self.opts.optimizer == "adam":
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.opts.lr)
        elif self.opts.optimizer == "adadelta":
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.opts.lr)
        elif self.opts.optimizer == "adamw":
            self.optimizer = optim.AdamW(self.network.parameters(), lr=self.opts.lr)
        else:
            self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum=self.opts.momentum)
            print(f"WARNING: {self.opts.optimizer} not recognized as optimizer, picked SGD instead")
            
        if self.opts.scheduler == "step":
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=750, gamma=0.5)
        elif self.opts.scheduler == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=50)
        elif self.opts.scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        else:
            self.scheduler = None

        # Choose the loss function
        criterions = {"mse": nn.MSELoss, "l1": nn.L1Loss, "ncc": ncc.NCC, "smoothl1": nn.SmoothL1Loss}
        
        self.hu_criterion = criterions[self.opts.hu_loss_function]()
        self.sdf_criterion = criterions[self.opts.sdf_loss_function]()
            
           
        self.triple_point_loss = tripleLoss.TriplePointLoss()
        self.losses = defaultdict(list)
    
    
    def fit(self, epochs=None):
        """Train the network."""

        # Determine epochs
        if epochs is None:
            epochs = self.opts.epochs

        # Set seed
        torch.manual_seed(self.opts.seed)

        # Perform training iterations
        if self.opts.verbose:
            print(f"\nNetwork contains {general.count_parameters(self.network)} trainable parameters.")
            pbar = tqdm.tqdm(range(epochs), ncols=100)
        else:
            pbar = range(epochs)

        for i in pbar:
            self.training_iteration(i)
            if self.scheduler is not None:
                pbar.set_postfix(loss=f"{self.losses['total_loss'][i]:.6f}", lr=f"{self.scheduler.get_last_lr()[0]:.6f}")
            else:
                pbar.set_postfix(loss=f"{self.losses['total_loss'][i]:.6f}")

            
    def training_iteration(self, epoch):
        """Perform one iteration of training."""

        # Reset the gradient
        self.network.train()

        loss = 0.0
        indices = torch.randperm(self.possible_coordinate_tensor.shape[0], device="cuda")[: self.opts.batch_size]
        coordinate_tensor = self.possible_coordinate_tensor[indices, :]
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        
        time_indices_source = torch.multinomial(self.time_weights,
                                                #torch.ones(self.opts.cycle_length), 
                                                self.opts.batch_size, 
                                                replacement=True)
        time_coordinates_source = self.encoded_time_steps[time_indices_source]
        # time_coordinates_source = time_coordinates_source.requires_grad_(True)
        
        time_indices_target = torch.multinomial(self.time_weights,
                                                #torch.ones(self.opts.cycle_length), 
                                                self.opts.batch_size, 
                                                replacement=True)
        time_coordinates_target = self.encoded_time_steps[time_indices_target]
        # time_coordinates_target = time_coordinates_target.requires_grad_(True)
        
        # coordinates map from target to source
        input_coordinates = torch.cat([coordinate_tensor, 
                                       time_coordinates_target, 
                                       time_coordinates_source
                                       ], dim=-1)

        output = self.network(input_coordinates)
        coordinate_transformed = torch.add(output, coordinate_tensor)
        # output = coord_temp
        
        # transforming the coordinates from target to source and thereby from end to start
        target_image      = general.temporal_interpolation(self.image_stack, 
                                                           time_indices_target, 
                                                           coordinate_tensor)
        transformed_image = general.temporal_interpolation(self.image_stack, 
                                                           time_indices_source, 
                                                           coordinate_transformed)
        # target_image, transformed_image = torch.zeros_like(time_indices_target).float().cuda(), torch.zeros_like(time_indices_target).float().cuda()
         # Compute the loss
        if not self.use_both_sdf_and_hu:
            hu_loss = self.hu_criterion(transformed_image, target_image)
            loss += hu_loss
            self.losses["hu_loss"].append(loss.detach().cpu().numpy())
        else:
            hu_loss =  self.hu_criterion(transformed_image[...,0], target_image[...,0])
            sdf_loss = self.sdf_criterion(transformed_image[...,1], target_image[...,1])
            self.losses["hu_loss"].append(hu_loss.detach().cpu().numpy())
            self.losses["sdf_loss"].append(sdf_loss.detach().cpu().numpy())
            
            # loss += (1-self.opts.sdf_alpha) * hu_loss + self.opts.sdf_alpha * sdf_loss
            loss += hu_loss + self.opts.sdf_alpha * sdf_loss

        # perform n forward passes through the network for time consistency
        coordinates_moved_forward, coordinates_moved_backward = self.do_single_forward_passes(coordinate_tensor, time_indices_target, time_indices_source)
        
        point_loss = self.triple_point_loss(coordinate_transformed, coordinates_moved_forward, coordinates_moved_backward)
        self.losses["point_loss"].append(point_loss.detach().cpu().numpy())
        loss += self.opts.point_alpha * point_loss
           
        
        # why do this? 
        # Relativation of output
        # output_rel = torch.subtract(output, coordinate_tensor)

        # Regularization
        if self.opts.jacobian_symmetric:
            jac_loss = regularizers.compute_jacobian_symmetric_loss(coordinate_tensor, output)
            loss += self.opts.alpha_jacobian * jac_loss
            self.losses["jacobian_loss"].append(jac_loss.detach().cpu().numpy())
        if self.opts.pTV:
            pTV_loss = regularizers.compute_jacobian_symmetric_loss(coordinate_tensor, output)
            loss += self.opts.alpha_jacobian * pTV_loss
            self.losses["pTV_loss"].append(pTV_loss.detach().cpu().numpy())

        self.losses["total_loss"].append(loss.detach().cpu().numpy())
        
        # Perform the backpropagation and update the parameters accordingly
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.opts.scheduler == "plateau":
            self.scheduler.step(hu_loss)
        elif self.scheduler is not None:
            self.scheduler.step()


    
    def step_cycle(self, time_indices, step=1):
        return (time_indices + step) % self.opts.cycle_length
    
    def do_single_forward_passes(self, coordinate_tensor, time_indices_target, time_indices_source):
        """
        Perform n sequential forward passes through the network.

        Args:
            coordinate_tensor (torch.Tensor): The input coordinate tensor.
            time_indices_target (torch.Tensor): The time indices - start time.
            time_indices_source (torch.Tensor): The time indices - end time.

        Returns:
            torch.Tensor: The output tensor after n forward passes.
            torch.Tensor: The moved coordinates after n forward passes.
        """
        coordinates_moved_forward, coordinates_moved_backward = coordinate_tensor.clone(), coordinate_tensor.clone()
        output_forward, output_backward = torch.zeros_like(coordinate_tensor), torch.zeros_like(coordinate_tensor)
        
        steps = (time_indices_source - time_indices_target) % self.opts.cycle_length
        steps_reverse = self.opts.cycle_length - steps      

        for i in range(steps.max()):
            idx = steps > i
            tmp_time_indices = time_indices_target[idx]
            tmp_coordinate_tensor = coordinates_moved_forward[idx]
            start_time = self.step_cycle(tmp_time_indices, step=i)
            end_time   = self.step_cycle(tmp_time_indices, step=i+1)

            tmp_input_coordinates = torch.cat([tmp_coordinate_tensor, self.encoded_time_steps[start_time], self.encoded_time_steps[end_time]], dim=-1)

            tmp_output = self.network(tmp_input_coordinates)
            output_forward[idx] = tmp_output    
            tmp_coords = torch.add(tmp_output, tmp_coordinate_tensor)
            coordinates_moved_forward[idx] = tmp_coords
            
        for i in range(steps_reverse.max()):
            idx = steps_reverse > i
            tmp_time_indices = time_indices_target[idx]
            tmp_coordinate_tensor = coordinates_moved_backward[idx]
            start_time = self.step_cycle(tmp_time_indices, step=-i)
            end_time   = self.step_cycle(tmp_time_indices, step=-i-1)
            
            tmp_input_coordinates = torch.cat([tmp_coordinate_tensor, self.encoded_time_steps[start_time], self.encoded_time_steps[end_time]], dim=-1)
            
            tmp_output = self.network(tmp_input_coordinates)
            output_backward[idx] = tmp_output
            tmp_coords = torch.add(tmp_output, tmp_coordinate_tensor)
            coordinates_moved_backward[idx] = tmp_coords

        return coordinates_moved_forward, coordinates_moved_backward
                    
    def full_cycle_regularization(self, coordinate_tensor, time_indices, calculate_velocity=False):
        #TODO: remove unused function
        # calculate velocity from the output of the network
        coordinates_moved_forward, coordinates_moved_backward = coordinate_tensor.clone(), coordinate_tensor.clone()
        velocity_forward, velocity_backward = torch.zeros((self.opts.batch_size, *coordinate_tensor.shape)), torch.zeros((self.opts.batch_size, *coordinate_tensor.shape))
        
        for i in range(self.opts.cycle_length):
            start_time = self.step_cycle(time_indices, step=i)
            end_time = self.step_cycle(time_indices, step=i+1)
            
            input_coordinates = torch.cat([coordinates_moved_forward, self.encoded_time_steps[start_time], self.encoded_time_steps[end_time]], dim=-1)
            output = self.network(input_coordinates)
            new_coordiantes = torch.add(output, coordinates_moved_forward)
            if calculate_velocity:
                #TODO: calculate velocity correcly
                velocity = new_coordiantes - coordinates_moved_forward
                velocity_forward[i] = velocity
            coordinates_moved_forward = new_coordiantes
            
        for i in range(self.opts.cycle_length):
            start_time = self.step_cycle(time_indices, step=-i)
            end_time = self.step_cycle(time_indices, step=-i-1)
            
            input_coordinates = torch.cat([coordinates_moved_backward, self.encoded_time_steps[start_time], self.encoded_time_steps[end_time]], dim=-1)
            output = self.network(input_coordinates)
            new_coordiantes = torch.add(output, coordinates_moved_backward)
            if calculate_velocity:
                velocity = new_coordiantes - coordinates_moved_backward
                velocity_backward[i] = velocity
            coordinates_moved_backward = new_coordiantes
            
        if calculate_velocity:
            return coordinates_moved_forward, coordinates_moved_backward, velocity_forward, velocity_backward
        else:
            return coordinates_moved_forward, coordinates_moved_backward
    
    def transform_volume(self, time_indices_source, time_indices_target, dims=None, image_stack=None, interpolation="trilinear", use_taylor=False, do_target_encoding=True):
        #TODO: make work for new input...
        #
        dims = self.shape if dims is None else dims
        image_stack = torch.tensor(image_stack.astype(float), device="cuda").float() if isinstance(image_stack, np.ndarray) else image_stack.cuda()
        
        coordinate_tensor = general.make_masked_coordinate_tensor(dims)
        coordinate_chunks = torch.split(coordinate_tensor, self.opts.batch_size)
        time_coordinates_source = self.encoded_time_steps[time_indices_source].cuda()
        
        if do_target_encoding:
            time_coordinates_target = self.encoded_time_steps[time_indices_target]
        else:
            time_coordinates_target = time_indices_target.cuda()
        
        transformed_points = []
        for chunk in coordinate_chunks:
            tmp_time_target = time_coordinates_target.repeat(chunk.shape[0], 1)
            tmp_time_source = time_coordinates_source.repeat(chunk.shape[0], 1)
            if use_taylor:
                coord_temp, uncertainty = self.taylor_transform_points(chunk, tmp_time_target, tmp_time_source)
                transformed_points.append(coord_temp.cpu().detach())
                # print(uncertainty.mean())
            else:
                input_coordinates = torch.cat([chunk, tmp_time_target, tmp_time_source], dim=-1)
                output = self.network(input_coordinates)
                coord_temp = torch.add(output, chunk)
                transformed_points.append(coord_temp.cpu().detach())
            # coord_temp = torch.add(output, chunk)
            # outputs.append(output.cpu().detach())
            
        # outputs = torch.cat(outputs).cuda()
        # Shift coordinates by 1/n * v
        # coord_temp = torch.add(outputs, coordinate_tensor)
        coord_temp = torch.cat(transformed_points).cuda()
        
        transformed_image = general.temporal_interpolation(image_stack, time_indices_source, coord_temp, interpolation=interpolation)
        
        return transformed_image.cpu().detach().reshape(dims)

    def transform_volume_over_cycle(self, image_stack, start_time=0, method="direct", interpolation="trilinear", dims=None):
        
        assert method in ["direct", "sequential", "taylor", "combined"]

        dims = self.shape if dims is None else dims
        image_stack = image_stack.unsqueeze(0) if image_stack.ndim == 3 else \
                       image_stack[..., 0] if image_stack.ndim == 5  else  \
                       image_stack
        
        transformed_images = torch.zeros((self.opts.cycle_length, *self.shape))
        transformed_images[0] = image_stack[0]
        
        for i in range(1, self.opts.cycle_length):
            start_time_tmp = self.step_cycle(start_time, step=i-1) if method=="sequential" else start_time
            end_time_tmp = self.step_cycle(start_time, step=i)
            transformed_images[i] = self.transform_volume(start_time_tmp, 
                                                          end_time_tmp, 
                                                          dims, 
                                                          transformed_images if method=="sequential" else image_stack, 
                                                          interpolation,
                                                          method == "taylor")
            
        return transformed_images #.numpy()
    
    def temporal_resulution_transform(self, image_stack, temporal_res, method="direct", interpolation="trilinear", dims=None, save_slices_only=False, use_gt_images=True):
        
        # remove sdf channel if present
        image_stack = image_stack[..., 0] if image_stack.ndim == 5  else image_stack
        
        if save_slices_only:
            transformed_images = torch.zeros((temporal_res, self.shape[0],self.shape[1]))
            transformed_images[0] = image_stack[0,:,:,28]
        else:
            transformed_images = torch.zeros((temporal_res, *self.shape))
            transformed_images[0] = image_stack[0]
        
        short_seq = np.linspace(0, 1, self.opts.cycle_length, endpoint=False)  # 10 steps from 0 to 1
        long_seq = np.linspace(0, 1, temporal_res, endpoint=False)
        
        indices = np.searchsorted(short_seq, long_seq, side='right') - 1
        # Ensure indices stay within valid bounds
        indices = np.clip(indices, 0, len(short_seq) - 2)
        
        for i in tqdm.trange(1, temporal_res):
                target_time_enc = general.encode_time(i, temporal_res)
                source_time_backwards = indices[i]
                source_time_forwards = self.step_cycle(indices[i])
                if i == source_time_backwards and use_gt_images:
                    transformed_im = image_stack[source_time_backwards]
                else:
                    transformed_backwards = self.transform_volume(source_time_backwards, 
                                                                target_time_enc, 
                                                                dims, 
                                                                image_stack, 
                                                                interpolation, 
                                                                method=="taylor", 
                                                                do_target_encoding=False)
                    transformed_forwards = self.transform_volume(source_time_forwards, 
                                                                target_time_enc, 
                                                                dims, 
                                                                image_stack, 
                                                                interpolation, 
                                                                method=="taylor", 
                                                                do_target_encoding=False)
                    transformed_im = (transformed_backwards + transformed_forwards) / 2
                if save_slices_only:
                    transformed_images[i] = transformed_im[:,:,28]
                else:
                    transformed_images[i] = transformed_im
        return transformed_images
    
    def temporal_reconstruction(self, image_stack, gt_stack, reference=None, method="direct", interpolation="trilinear", dims=None, calculate_dice=False, only_bf=True):
        
        # remove sdf channel if present
        image_stack = image_stack[..., 0] if image_stack.ndim == 5  else image_stack
        gt_stack = gt_stack[..., 0] if gt_stack.ndim == 5  else gt_stack
        
        recon_ids = torch.argwhere(self.time_weights==0).view(-1)
        train_ids = torch.argwhere(self.time_weights>0).view(-1)
        forward_ids = train_ids[torch.searchsorted(train_ids, recon_ids, right=True) % len(train_ids)]
        backward_ids = train_ids[torch.searchsorted(train_ids, recon_ids, right=False) - 1]
        # transformed_images = torch.zeros((len(recon_ids), *self.shape))
        metrics = {}
        
        if reference is not None:
            os.makedirs(os.path.join(self.opts.save_folder, "reconstructions"), exist_ok=True)
        
        for i, r_idx in enumerate(recon_ids):
            target_time_enc = general.encode_time(r_idx, self.opts.cycle_length)
            # forward_idx = self.step_cycle(r_idx, step=1)
            # train_ids[train_ids > r_idx].min()
            # backward_idx = self.step_cycle(r_idx, step=-1)
            # train_ids[train_ids < r_idx].max()
            if only_bf:
                train_ids = torch.tensor([forward_ids[i], backward_ids[i]], device="cuda")
            transformed_images = []
            for t_idx in train_ids:
                source_time = t_idx
                transformed = self.transform_volume(source_time, 
                                                            target_time_enc, 
                                                            dims, 
                                                            image_stack, 
                                                            interpolation, 
                                                            method=="taylor", 
                                                            do_target_encoding=False)
                transformed_images.append(transformed)
            
                if t_idx == forward_ids[i]:
                    forward_image = transformed.clone()
                elif t_idx == backward_ids[i]:
                    backward_image = transformed.clone()
            
            
            transformed_back_forward = (forward_image + backward_image) / 2
            transformed_mean = torch.stack(transformed_images).float().mean(dim=0)
            transformed_median = torch.stack(transformed_images).float().median(dim=0)[0]
            
            save_prefix = f"recon_{r_idx:02d}" if not calculate_dice else f"seg_{r_idx:02d}"
            if reference is not None:
                general.save_torch_volume_as_sitk(transformed_back_forward, os.path.join(self.opts.save_folder, "reconstructions", f"{save_prefix}_bf.nii.gz"), reference, convert_to_HU=not calculate_dice)
                if not only_bf:
                    general.save_torch_volume_as_sitk(transformed_mean, os.path.join(self.opts.save_folder, "reconstructions", f"{save_prefix}_mean.nii.gz"), reference, convert_to_HU=not calculate_dice)
                    general.save_torch_volume_as_sitk(transformed_median, os.path.join(self.opts.save_folder, "reconstructions", f"{save_prefix}_median.nii.gz"), reference, convert_to_HU=not calculate_dice)
            
            if calculate_dice:
                metrics[f"d_{r_idx}"] = {"segmentation": general.dice_score(gt_stack[r_idx], transformed_back_forward.round())[0],
                                        #  "mean": general.dice_score(gt_stack[r_idx], transformed_mean.round())[0],
                                        #  "median": general.dice_score(gt_stack[r_idx], transformed_median.round())[0]
                                         }
            else:
                metrics[f"f_{r_idx}"] = {"recon": general.compute_reconstruction_metrics(gt_stack[r_idx], transformed_back_forward, reference=reference),
                                    # "mean": general.compute_reconstruction_metrics(gt_stack[r_idx], transformed_mean, reference=reference),
                                    # "median": general.compute_reconstruction_metrics(gt_stack[r_idx], transformed_median, reference=reference)
                                    }
        return metrics
            
    def transform_points(self, points, time_coordinates_start, time_coordinates_end, return_jacobian=False, do_encoding=False):
        
        #TODO: make cycle consistent (forward and backward pass for consistency)
        
        coordinate_tensor = torch.tensor(points, device="cuda").float()
        
        # time_coordinates_start = self.encoded_time_steps[time_indices_start, :]
        # time_coordinates_end = self.encoded_time_steps[time_indices_end, :]
        
        if do_encoding:
            if isinstance(time_coordinates_start, int):
                start_time = torch.tensor([time_coordinates_start]*points.shape[0], device="cuda")
            time_coordinates_start = self.encoded_time_steps[start_time]
            if isinstance(time_coordinates_end, int):
                end_time = torch.tensor([time_coordinates_end]*points.shape[0], device="cuda")
            time_coordinates_end = self.encoded_time_steps[end_time]
        input_coordinates = torch.cat([coordinate_tensor, time_coordinates_start, time_coordinates_end], dim=-1)
        
        coordinate_chunks = torch.split(input_coordinates, self.opts.batch_size)
        outputs, jacobians = [], []
        for chunk in coordinate_chunks:
            output = self.network(chunk)
            # coord_temp = torch.add(output, chunk)
            outputs.append(output.cpu().detach())
            if return_jacobian:
                jacobian = regularizers.compute_jacobian_matrix(chunk[:,:3], output, add_identity=False)
                jacobians.append(jacobian.cpu().detach().numpy())
            
        outputs = torch.cat(outputs).cuda()

        # Shift coordinates by 1/n * v
        coord_temp = torch.add(outputs, coordinate_tensor)
        if return_jacobian:
            jacobians = np.concatenate(jacobians)
            return coord_temp.cpu().detach().numpy(), jacobians
        else:
            return coord_temp.cpu().detach().numpy()
    
    def transform_points_over_cycle(self, points, start_time=0, method="direct", reference=None, order=1):
        """
        Transforms the given points over a cycle of time.
        Args:
            points (numpy.ndarray): The points to be transformed.
            start_time (int, optional): The starting time of the cycle. Defaults to 0.
            method (str, optional): The method to use for transformation. Defaults to "direct".
        Returns:
            numpy.ndarray: The transformed points over the cycle.
        """
        assert method in ["direct", "sequential", "taylor", "combined", "jacobian"]
        # check if points are in interval [-1,1] other scale them
        if reference is not None:
            points = general.scale_points_from_reference_to_1_1(points, reference)
        
        
        if isinstance(start_time, int):
            start_time = torch.tensor([start_time]*points.shape[0], device="cuda")
        start_encoded = self.encoded_time_steps[start_time]
        
        transformed_points = np.zeros((self.opts.cycle_length+1, *points.shape))
        transformed_points[0] = points
        jacobians = []
        
        for i in range(1, self.opts.cycle_length+1):
            start_time_stepped = self.step_cycle(start_time, step=i-1) 
            end_time_stepped = self.step_cycle(start_time, step=i)
            
            start_stepped_encoded = self.encoded_time_steps[start_time_stepped]
            end_stepped_encoded = self.encoded_time_steps[end_time_stepped]
            
            if method == "direct":
                tmp_points = self.transform_points(points, start_encoded, end_stepped_encoded)
            elif method == "sequential":
                tmp_points = self.transform_points(transformed_points[i-1], start_stepped_encoded, end_stepped_encoded)
            elif method == "taylor":
                tmp_points, _ = self.taylor_transform_points(points, start_encoded, end_stepped_encoded, order=order)
                tmp_points = tmp_points.cpu().detach().numpy()
            elif method == "combined":
                tmp_points_direct = self.transform_points(points, start_encoded, end_stepped_encoded)
                tmp_points_seq = self.transform_points(transformed_points[i-1], start_stepped_encoded, end_stepped_encoded)
                tmp_points = (tmp_points_direct + tmp_points_seq) / 2
            elif method == "jacobian":
                tmp_points, jacobian = self.transform_points_jacobian(points, start_time, end_time_stepped)
                jacobians.append(jacobian)

            transformed_points[i] = tmp_points

        if reference is not None:
            transformed_points = general.scale_points_from_1_1_to_reference(transformed_points, reference)
        
        if method == "jacobian":
            return transformed_points, jacobians
        else:
            return transformed_points
    
    def taylor_transform_points(self, points, time_coordinates_start, time_coordinates_end, order=1):
        """
        Transform the given points using a taylor expansion of the transformation function.
        
        Args:
            points (numpy.ndarray): The points to be transformed.
            time_indices_start (int): The starting time index.
            time_indices_end (int): The ending time index.
            order (int, optional): The order of the taylor expansion. Defaults to 1.
            
        Returns:
            numpy.ndarray: The transformed points.
            numpy.ndarray: The uncertainty of the transformation, calculated as the difference between x and f^-1(f(x)).
        """
        
        coordinate_tensor = torch.tensor(deepcopy(points), device="cuda").float()
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        
        # time_coordinates_start = self.encoded_time_steps[time_indices_start, :]
        # time_coordinates_end = self.encoded_time_steps[time_indices_end, :]
        
        input_forward = torch.cat([coordinate_tensor, time_coordinates_start, time_coordinates_end], dim=-1)
        output_forward = self.network(input_forward)
        coordinates_forward = torch.add(output_forward, coordinate_tensor)
        
        input_backward = torch.cat([coordinates_forward, time_coordinates_end, time_coordinates_start], dim=-1)
        output_backward = self.network(input_backward)
        coordinates_backward = torch.add(output_backward, coordinates_forward)
        
        dist_start_to_backward = coordinate_tensor - coordinates_backward
        # add identity matrix to jacobian?
        jacobian_backward = regularizers.compute_jacobian_matrix(coordinates_forward, coordinates_backward, add_identity=False)

        dist_start_to_backward_view= dist_start_to_backward.view(dist_start_to_backward.shape[0], 1, dist_start_to_backward.shape[1])
        inv_jacobian_backward = torch.linalg.inv(jacobian_backward).transpose(-2, -1)
        offset_1st_order = torch.bmm(dist_start_to_backward_view, inv_jacobian_backward).squeeze(1).detach()
        transformed_offset = offset_1st_order
        
        if order > 1:
            hessian_backward = regularizers.compute_jacobian_matrix(coordinates_forward, coordinates_forward+offset_1st_order, add_identity=False)
            hessian_backward_T = hessian_backward.transpose(-2, -1)
            offset_2nd_order = torch.bmm(dist_start_to_backward_view, hessian_backward_T).squeeze(1).detach()
            
            transformed_offset += 0.5 * offset_2nd_order
        
        mid_points = coordinates_forward + 0.5 * transformed_offset
        uncertainty = torch.norm(transformed_offset, dim=-1)
        return mid_points, uncertainty.cpu().detach().numpy()
        
    def transform_points_jacobian(self, points, time_indices_start, time_indices_end):
        """
        Transform the given points and calculate the jacobian of the transformation.
        
        Args:
            points (numpy.ndarray): The points to be transformed.
            time_indices_start (int): The starting time index.
            time_indices_end (int): The ending time index.
            
        Returns:
            numpy.ndarray: The transformed points.
            numpy.ndarray: The jacobian of the transformation.
        """
        coordinate_tensor = torch.tensor(points, device="cuda").float()
        coordinate_tensor = coordinate_tensor.requires_grad_(True)
        
        time_coordinates_start = self.encoded_time_steps[time_indices_start, :]
        time_coordinates_end = self.encoded_time_steps[time_indices_end, :]
        
        input_coordinates = torch.cat([coordinate_tensor, time_coordinates_start, time_coordinates_end], dim=-1)
        
        output = self.network(input_coordinates)
        coordinates = torch.add(output, coordinate_tensor)
        
        # add identity matrix to jacobian?
        jacobian = regularizers.compute_jacobian_matrix(coordinate_tensor, output, add_identity=True)
        return coordinates.cpu().detach().numpy(), jacobian.cpu().detach().numpy()
    
    def calculate_jacobians(self):
        """
        Calculate the jacobian of the transformations over the cycle.
        
        Args:
            None.
        Returns:
            list: The jacobians of the transformations.
        """
        points = self.possible_coordinate_tensor
        
        coordinate_tensor = torch.tensor(points, device="cuda").float()
             
        all_jacobians = []

        for i in range(self.opts.cycle_length):

            
            chunks = torch.split(coordinate_tensor, self.opts.batch_size)
            jacs = []
            for chunk in chunks:
                chunk = chunk.requires_grad_(True)
                start_time = torch.tensor([0]*chunk.shape[0], device="cuda")
                end_time = torch.tensor([i]*chunk.shape[0], device="cuda")
                input_coordinates = torch.cat([chunk, self.encoded_time_steps[start_time], self.encoded_time_steps[end_time]], dim=-1)
                output = self.network(input_coordinates)
                # coordinates = torch.add(output, chunk)
                
                jac = regularizers.compute_jacobian_matrix(chunk, output, add_identity=True)
                jacs.append(jac.cpu().detach().numpy())
            
            all_jacobians.append(np.concatenate(jacs))
            
        return all_jacobians
        
    def save_network(self, filename=None):
        """Save the network to a file."""

        torch.save({"state_dict": self.network.state_dict(),
                    "losses": self.losses,
                    },
                    filename if filename is not None else self.opts.network_name
                )
        
    def load_network(self, filename=None):
        """Load the network from a file."""
        
        checkpoint = torch.load(filename if filename is not None else self.opts.network_name)
        
        self.network.load_state_dict(checkpoint["state_dict"])
        self.losses = checkpoint["losses"]
            
        self.network.cuda()
            