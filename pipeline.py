# !/usr/bin/env python
"""
Purpose: 3D semi-supervised patch-based training pipeline implementation and 3D patch-based overlapping inference.
Features:
* Baseline UNet and UNet-MSS training and testing
* Training baseline networks with MIP information in the form single-axis MIP comparisons with a selection of axis
* Training baseline networks with MIP information in the form of multiple-axes MIP comparisons
* Training baseline networks with Deformation-aware learning with siamese architecture
* Training baseline networks with Deformation-awareness along with MIP information in the form of MIP Loss
"""

import random
import torch
import torch.utils.data
import torchio as tio
from skimage.filters import threshold_otsu
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from Evaluation.evaluate import (IOU, Dice, FocalTverskyLoss)
from Utils.datasets import SRDataset
from Utils.elastic_transform import RandomElasticDeformation, warp_image
from Utils.madam import Madam
from Utils.result_analyser import *
from Utils.vessel_utils import (load_model, load_model_with_amp,
                                save_model, write_summary, write_epoch_summary)

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"


class Pipeline:

    def __init__(self, cmd_args, model, logger, dir_path, checkpoint_path, writer_training, writer_validating,
                 test_set=None, wandb=None):
        """
        :param cmd_args: command line arguments for initializing network parameters, experimental hyper-parameters and
        environmental parameters
        :param model: initialized baseline network i.e., 1: UNet, 2: UNet-MSS, 3: Attention UNet
        :param logger: File logger
        :param dir_path: Dataset Folder location
        :param checkpoint_path: Path to saved model state dictionary
        :param writer_training: Initialized train writer
        :param writer_validating: Initialized validation writer
        :param test_set: If specified, will be used explicitly for testing
        :param wandb: Initialized 'Weights and Biases' configuration for logging
        """
        self.logger = logger
        self.wandb = wandb
        self.model = model
        self.lr_1 = cmd_args.learning_rate
        self.num_epochs = cmd_args.num_epochs
        self.use_madam = str(cmd_args.use_madam).lower() == "true"
        if self.use_madam:
            self.optimizer = Madam(model.parameters(), lr=cmd_args.learning_rate)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
        self.clip_grads = str(cmd_args.clip_grads).lower() == "true"
        self.with_apex = str(cmd_args.apex).lower() == "true"
        self.deform = str(cmd_args.deform).lower() == "true"
        self.with_mip = str(cmd_args.with_mip).lower() == "true"
        self.mip_axis = cmd_args.mip_axis
        self.mip_loss_param_smooth = cmd_args.mip_loss_param_smooth
        self.mip_loss_param_gamma = cmd_args.mip_loss_param_gamma
        self.mip_loss_param_alpha = cmd_args.mip_loss_param_alpha

        self.writer_training = writer_training
        self.writer_validating = writer_validating
        self.checkpoint_path = checkpoint_path
        self.DATASET_FOLDER = dir_path
        self.output_path = cmd_args.output_path

        self.model_name = cmd_args.model_name

        # image input parameters
        self.patch_size = cmd_args.patch_size
        self.stride_depth = cmd_args.stride_depth
        self.stride_length = cmd_args.stride_length
        self.stride_width = cmd_args.stride_width
        self.samples_per_epoch = cmd_args.samples_per_epoch

        # execution configs
        self.batch_size = cmd_args.batch_size
        self.num_worker = cmd_args.num_worker

        # Losses
        self.dice = Dice()
        self.focalTverskyLoss = FocalTverskyLoss()
        self.mip_loss = FocalTverskyLoss(smooth=self.mip_loss_param_smooth, gamma=self.mip_loss_param_gamma,
                                         alpha=self.mip_loss_param_alpha)
        self.iou = IOU()
        self.floss_coeff = cmd_args.floss_coeff
        self.mip_loss_coeff = cmd_args.mip_loss_coeff

        self.LOWEST_LOSS = float('inf')
        self.test_set = test_set

        if self.with_apex:
            self.scaler = GradScaler()
        if self.logger is not None:
            self.logger.info("learning rate " + str(self.lr_1))
            self.logger.info("batch size " + str(self.batch_size))
            self.logger.info("patch size " + str(self.patch_size))
            self.logger.info("Gradient Clipping " + str(self.clip_grads))
            self.logger.info("With mixed precision " + str(self.with_apex))
            self.logger.info("With MIP " + str(self.with_mip))
            if self.with_mip:
                self.logger.info("MIP axis " + str(self.mip_axis))
                self.logger.info("floss coefficient " + str(self.floss_coeff))
                self.logger.info("mip loss coefficient " + str(self.mip_loss_coeff))

        # set probabilistic property
        if "Models.prob" in self.model.__module__:
            self.isProb = True
            from Models.prob_unet.utils import l2_regularisation
            self.l2_regularisation = l2_regularisation
        else:
            self.isProb = False

        if cmd_args.train:  # Only if training is to be performed
            if self.with_mip:
                # This is to configure training subjects with adapted SRDataset to collect co-ordinates of the patches
                # and their respective patches on MIP of the 3D label
                traindataset = SRDataset(logger=self.logger, patch_size=self.patch_size,
                                         dir_path=self.DATASET_FOLDER + '/train/',
                                         label_dir_path=self.DATASET_FOLDER + '/train_label/',
                                         # TODO: implement non-iso patch-size, now only using the first element
                                         stride_depth=self.stride_depth, stride_length=self.stride_length,
                                         stride_width=self.stride_width, size=None, fly_under_percent=None,
                                         # TODO: implement fly_under_percent, if needed
                                         patch_size_us=self.patch_size, pre_interpolate=None, norm_data=False,
                                         pre_load=True,
                                         return_coords=True)
                # TODO implement patch_size_us if required - patch_size//scaling_factor
                # Sampler reduces the 30200 samples generated from SRDataset to selected samples_per_epoch
                sampler = torch.utils.data.RandomSampler(data_source=traindataset, replacement=True,
                                                         num_samples=self.samples_per_epoch)
                self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size, shuffle=False,
                                                                num_workers=self.num_worker, pin_memory=True,
                                                                sampler=sampler)
            else:
                # Use torchio to generate subject dataset from training volumes
                traindataset = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_FOLDER + '/train/',
                                                          label_path=self.DATASET_FOLDER + '/train_label/',
                                                          logger=self.logger,
                                                          patch_size=self.patch_size,
                                                          stride_depth=self.stride_depth,
                                                          stride_width=self.stride_width,
                                                          stride_length=self.stride_length,
                                                          samples_per_epoch=self.samples_per_epoch,
                                                          is_train=True,
                                                          with_mip=False)
                self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.batch_size,
                                                                shuffle=True, num_workers=self.num_worker)
            # If training is with MIP, validation dataset is prepared similar to training with SRDataset
            # Otherwise torchio concat dataset will be prepared from validation volumes
            validationdataset = Pipeline.create_tio_sub_ds(vol_path=self.DATASET_FOLDER + '/validate/',
                                                           label_path=self.DATASET_FOLDER + '/validate_label/',
                                                           logger=self.logger,
                                                           patch_size=self.patch_size,
                                                           stride_depth=self.stride_depth,
                                                           stride_width=self.stride_width,
                                                           stride_length=self.stride_length,
                                                           samples_per_epoch=self.samples_per_epoch,
                                                           is_train=False,
                                                           with_mip=self.with_mip)
            self.validate_loader = torch.utils.data.DataLoader(validationdataset, batch_size=self.batch_size,
                                                               shuffle=False, num_workers=self.num_worker,
                                                               pin_memory=True)

    @staticmethod
    def create_tio_sub_ds(vol_path, label_path, logger, patch_size,
                          stride_depth, stride_length, stride_width, samples_per_epoch,
                          is_train=True, with_mip=False, get_subjects_only=False, is_eval=False):
        """
        Purpose: Creates 3D patches from 3D volumes using torchio and SRDataset
        :param vol_path: Path to 3D MRA volumes
        :param label_path: Path to 3D Ground Truth Segmentations
        :param logger: File logger
        :param patch_size: Individual patch dimension(Only patches of equal length, width and depth are created)
        :param stride_depth: Stride Depth for patch creation
        :param stride_length: Stride Length for patch creation
        :param stride_width: Stride Width for patch creation
        :param samples_per_epoch: Number of samples to be selected for each epoch
        :param is_train: When set, returns training dataset
        :param with_mip: When set, returns subjects(including 3D patches, their co-ordinates and
        corresponding label patches with and without MIP). Otherwise returns torchio subject dataset as
        patches queue in case of training and concat dataset in case of validation and testing.
        :param get_subjects_only: If set, returns subjects array with 4D volumes array(channel x width x depth x height)
        """
        if with_mip:
            # Use SRDataset to collect 3D patches, their co-ordinates, their corresponding label patches and
            # patches of MIP of 3D label
            subjects = SRDataset(logger=logger, patch_size=patch_size,
                                 dir_path=vol_path,
                                 label_dir_path=label_path,
                                 # TODO: implement non-iso patch-size, now only using the first element
                                 stride_depth=stride_depth, stride_length=stride_length,
                                 stride_width=stride_width, size=None, fly_under_percent=None,
                                 # TODO: implement fly_under_percent, if needed
                                 patch_size_us=patch_size, pre_interpolate=None, norm_data=False,
                                 pre_load=True,
                                 return_coords=True)
            # TODO implement patch_size_us if required - patch_size//scaling_factor
        else:
            # Iteratively construct torchio subject dataset with ScalarImages and labels and then create patches queue
            vols = glob(vol_path + "/*.nii") + glob(vol_path + "/*.nii.gz")
            if not is_eval:
                labels = glob(label_path + "*.nii") + glob(label_path + "*.nii.gz")
            subjects = []
            for i in range(len(vols)):
                v = vols[i]
                filename = os.path.basename(v).split('.')[0]
                if not is_eval:
                    l = [s for s in labels if filename in s][0]
                    subject = tio.Subject(
                        img=tio.ScalarImage(v),
                        label=tio.LabelMap(l),
                        subjectname=filename,
                    )
                else:
                    subject = tio.Subject(
                        img=tio.ScalarImage(v),
                        subjectname=filename,
                    )
                transforms = tio.ToCanonical(), tio.Resample(tio.ScalarImage(v))
                transform = tio.Compose(transforms)
                subject = transform(subject)
                subjects.append(subject)

        if get_subjects_only:
            return subjects

        if is_train and not with_mip:
            subjects_dataset = tio.SubjectsDataset(subjects)
            sampler = tio.data.UniformSampler(patch_size)
            patches_queue = tio.Queue(
                subjects_dataset,
                max_length=(samples_per_epoch // len(subjects)) * 2,
                samples_per_volume=samples_per_epoch // len(subjects),
                sampler=sampler,
                num_workers=0,
                start_background=True
            )
            return patches_queue
        else:
            overlap = np.subtract(patch_size, (stride_length, stride_width, stride_depth))
            grid_samplers = []
            for i in range(len(subjects)):
                grid_sampler = tio.inference.GridSampler(
                    subjects[i],
                    patch_size,
                    overlap,
                )
                grid_samplers.append(grid_sampler)
            return torch.utils.data.ConcatDataset(grid_samplers)

    @staticmethod
    def normaliser(batch):
        """
        Purpose: Normalise pixel intensities by comparing max values in the 3D patch
        :param batch: 5D array (batch_size x channel x width x depth x height)
        """
        for i in range(batch.shape[0]):
            batch[i] = batch[i] / batch[i].max()
        return batch

    def load(self, checkpoint_path=None, load_best=True, fold_index=""):
        """
        Purpose: Continue training from previous checkpoint or load an existing checkpoint for testing
        :param checkpoint_path: Path to the saved network state dictionary. If not specified, the path to checkpoint
        location of current directory is used.
        :param load_best: If set, uses best checkpoint from the checkpoint location. Otherwise uses last checkpoint.
        :param fold_index: If the model was trained with cross validation, the fold index is passed using this.
        """
        if checkpoint_path is None:
            checkpoint_path = self.checkpoint_path

        if self.with_apex:
            self.model, self.optimizer, self.scaler = load_model_with_amp(self.model, self.optimizer, checkpoint_path,
                                                                          batch_index="best" if load_best else "last",
                                                                          fold_index=fold_index)
        else:
            self.model, self.optimizer = load_model(self.model, self.optimizer, checkpoint_path,
                                                    batch_index="best" if load_best else "last", fold_index=fold_index)

    def train(self):
        """
        Purpose: Training Pipeline including Leave-One-Out Validation.
        Performs a variety of trainings specified by command line parameters.
        """
        self.logger.debug("Training...")

        training_batch_index = 0
        for epoch in range(self.num_epochs):
            print("Train Epoch: " + str(epoch) + " of " + str(self.num_epochs))
            self.model.train()  # make sure to assign mode:train, because in validation, mode is assigned as eval
            total_floss = 0
            total_mip_loss = 0
            total_loss = 0
            batch_index = 0
            for batch_index, patches_batch in enumerate(tqdm(self.train_loader)):

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()

                local_batch = torch.movedim(local_batch, -1, -3)
                local_labels = torch.movedim(local_labels, -1, -3)

                # Transfer to GPU
                self.logger.debug('Epoch: {} Batch Index: {}'.format(epoch, batch_index))

                # Clear gradients
                self.optimizer.zero_grad()

                # try:
                with autocast(enabled=self.with_apex):
                    loss_ratios = [1, 0.66, 0.34]  # TODO param

                    floss = torch.tensor(0.001).float().cuda()
                    mip_loss = torch.tensor(0.001).float().cuda()
                    output1 = 0
                    level = 0

                    # -------------------------------------------------------------------------------------------------
                    # First Branch Supervised error
                    if not self.isProb:
                        for output in self.model(local_batch):
                            if level == 0:
                                output1 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))
                            output = torch.sigmoid(output)
                            floss += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                            if self.with_mip:
                                # Compute MIP loss from the patch on the MIP of the 3D label and the patch prediction
                                mip_loss_patch = torch.tensor(0.001).float().cuda()
                                for idx, op in enumerate(output):
                                    if self.mip_axis == "multi":
                                        op_mip_z = torch.amax(op, -1)
                                        op_mip_y = torch.amax(op, 2)
                                        op_mip_x = torch.amax(op, 1)
                                        mip_loss_patch += self.mip_loss(op_mip_z,
                                                                        patches_batch['ground_truth_mip_z_patch']
                                                                        [idx].float().cuda()) + \
                                                          self.mip_loss(op_mip_y,
                                                                        patches_batch['ground_truth_mip_y_patch']
                                                                        [idx].float().cuda()) + \
                                                          self.mip_loss(op_mip_x,
                                                                        patches_batch['ground_truth_mip_x_patch']
                                                                        [idx].float().cuda())
                                    else:
                                        axis = -1
                                        if self.mip_axis == "x":
                                            axis = 1
                                        if self.mip_axis == "y":
                                            axis = 2
                                        op_mip = torch.amax(op, axis)
                                        mip_loss_patch += self.mip_loss(op_mip,
                                                                        patches_batch[str.format(
                                                                            'ground_truth_mip_{}_patch', self.mip_axis)]
                                                                        [idx].float().cuda())
                                mip_loss += loss_ratios[level] * (mip_loss_patch / len(output))
                            level += 1
                    else:
                        self.model.forward(local_batch, local_labels, training=True)
                        elbo = self.model.elbo(local_labels, analytic_kl=True)
                        reg_loss = self.l2_regularisation(self.model.posterior) + self.l2_regularisation(
                            self.model.prior) + self.l2_regularisation(self.model.fcomb.layers)
                        if self.with_apex:
                            floss = [
                                self.model.mean_reconstruction_loss if self.model.use_mean_recon_loss else self.model.reconstruction_loss,
                                -(self.model.beta * self.model.kl),
                                self.model.reg_alpha * reg_loss]
                        else:
                            floss = -elbo + self.model.reg_alpha * reg_loss

                    # Elastic Deformations
                    if self.deform:
                        # Each batch must be randomly deformed
                        elastic = RandomElasticDeformation(
                            num_control_points=random.choice([5, 6, 7]),
                            max_displacement=random.choice([0.01, 0.015, 0.02, 0.025, 0.03]),
                            locked_borders=2
                        )
                        elastic.cuda()

                        with autocast(enabled=False):
                            local_batch_xt, displacement, _ = elastic(local_batch)
                            local_labels_xt = warp_image(local_labels, displacement, multi=True)
                        floss2 = 0

                        level = 0
                        # ------------------------------------------------------------------------------
                        # Second Branch Supervised error
                        for output in self.model(local_batch_xt):
                            if level == 0:
                                output2 = output
                            if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))

                            output = torch.sigmoid(output)
                            floss2 += loss_ratios[level] * self.focalTverskyLoss(output, local_labels_xt)
                            level += 1

                        # -------------------------------------------------------------------------------------------
                        # Consistency loss
                        with autocast(enabled=False):
                            output1T = warp_image(output1.float(), displacement, multi=True)
                        floss_c = self.focalTverskyLoss(torch.sigmoid(output2), output1T)

                        # -------------------------------------------------------------------------------------------
                        # Total loss
                        floss = floss + floss2 + floss_c

                    loss = (self.floss_coeff * floss) + (self.mip_loss_coeff * mip_loss)

                self.logger.info("Epoch:" + str(epoch) + " Batch_Index:" + str(batch_index) + "Training..." +
                                 "\n focalTverskyLoss:" + str(floss) + "mipLoss: " + str(mip_loss))

                # Calculating gradients
                if self.with_apex:
                    if type(loss) is list:
                        for i in range(len(loss)):
                            if i + 1 == len(loss):  # final loss
                                self.scaler.scale(loss[i]).backward()
                            else:
                                self.scaler.scale(loss[i]).backward(retain_graph=True)
                        loss = torch.sum(torch.stack(loss))
                    else:
                        self.scaler.scale(loss).backward()

                    if self.clip_grads:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    if self.clip_grads:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
                        # torch.nn.utils.clip_grad_value_(self.model.parameters(), 1)

                    self.optimizer.step()

                if training_batch_index % 50 == 0:  # Save best metric evaluation weights
                    write_summary(self.writer_training, self.logger, training_batch_index,
                                  focal_tversky_loss=floss, mip_loss=mip_loss, total_loss=loss)
                training_batch_index += 1

                # Initialising the average loss metrics
                total_floss += floss.detach().item()
                total_mip_loss += mip_loss.detach().item()
                total_loss += loss.detach().item()

                if self.deform:
                    del elastic
                    torch.cuda.empty_cache()

            # Calculate the average loss per batch in one epoch
            total_floss /= (batch_index + 1.0)
            total_mip_loss /= (batch_index + 1.0)
            total_loss /= (batch_index + 1.0)

            # Print every epoch
            self.logger.info("Epoch:" + str(epoch) + " Average Training..." +
                             "\n focalTverskyLoss:" + str(total_floss)
                             + "mipLoss: " + str(total_mip_loss)
                             + "totalLoss: " + str(total_loss))
            write_epoch_summary(self.writer_training, epoch, focal_tversky_loss=total_floss, mip_loss=total_mip_loss,
                                total_loss=total_loss)
            if self.wandb is not None:
                self.wandb.log({"focalTverskyLoss_train": total_floss, "mipLoss_train": total_mip_loss,
                                "totalLoss_train": total_loss})

            save_model(self.checkpoint_path, {
                'epoch_type': 'last',
                'epoch': epoch,
                # Let is always overwrite, we need just the last checkpoint and best checkpoint(saved after validate)
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()
            })

            torch.cuda.empty_cache()  # to avoid memory errors
            self.validate(training_batch_index, epoch)
            torch.cuda.empty_cache()  # to avoid memory errors

        return self.model

    def validate(self, training_index, epoch):
        """
        Purpose: Method to perform Leave-One-Out validation
        :param training_index: Epoch after which validation is performed(can be anything for test)
        :param epoch: current training epoch
        """
        self.logger.debug('Validating...')
        print("Validate Epoch: " + str(epoch) + " of " + str(self.num_epochs))

        floss, mip_loss, loss, binloss, dloss, dscore, jaccard_index = 0, 0, 0, 0, 0, 0, 0
        no_patches = 0
        self.model.eval()
        data_loader = self.validate_loader
        writer = self.writer_validating
        with torch.no_grad():
            for index, patches_batch in enumerate(tqdm(data_loader)):
                self.logger.info("loading" + str(index))
                no_patches += 1

                local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                local_labels = patches_batch['label'][tio.DATA].float().cuda()

                local_batch = torch.movedim(local_batch, -1, -3)
                local_labels = torch.movedim(local_labels, -1, -3)

                floss_iter = 0
                mip_loss_iter = 0
                output1 = 0
                try:
                    with autocast(enabled=self.with_apex):
                        # Forward propagation
                        loss_ratios = [1, 0.66, 0.34]  # TODO param
                        level = 0

                        # Forward propagation
                        if not self.isProb:
                            for output in self.model(local_batch):
                                if level == 0:
                                    output1 = output
                                if level > 0:  # then the output size is reduced, and hence interpolate to patch_size
                                    output = torch.nn.functional.interpolate(input=output, size=(64, 64, 64))
                                output = torch.sigmoid(output)
                                floss_iter += loss_ratios[level] * self.focalTverskyLoss(output, local_labels)
                                if self.with_mip:
                                    # Compute MIP loss from the patch on the MIP of the 3D label
                                    # and the patch prediction
                                    mip_loss_patch = torch.tensor(0.001).float().cuda()
                                    for idx, op in enumerate(output):
                                        if self.mip_axis == "multi":
                                            op_mip_z = torch.amax(op, -1)
                                            op_mip_y = torch.amax(op, 2)
                                            op_mip_x = torch.amax(op, 1)
                                            mip_loss_patch += self.mip_loss(op_mip_z,
                                                                            patches_batch['ground_truth_mip_z_patch']
                                                                            [idx].float().cuda()) + \
                                                              self.mip_loss(op_mip_y,
                                                                            patches_batch['ground_truth_mip_y_patch']
                                                                            [idx].float().cuda()) + \
                                                              self.mip_loss(op_mip_x,
                                                                            patches_batch['ground_truth_mip_x_patch']
                                                                            [idx].float().cuda())
                                        else:
                                            axis = -1
                                            if self.mip_axis == "x":
                                                axis = 1
                                            if self.mip_axis == "y":
                                                axis = 2
                                            op_mip = torch.amax(op, axis)
                                            mip_loss_patch += self.mip_loss(op_mip,
                                                                            patches_batch[str.format(
                                                                                'ground_truth_mip_{}_patch',
                                                                                self.mip_axis)]
                                                                            [idx].float().cuda())
                                    mip_loss_iter += loss_ratios[level] * (mip_loss_patch / len(output))
                                level += 1
                        else:
                            self.model.forward(local_batch, training=False)
                            output1 = torch.sigmoid(self.model.sample(testing=True))
                            floss_iter = self.focalTverskyLoss(output1, local_labels)
                except Exception as error:
                    self.logger.exception(error)

                floss += floss_iter
                mip_loss += mip_loss_iter
                loss = floss + mip_loss
                dl, ds = self.dice(torch.sigmoid(output1), local_labels)
                dloss += dl.detach().item()

        # Average the losses
        floss = floss / no_patches
        mip_loss = mip_loss / no_patches
        loss = loss / no_patches
        dloss = dloss / no_patches
        process = ' Validating'
        self.logger.info("Epoch:" + str(training_index) + process + "..." +
                         "\n FocalTverskyLoss:" + str(floss) +
                         "\n DiceLoss:" + str(dloss)
                         + "\n mipLoss:" + str(mip_loss)
                         + "\n totalLoss: " + str(loss))

        write_summary(writer, self.logger, training_index,
                      focal_tversky_loss=floss, mip_loss=mip_loss, total_loss=loss, dice_loss=dloss)
        write_epoch_summary(writer, epoch, focal_tversky_loss=floss, mip_loss=mip_loss, total_loss=loss)
        if self.wandb is not None:
            self.wandb.log({"focalTverskyLoss_val": floss, "mipLoss_val": mip_loss, "totalLoss_val": loss})

        if self.LOWEST_LOSS > floss:  # Save best metric evaluation weights
            self.LOWEST_LOSS = floss
            self.logger.info(
                'Best metric... @ epoch:' + str(training_index) + ' Current Lowest loss:' + str(self.LOWEST_LOSS))

            save_model(self.checkpoint_path, {
                'epoch_type': 'best',
                'epoch': epoch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'amp': self.scaler.state_dict()})

    def test(self, test_logger, save_results=True, test_subjects=None):
        """
        Purpose: Performs Generalization Performance Testing of the given model along with
        inference prediction on specified 3D MRA volume.
        :param test_logger: File logger
        :param save_results: If set, saves resulting segmentation as .nii.gz, the color MIP overlay of comparison of
        segmentation prediction against the ground truth.
        :param test_subjects: If specified, uses the given array of subjects(torchio subjects array) to
        prepare test loader. Otherwise creates test subjects from '/test/' and '/test_label/' folders.
        """
        test_logger.debug('Testing...')

        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER + '/test/'
            test_label_path = self.DATASET_FOLDER + '/test_label/'
            test_subjects = Pipeline.create_tio_sub_ds(vol_path=test_folder_path,
                                                       label_path=test_label_path,
                                                       logger=test_logger,
                                                       patch_size=self.patch_size,
                                                       stride_width=self.stride_width,
                                                       stride_depth=self.stride_depth,
                                                       stride_length=self.stride_length,
                                                       samples_per_epoch=self.samples_per_epoch,
                                                       with_mip=False,
                                                       is_train=False,
                                                       get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))

        df = pd.DataFrame(columns=["Subject", "Dice", "IoU"])
        result_root = os.path.join(self.output_path, self.model_name, "results")
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                affine = test_subject['img'][tio.AFFINE]
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = Pipeline.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1).type(local_batch.type())
                    aggregator.add_batch(output, locations)

                predicted = aggregator.get_output_tensor().squeeze().numpy()

                try:
                    thresh = threshold_otsu(predicted)
                    result = predicted > thresh
                except Exception as error:
                    test_logger.exception(error)
                    result = predicted > 0.5
                    # exception will be thrown only if input image seems to have just one color 1.0.
                result = result.astype(np.float32)

                if label is not None:
                    datum = {"Subject": subjectname}
                    dice3d = dice(result, label)
                    iou3d = iou(result, label)
                    datum = pd.DataFrame.from_dict({**datum, "Dice": [dice3d], "IoU": [iou3d]})
                    df = pd.concat([df, datum], ignore_index=True)

                if save_results:
                    save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"), affine=affine)

                    result_mip = np.max(result, axis=-1)
                    Image.fromarray((result_mip * 255).astype('uint8'), 'L').save(
                        os.path.join(result_root, subjectname + "_MIP.tif"))

                    if label is not None:
                        overlay = create_diff_mask_binary(result, label)
                        save_tif_rgb(overlay, os.path.join(result_root, subjectname + "_colour.tif"))

                        overlay_mip = create_diff_mask_binary(result_mip, np.max(label, axis=-1))
                        color_mip = Image.fromarray(overlay_mip.astype('uint8'), 'RGB')
                        color_mip.save(
                            os.path.join(result_root, subjectname + "_colourMIP.tif"))
                        if self.wandb is not None:
                            self.wandb.log({"" + subjectname: self.wandb.Image(color_mip)})

                test_logger.info("Testing " + subjectname + "..." +
                                 "\n Dice:" + str(dice3d) +
                                 "\n JacardIndex:" + str(iou3d))
                if self.wandb is not None:
                    self.wandb.log({"subjectname": subjectname, "Dice": dice3d, "JacardIndex": iou3d})

        df.to_excel(os.path.join(result_root, "Results_Main.xlsx"))

    def eval(self, test_subjects=None, model_name="mip_model1"):
        """
        Purpose: Render inference on nifti 3D volumes specified using pretrained UNet-MSS network
        :param test_subjects: Optionally an array of tio.subjects can be passed on which the inference will be to drawn
        :param model_name: name of the results folder at output path
        """
        if test_subjects is None:
            test_folder_path = self.DATASET_FOLDER
            test_label_path = None  # self.DATASET_FOLDER + '/test_label/'
            test_subjects = Pipeline.create_tio_sub_ds(vol_path=test_folder_path,
                                                       label_path=test_label_path,
                                                       logger=None,
                                                       patch_size=self.patch_size,
                                                       stride_width=self.stride_width,
                                                       stride_depth=self.stride_depth,
                                                       stride_length=self.stride_length,
                                                       samples_per_epoch=self.samples_per_epoch,
                                                       with_mip=False,
                                                       is_train=False,
                                                       is_eval=True,
                                                       get_subjects_only=True)

        overlap = np.subtract(self.patch_size, (self.stride_length, self.stride_width, self.stride_depth))
        result_root = os.path.join(self.output_path, model_name)
        os.makedirs(result_root, exist_ok=True)

        self.model.eval()

        with torch.no_grad():
            for test_subject in test_subjects:
                affine = test_subject['img'][tio.AFFINE]
                if 'label' in test_subject:
                    label = test_subject['label'][tio.DATA].float().squeeze().numpy()
                    del test_subject['label']
                else:
                    label = None
                subjectname = test_subject['subjectname']
                del test_subject['subjectname']

                grid_sampler = tio.inference.GridSampler(
                    test_subject,
                    self.patch_size,
                    overlap,
                )
                aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
                patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=self.batch_size, shuffle=False,
                                                           num_workers=self.num_worker)

                for index, patches_batch in enumerate(tqdm(patch_loader)):
                    local_batch = self.normaliser(patches_batch['img'][tio.DATA].float().cuda())
                    locations = patches_batch[tio.LOCATION]

                    local_batch = torch.movedim(local_batch, -1, -3)

                    with autocast(enabled=self.with_apex):
                        if not self.isProb:
                            output = self.model(local_batch)
                            if type(output) is tuple or type(output) is list:
                                output = output[0]
                            output = torch.sigmoid(output).detach().cpu()
                        else:
                            self.model.forward(local_batch, training=False)
                            output = self.model.sample(
                                testing=True).detach().cpu()  # TODO: need to check whether sigmoid is needed for prob

                    output = torch.movedim(output, -3, -1).type(local_batch.type())
                    aggregator.add_batch(output, locations)

                predicted = aggregator.get_output_tensor().squeeze().numpy()

                try:
                    thresh = threshold_otsu(predicted)
                    result = predicted > thresh
                except Exception as error:
                    print(error)
                    result = predicted > 0.5  # exception only if input image seems to have just one color 1.0.
                result = result.astype('uint16')

                save_nifti(result, os.path.join(result_root, subjectname + ".nii.gz"), affine=affine)

    def predict(self, image_path, label_path, predict_logger):
        """
        Purpose: Predicts segmentation for given 3D MRA Volume
        :param image_path: Path to 3D nifti MRA volume in .nii or .nii.gz format
        :param label_path: Path to 3D Ground Truth Segmentation in .nii or .nii.gz format
        :param predict_logger: File logger
        """
        image_name = os.path.basename(image_path).split('.')[0]

        subdict = {
            "img": tio.ScalarImage(image_path),
            "subjectname": image_name,
        }

        if bool(label_path):
            subdict["label"] = tio.LabelMap(label_path)

        subject = tio.Subject(**subdict)

        self.test(predict_logger, save_results=True, test_subjects=[subject])
