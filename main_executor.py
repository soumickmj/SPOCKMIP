#!/usr/bin/env python
"""
Purpose: Entry Point for the Vessel Segmentation Solution
"""

import argparse
import random
import numpy as np
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from Utils.logger import Logger
from Utils.model_manager import get_model
from pipeline import Pipeline
from cross_validation_pipeline import CrossValidationPipeline

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(2020)
np.random.seed(2020)
random.seed(2020)

# torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model",
                        type=int,
                        default=2,
                        help="1{U-Net}; \n"
                             "2{U-Net_Deepsup}; \n"
                             "3{Attention-U-Net}; \n"
                             "4{Probabilistic-U-Net};")
    parser.add_argument("-model_name",
                        default="Model_MIP_v1",
                        help="Name of the model")
    parser.add_argument("-dataset_path",
                        default="",
                        help="Path to folder containing dataset."
                             "If cross validation is to be performed, please create the dataset folder consisting of"
                             "train and train_label folders along with test and test_label folders"
                             "e.g."
                             "/sample_dataset"
                             "  /train"
                             "  /train_label"
                             "  /test"
                             "  /test_label"
                             "else if training is to be performed, prepare the dataset folder consisting of"
                             "train, train_label, validate, validate_label, test and test_label folders"
                             "e.g.,"
                             "/sample_dataset"
                             "  /train"
                             "  /train_label"
                             "  /validate"
                             "  /validate_label"
                             "  /test"
                             "  /test_label"
                             "each folder must contain at least one 3D MRA volume in nifti .nii or nii.gz formats"
                             "else if inference is to be performed, specify folder consisting of"
                             "3D MRA volume in nifti .nii or nii.gz formats"
                             "Example: /home/dataset/")
    parser.add_argument("-output_path",
                        default="",
                        help="Folder path to store output "
                             "Example: /home/output/")

    parser.add_argument('-train',
                        default=False,
                        help="To train the model")
    parser.add_argument('-test',
                        default=False,
                        help="To test the model")
    parser.add_argument('-eval',
                        default=False,
                        help="To render inference of specified nifti 3D volumes")
    parser.add_argument('-cross_validate',
                        default=False,
                        help="To train with k-fold cross validation")
    parser.add_argument('-predict',
                        default=False,
                        help="To predict a segmentation output of the model and to get a diff between label and output")
    parser.add_argument('-predictor_path',
                        default="",
                        help="Path to the input image to predict an output, ex:/home/test/ww25.nii ")
    parser.add_argument('-predictor_label_path',
                        default="",
                        help="Path to the label image to find the diff between label an output, "
                             "e.g.,:/home/test/ww25_label.nii ")
    parser.add_argument('-load_path',
                        default="./",
                        help="Path to checkpoint of existing model to load, ex:/home/model/checkpoint")
    parser.add_argument('-load_best',
                        default=True,
                        help="Specifiy whether to load the best checkpoint or the last. "
                             "Also to be used if Train and Test both are true.")
    parser.add_argument('-pre_train',
                        default=True,
                        help="Specifiy whether to load the pre trained weights for training/tuning.")
    parser.add_argument('-deform',
                        default=False,
                        help="To use deformation for training")
    parser.add_argument('-clip_grads',
                        default=True,
                        help="To use deformation for training")
    parser.add_argument('-apex',
                        default=True,
                        help="To use half precision on model weights.")
    parser.add_argument('-with_mip',
                        default=True,
                        help="Train with MIP Loss")
    parser.add_argument('-use_madam',
                        default=False,
                        help="Set this to True to use madam optimizer")

    parser.add_argument("-batch_size",
                        type=int,
                        default=15,
                        help="Batch size for training")
    parser.add_argument("-num_epochs",
                        type=int,
                        default=50,
                        help="Number of epochs for training")
    parser.add_argument("-learning_rate",
                        type=float,
                        default=0.0001,
                        help="Learning rate")
    parser.add_argument("-patch_size",
                        type=int,
                        default=64,
                        help="Patch size of the input volume")
    parser.add_argument("-stride_depth",
                        type=int,
                        default=16,
                        help="Strides for dividing the input volume into patches in depth dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-stride_width",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in width dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-stride_length",
                        type=int,
                        default=32,
                        help="Strides for dividing the input volume into patches in length dimension "
                             "(To be used during validation and inference)")
    parser.add_argument("-samples_per_epoch",
                        type=int,
                        default=8000,
                        help="Number of samples per epoch")
    parser.add_argument("-num_worker",
                        type=int,
                        default=8,
                        help="Number of worker threads")
    parser.add_argument("-floss_coeff",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss in total loss")
    parser.add_argument("-mip_loss_coeff",
                        type=float,
                        default=0.3,
                        help="Loss coefficient for mip_loss in total loss")
    parser.add_argument("-floss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for floss_param_smooth")
    parser.add_argument("-floss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for floss_param_gamma")
    parser.add_argument("-floss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for floss_param_alpha")
    parser.add_argument("-mip_loss_param_smooth",
                        type=float,
                        default=1,
                        help="Loss coefficient for mip_loss_param_smooth")
    parser.add_argument("-mip_loss_param_gamma",
                        type=float,
                        default=0.75,
                        help="Loss coefficient for mip_loss_param_gamma")
    parser.add_argument("-mip_loss_param_alpha",
                        type=float,
                        default=0.7,
                        help="Loss coefficient for mip_loss_param_alpha")
    parser.add_argument("-mip_axis",
                        type=str,
                        default="z",
                        help="Set projection axis. Default is z-axis. use axis in [x, y, z] or 'multi'")
    parser.add_argument("-k_folds",
                        type=int,
                        default=5,
                        help="Set the number of folds for cross validation")
    parser.add_argument("-fold_index",
                        type=str,
                        default="",
                        help="Set the number of folds for cross validation")

    parser.add_argument("-wandb",
                        default=False,
                        help="Set this to true to include wandb logging")
    parser.add_argument("-wandb_project",
                        type=str,
                        default="",
                        help="Set this to wandb project name e.g., 'DS6_VesselSeg2'")
    parser.add_argument("-wandb_entity",
                        type=str,
                        default="",
                        help="Set this to wandb project name e.g., 'ds6_vessel_seg2'")
    parser.add_argument("-wandb_api_key",
                        type=str,
                        default="",
                        help="API Key to login that can be found at https://wandb.ai/authorize")

    args = parser.parse_args()

    if str(args.deform).lower() == "true":
        args.model_name += "_Deform"

    MODEL_NAME = args.model_name
    DATASET_FOLDER = args.dataset_path
    OUTPUT_PATH = args.output_path

    LOAD_PATH = args.load_path
    CHECKPOINT_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '/checkpoint/'

    if str(args.eval).lower() == "true":
        TENSORBOARD_PATH_TRAINING = None
        TENSORBOARD_PATH_VALIDATION = None
        TENSORBOARD_PATH_TESTING = None
        LOGGER_PATH = None
        logger = None
        test_logger = None
        writer_training = None
        writer_validating = None

    else:
        TENSORBOARD_PATH_TRAINING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_training/'
        TENSORBOARD_PATH_VALIDATION = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_validation/'
        TENSORBOARD_PATH_TESTING = OUTPUT_PATH + "/" + MODEL_NAME + '/tensorboard/tensorboard_testing/'

        LOGGER_PATH = OUTPUT_PATH + "/" + MODEL_NAME + '.log'

        logger = Logger(MODEL_NAME, LOGGER_PATH).get_logger()
        test_logger = Logger(MODEL_NAME + '_test', LOGGER_PATH).get_logger()

        writer_training = SummaryWriter(TENSORBOARD_PATH_TRAINING)
        writer_validating = SummaryWriter(TENSORBOARD_PATH_VALIDATION)

    # Model
    model = torch.nn.DataParallel(get_model(args.model))
    model.cuda()

    wandb = None
    if str(args.wandb).lower() == "true":
        import wandb
        wandb.login(key=args.wandb_api_key)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, group=args.model_name, notes=args.model_name)
        wandb.config = {
            "learning_rate": args.learning_rate,
            "epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "patch_size": args.patch_size,
            "samples_per_epoch": args.samples_per_epoch,
            "mip_loss_coeff": args.mip_loss_coeff,
            "floss_coeff": args.floss_coeff
        }

    # Choose pipeline based on whether or not to perform cross validation
    # If cross validation is to be performed, please create the dataset folder consisting of
    # train and train_label folders along with test and test_label folders
    # e.g.,
    # /sample_dataset
    #   /train
    #   /train_label
    #   /test
    #   /test_label
    # Otherwise prepare the dataset folder consisting of
    # train, train_label, validate, validate_label, test and test_label folders
    # e.g.,
    # /sample_dataset
    #   /train
    #   /train_label
    #   /validate
    #   /validate_label
    #   /test
    #   /test_label
    # Each folder must contain at least one 3D MRA volume in nifti .nii or nii.gz formats
    if str(args.cross_validate).lower() == "true":
        pipeline = CrossValidationPipeline(cmd_args=args, model=model, logger=logger,
                                           dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                                           writer_training=writer_training, writer_validating=writer_validating,
                                           test_logger=test_logger, wandb=wandb)
    else:
        pipeline = Pipeline(cmd_args=args, model=model, logger=logger,
                            dir_path=DATASET_FOLDER, checkpoint_path=CHECKPOINT_PATH,
                            writer_training=writer_training, writer_validating=writer_validating, wandb=wandb)

    # loading existing checkpoint if supplied
    if str(args.pre_train).lower() == "true":
        pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best, fold_index=args.fold_index)

    if str(args.train).lower() == "true":
        pipeline.train()
        torch.cuda.empty_cache()  # to avoid memory errors

    if str(args.test).lower() == "true":
        if str(args.load_best).lower() == "true":
            if bool(LOAD_PATH):
                pipeline.load(checkpoint_path=LOAD_PATH, load_best=args.load_best, fold_index=args.fold_index)
            else:
                pipeline.load(load_best=args.load_best)
        pipeline.test(test_logger=test_logger, fold_index=args.fold_index)
        torch.cuda.empty_cache()  # to avoid memory errors

    if str(args.eval).lower() == "true":
        pipeline.eval(model_name=args.model_name)

    if str(args.predict).lower() == "true":
        pipeline.predict(args.predictor_path, args.predictor_label_path, predict_logger=test_logger)

    if str(args.eval).lower() != "true":
        writer_training.close()
        writer_validating.close()
