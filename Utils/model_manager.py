#!/usr/bin/env python
"""

Purpose : model selector

"""
from Models.attentionunet3d import AttUNet
from Models.prob_unet.probabilistic_unet import ProbabilisticUnet
from Models.unet3d import UNet, UNetDeepSup

__author__ = "Kartik Prabhu, Mahantesh Pattadkal, and Soumick Chatterjee"
__copyright__ = "Copyright 2020, Faculty of Computer Science, Otto von Guericke University Magdeburg, Germany"
__credits__ = ["Kartik Prabhu", "Mahantesh Pattadkal", "Soumick Chatterjee"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Soumick Chatterjee"
__email__ = "soumick.chatterjee@ovgu.de"
__status__ = "Production"

MODEL_UNET = 1
MODEL_UNET_DEEPSUP = 2
MODEL_ATTENTION_UNET = 3
MODEL_PROBABILISTIC_UNET = 4


def get_model(model_no):  # Send model params from outside
    default_model = UNet()  # Default
    model_list = {
        1: UNet(),
        2: UNetDeepSup(),
        3: AttUNet(),
        4: ProbabilisticUnet(num_filters=[32, 64, 128, 192])
        # 4: ProbabilisticUnet(num_filters=[64,128,256,512,1024])
    }
    return model_list.get(model_no, default_model)
