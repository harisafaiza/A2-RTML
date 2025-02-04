# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/12mSsZ62ai_Ru9nxtW38JZn1e_n_mhLDv
"""

import torch.nn as nn

class Darknet(nn.Module):
    def __init__(self, config_file):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(config_file)
        self.net_info, self.module_list = self.create_modules(self.blocks)

    def forward(self, x, CUDA=False):
        outputs = {}
        detections = None

        for i, module in enumerate(self.blocks):
            if module["type"] == "convolutional" or module["type"] == "upsample":
                x = self.module_list[i](x)
            elif module["type"] == "route":
                layers = [int(a) for a in module["layers"].split(',')]
                x = torch.cat([outputs[i + layer] for layer in layers], 1)
            elif module["type"] == "yolo":
                anchors = self.module_list[i][0].anchors
                x = predict_transform(x, self.net_info["height"], anchors, len(module["mask"]))
                detections = x if detections is None else torch.cat((detections, x), 1)
            outputs[i] = x
        return detections