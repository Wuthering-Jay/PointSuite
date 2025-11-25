import torch
import pytorch_lightning as pl

# 加载ckpt文件查看内容
ckpt_path = r"E:\code\PointSuite\outputs\dales\csv_logs\version_48\checkpoints\dales-epoch=01-mean_iou=0.6521.ckpt"
checkpoint = torch.load(ckpt_path, weights_only=False)
print("Checkpoint keys:", checkpoint.keys())
print("Hyperparameters:", checkpoint['hyper_parameters'])
print("hyper_parameters class names:", checkpoint['hyper_parameters'].get("class_names")[1])
        