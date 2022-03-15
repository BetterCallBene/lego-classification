# %%
#%matplotlib inline
import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.config import get_cfg

#import matplotlib.pyplot as plt
#import cv2


# %%
dataset_name = "lego-classification"

# %%
register_coco_instances(dataset_name, {}, "./data/train-2.json", "./data/images")

# %%
lego_classificiation_metadata = MetadataCatalog.get(dataset_name)
dataset_dicts = DatasetCatalog.get(dataset_name)

# %%
import random
from detectron2.utils.visualizer import Visualizer

# %%
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=lego_classificiation_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     plt.imshow(vis.get_image()[:, :, ::-1])

# %%


cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = (dataset_name,)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 500    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512   # faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(lego_classificiation_metadata.thing_classes)  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.INPUT.MIN_SIZE_TRAIN = (300, 300, 300)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.



# %%
from detectron2.engine import DefaultTrainer
import torch
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()


