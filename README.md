# FusionLane

Source code for the paper [*FusionLane: Multi-Sensor Fusion for Lane Marking Semantic Segmentation Using Deep Neural Networks*](https://ieeexplore.ieee.org/document/9237136).

## Original paper code

The files in this directory (`model_pt.py`, `train_pt.py`, `dataset_pt.py`, `infer_pt.py`, `xception.py`, etc.) are the original TensorFlow/PyTorch implementation as described in the paper.

**Requirements:** TensorFlow >= 1.12, NumPy, matplotlib, opencv-python

## Improved version

See [fusionlane_improved/](fusionlane_improved/) for a cleaned-up PyTorch re-implementation that adds:

- ImageNet normalization and data augmentation
- Hybrid Cross-Entropy + Dice loss
- Confidence-filtered post-processing and morphological cleanup
- LR scheduling, gradient clipping, and early stopping
- Support for raw road images and dashcam video (no dataset required)

Full documentation, setup instructions, and a Works Cited section are in [fusionlane_improved/README.md](fusionlane_improved/README.md).

## Works Cited

```
@article{yin2020fusionlane,
  title={Fusionlane: Multi-sensor fusion for lane marking semantic segmentation using deep neural networks},
  author={Yin, Ruochen and Cheng, Yong and Wu, Huapeng and Song, Yuntao and Yu, Biao and Niu, Runxin},
  journal={IEEE Transactions on Intelligent Transportation Systems},
  volume={23},
  number={2},
  pages={1543--1553},
  year={2020},
  publisher={IEEE}
}
```
