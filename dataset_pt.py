"""
FusionLane dataset for PyTorch.

Uses TensorFlow (CPU-only, for TFRecord parsing only) to read the data,
then yields plain numpy arrays that PyTorch consumes.

Must be used with DataLoader(num_workers=0) because TF sessions are not
fork-safe across worker processes.

Each yielded sample:
    image  : FloatTensor [3, 321, 321]   raw pixel values in [0, 255]
    region : FloatTensor [1, 321, 321]
    label  : LongTensor  [321, 321]      class index; 20 = ignored/void
"""

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Suppress TF info/warning noise (only errors shown)
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

_H = 321
_W = 321
_IGNORE = 20   # void label value


# ---------------------------------------------------------------------------
# TFRecord helpers (pure TF graph ops)
# ---------------------------------------------------------------------------

def _filenames(is_training, data_dir):
    prefix = 'train' if is_training else 'testing'
    return [os.path.join(data_dir, f'{prefix}-0000{i}-of-00004.tfrecord')
            for i in range(4)]


def _build_tf_dataset(is_training, data_dir, cp_x, cp_y):
    """
    Builds and returns a TF Dataset of (image[H,W,3], region[H,W,1], label[H,W,1])
    as float32 / int32 numpy-ready tensors, cropped to [_H, _W].
    """

    def _parse(raw):
        feat = {
            'image/encoded':  tf.FixedLenFeature((), tf.string, ''),
            'region/encoded': tf.FixedLenFeature((), tf.string, ''),
            'label/encoded':  tf.FixedLenFeature((), tf.string, ''),
        }
        p = tf.parse_single_example(raw, feat)

        img = tf.image.decode_image(tf.reshape(p['image/encoded'],  []), 3)
        img = tf.cast(tf.image.convert_image_dtype(img, tf.uint8), tf.float32)
        img.set_shape([None, None, 3])

        reg = tf.image.decode_image(tf.reshape(p['region/encoded'], []), 1)
        reg = tf.cast(tf.image.convert_image_dtype(reg, tf.uint8), tf.float32)
        reg.set_shape([None, None, 1])

        lbl = tf.image.decode_image(tf.reshape(p['label/encoded'],  []), 1)
        lbl = tf.cast(tf.image.convert_image_dtype(lbl, tf.uint8), tf.int32)
        lbl.set_shape([None, None, 1])

        return img, reg, lbl

    def _crop(img, reg, lbl):
        # Shift label so that 0-padding doesn't collide with valid classes
        lbl_f = tf.cast(lbl, tf.float32) - _IGNORE

        # Stack to one tensor for atomic crop
        combined = tf.concat([img, reg, lbl_f], axis=2)   # [H, W, 5]
        h = tf.shape(combined)[0]
        w = tf.shape(combined)[1]

        # Pad to at least (crop_h, crop_w) if source is smaller
        combined = tf.image.pad_to_bounding_box(
            combined, 0, 0,
            tf.maximum(_H, h),
            tf.maximum(_W, w),
        )

        # Deterministic crop from (cp_x, cp_y)
        combined = tf.cast(
            tf.image.crop_to_bounding_box(combined, cp_x, cp_y, _H, _W),
            tf.float32,
        )
        combined.set_shape([_H, _W, 5])

        img_c  = combined[:, :, :3]                             # [H, W, 3]
        reg_c  = combined[:, :, 3:4]                            # [H, W, 1]
        lbl_c  = tf.cast(combined[:, :, 4:5] + _IGNORE, tf.int32)  # [H, W, 1]

        return img_c, reg_c, lbl_c

    files = _filenames(is_training, data_dir)
    ds = (tf.data.Dataset
          .from_tensor_slices(files)
          .flat_map(tf.data.TFRecordDataset)
          .map(_parse, num_parallel_calls=1)
          .map(_crop,  num_parallel_calls=1))

    return ds


# ---------------------------------------------------------------------------
# PyTorch IterableDataset
# ---------------------------------------------------------------------------

class FusionLaneDataset(IterableDataset):
    """
    Parameters
    ----------
    data_dir    : path to folder containing the *.tfrecord files
    is_training : True for train split, False for test/val split
    cp_x, cp_y  : top-left corner of the crop window (randomised each epoch
                  in the training loop)
    """

    def __init__(self, data_dir, is_training, cp_x=0, cp_y=0):
        self.data_dir    = data_dir
        self.is_training = is_training
        self.cp_x        = cp_x
        self.cp_y        = cp_y

    def __iter__(self):
        # Build a fresh TF graph + session for each iteration.
        # (make_one_shot_iterator is exhausted after one pass — we just
        #  recreate it, which is safe with num_workers=0.)
        graph = tf.Graph()
        with graph.as_default():
            ds   = _build_tf_dataset(self.is_training, self.data_dir,
                                     self.cp_x, self.cp_y)
            it   = ds.make_one_shot_iterator()
            next_op = it.get_next()

        config = tf.ConfigProto()
        config.gpu_options.visible_device_list = ''   # CPU only for data loading
        with tf.Session(graph=graph, config=config) as sess:
            while True:
                try:
                    img_np, reg_np, lbl_np = sess.run(next_op)
                    # HWC → CHW, numpy → torch
                    img_t = torch.from_numpy(
                        img_np.transpose(2, 0, 1).copy())   # [3, H, W]  float32
                    reg_t = torch.from_numpy(
                        reg_np.transpose(2, 0, 1).copy())   # [1, H, W]  float32
                    lbl_t = torch.from_numpy(
                        lbl_np[:, :, 0].astype(np.int64))   # [H, W]     int64
                    yield img_t, reg_t, lbl_t
                except tf.errors.OutOfRangeError:
                    break
