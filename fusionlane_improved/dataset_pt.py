"""
FusionLane Improved — dataset loader.

Supports four input modes (auto-detected, no flag needed):
  1. TFRecord files  — place *.tfrecord files in data_dir
  2. Image folder    — place JPEG/PNG/BMP images in data_dir
  3. Video file      — point data_dir at an .mp4 / .avi / .mov / .mkv file
  4. Dummy data      — used automatically when none of the above are found

Region mask defaults to all-ones (full road visible) for image and video input.
Labels default to zero for image/video input (no annotation available).
"""

# =============================================================================
# EASY CONFIG — edit these values to change the smoke-test defaults.
# When dataset_pt.py is run directly (python dataset_pt.py) it uses these.
# =============================================================================
CONFIG = {
    "data_dir":    "./data",   # TFRecord folder | image folder | video file path
    "image_h":     512,        # height to resize every frame to
    "image_w":     512,        # width  to resize every frame to
}
# =============================================================================

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# ImageNet per-channel statistics (RGB order)
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Supported image extensions
_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# Supported video extensions
_VID_EXTS = {'.mp4', '.avi', '.mov', '.mkv', '.m4v', '.wmv', '.flv'}


def normalize_image(tensor):
    """
    Normalize a [4, H, W] float tensor:
      Channels 0-2 (RGB): divide by 255 then apply ImageNet mean/std.
      Channel  3 (region mask): left unchanged — assumed already in [0, 1].
    Returns the modified tensor (operates in-place on a float view).
    """
    t = tensor.float()
    t[:3] = t[:3] / 255.0
    t[:3] = (t[:3] - _MEAN.to(t.device)) / _STD.to(t.device)
    return t


class FusionLaneDataset(Dataset):
    """
    Binary lane segmentation dataset.
    Classes: 0 = background, 1 = lane.

    Input is auto-detected from data_dir:
      - TFRecord files  → multi-sensor data with region mask and labels
      - Image folder    → RGB only; region mask = all-ones, label = zeros
      - Video file      → frame-by-frame; same as image mode
      - (anything else) → synthetic dummy data for quick testing

    Each sample dict:
        image    : FloatTensor [4, H, W]   RGB (normalized) + region mask
        label    : LongTensor  [H, W]      0=bg, 1=lane  (zeros for image/video)
        seq_id   : LongTensor  scalar
        frame_id : LongTensor  scalar      0-3 within a 4-frame sequence
    """

    def __init__(self, data_dir='./data', split='train',
                 image_h=512, image_w=512):
        self.data_dir = data_dir
        self.split    = split
        self.image_h  = image_h
        self.image_w  = image_w
        self.augment  = (split == 'train')

        self._samples = self._load_samples()
        print(f"[FusionLaneDataset] split={split}  samples={len(self._samples)}  "
              f"augment={'YES' if self.augment else 'NO'}")

    # ------------------------------------------------------------------
    # Input-mode detection
    # ------------------------------------------------------------------

    def _is_video(self):
        return (os.path.isfile(self.data_dir) and
                os.path.splitext(self.data_dir)[1].lower() in _VID_EXTS)

    def _is_paired_folder(self):
        """data_dir/images/ and data_dir/masks/ both exist."""
        return (
            os.path.isdir(os.path.join(self.data_dir, 'images')) and
            os.path.isdir(os.path.join(self.data_dir, 'masks'))
        )

    def _is_image_folder(self):
        if not os.path.isdir(self.data_dir):
            return False
        return any(
            os.path.splitext(f)[1].lower() in _IMG_EXTS
            for f in os.listdir(self.data_dir)
        )

    def _tfrecords_present(self):
        if not os.path.isdir(self.data_dir):
            return False
        prefix = 'train' if self.split == 'train' else 'testing'
        return all(
            os.path.exists(os.path.join(self.data_dir,
                                        f'{prefix}-0000{i}-of-00004.tfrecord'))
            for i in range(4)
        )

    # ------------------------------------------------------------------
    # Sample loading — priority: TFRecord > video > image folder > dummy
    # ------------------------------------------------------------------

    def _load_samples(self):
        if self._tfrecords_present():
            try:
                return self._load_tfrecords()
            except Exception as exc:
                print(f"[FusionLaneDataset] TFRecord load failed ({exc}), falling back.")

        if self._is_video():
            try:
                return self._load_video()
            except Exception as exc:
                print(f"[FusionLaneDataset] Video load failed ({exc}), falling back.")

        # Paired images+masks folder — has real labels, preferred over bare image folder
        split_dir = os.path.join(self.data_dir, self.split)
        if (os.path.isdir(os.path.join(split_dir, 'images')) and
                os.path.isdir(os.path.join(split_dir, 'masks'))):
            try:
                return self._load_paired_folder(split_dir)
            except Exception as exc:
                print(f"[FusionLaneDataset] Paired folder load failed ({exc}), falling back.")

        if self._is_paired_folder():
            try:
                return self._load_paired_folder(self.data_dir)
            except Exception as exc:
                print(f"[FusionLaneDataset] Paired folder load failed ({exc}), falling back.")

        if self._is_image_folder():
            try:
                return self._load_image_folder()
            except Exception as exc:
                print(f"[FusionLaneDataset] Image folder load failed ({exc}), falling back.")

        return self._dummy_samples()

    def _dummy_samples(self):
        n_seq = 4 if self.split == 'val' else 10
        samples = []
        for seq_id in range(n_seq):
            for frame_id in range(4):
                samples.append({
                    'source':   'dummy',
                    'seq_id':   seq_id,
                    'frame_id': frame_id,
                })
        return samples

    def _load_image_folder(self):
        files = sorted(
            os.path.join(self.data_dir, f)
            for f in os.listdir(self.data_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )
        print(f"[FusionLaneDataset] Found {len(files)} images in {self.data_dir}")
        return [
            {
                'source':    'image',
                'img_path':  path,
                'seq_id':    i // 4,
                'frame_id':  i % 4,
            }
            for i, path in enumerate(files)
        ]

    def _load_paired_folder(self, base_dir):
        """
        Load from base_dir/images/ + base_dir/masks/ [+ base_dir/depths/].
        Each image must have a same-stem mask in masks/.
        depths/ is optional: if present, depth PNGs are loaded as channel 4
        (real LiDAR depth from prepare_kitti.py).  Absent → channel 4 = ones.
        White pixels (> 127) in mask = lane (class 1); black = background (0).
        """
        img_dir   = os.path.join(base_dir, 'images')
        mask_dir  = os.path.join(base_dir, 'masks')
        depth_dir = os.path.join(base_dir, 'depths')
        has_depth = os.path.isdir(depth_dir)

        img_files = sorted(
            f for f in os.listdir(img_dir)
            if os.path.splitext(f)[1].lower() in _IMG_EXTS
        )

        samples = []
        missing = 0
        for f in img_files:
            stem     = os.path.splitext(f)[0]
            img_path = os.path.join(img_dir, f)

            mask_path = None
            for ext in ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'):
                candidate = os.path.join(mask_dir, stem + ext)
                if os.path.exists(candidate):
                    mask_path = candidate
                    break
            if mask_path is None:
                missing += 1
                continue

            depth_path = None
            if has_depth:
                for ext in ('.png', '.jpg', '.tiff'):
                    candidate = os.path.join(depth_dir, stem + ext)
                    if os.path.exists(candidate):
                        depth_path = candidate
                        break

            samples.append({
                'source':     'paired',
                'img_path':   img_path,
                'mask_path':  mask_path,
                'depth_path': depth_path,   # None when no LiDAR depth available
                'seq_id':     len(samples) // 4,
                'frame_id':   len(samples) % 4,
            })

        if missing:
            print(f"[FusionLaneDataset] WARNING: {missing} image(s) skipped (no mask).")
        depth_count = sum(1 for s in samples if s['depth_path'])
        mode = f"fusion ({depth_count} with LiDAR depth)" if has_depth else "camera-only"
        print(f"[FusionLaneDataset] Paired folder [{mode}]: "
              f"{len(samples)} pairs from {base_dir}")
        if not samples:
            raise ValueError(f"No matching image+mask pairs found in {base_dir}")
        return samples

    def _load_video(self):
        import cv2
        cap = cv2.VideoCapture(self.data_dir)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {self.data_dir}")

        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30
        if n_frames > 500:
            print(f"[FusionLaneDataset] Video has {n_frames} frames "
                  f"({n_frames/fps:.0f}s); loading first 500.")
            n_frames = 500

        # Pre-read frames so we don't hold the file open across workers
        frames = []
        for _ in range(n_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)   # BGR uint8 numpy [H0, W0, 3]
        cap.release()

        print(f"[FusionLaneDataset] Loaded {len(frames)} frames from "
              f"{os.path.basename(self.data_dir)}")
        return [
            {
                'source':   'video',
                'frame':    frames[i],
                'seq_id':   i // 4,
                'frame_id': i % 4,
            }
            for i in range(len(frames))
        ]

    def _load_tfrecords(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

        prefix = 'train' if self.split == 'train' else 'testing'
        files  = [os.path.join(self.data_dir,
                               f'{prefix}-0000{i}-of-00004.tfrecord')
                  for i in range(4)]

        def _parse(raw):
            feat = {
                'image/encoded':  tf.FixedLenFeature((), tf.string, ''),
                'region/encoded': tf.FixedLenFeature((), tf.string, ''),
                'label/encoded':  tf.FixedLenFeature((), tf.string, ''),
            }
            p   = tf.parse_single_example(raw, feat)
            img = tf.cast(tf.image.decode_image(tf.reshape(p['image/encoded'],  []), 3), tf.float32)
            reg = tf.cast(tf.image.decode_image(tf.reshape(p['region/encoded'], []), 1), tf.float32)
            lbl = tf.cast(tf.image.decode_image(tf.reshape(p['label/encoded'],  []), 1), tf.int32)
            img.set_shape([None, None, 3])
            reg.set_shape([None, None, 1])
            lbl.set_shape([None, None, 1])
            return img, reg, lbl

        graph = tf.Graph()
        with graph.as_default():
            ds = (tf.data.Dataset
                  .from_tensor_slices(files)
                  .flat_map(tf.data.TFRecordDataset)
                  .map(_parse, num_parallel_calls=1))
            it      = ds.make_one_shot_iterator()
            next_op = it.get_next()

        cfg = tf.ConfigProto()
        cfg.gpu_options.visible_device_list = ''
        samples = []
        with tf.Session(graph=graph, config=cfg) as sess:
            idx = 0
            while True:
                try:
                    img_np, reg_np, lbl_np = sess.run(next_op)
                    samples.append({
                        'source':   'tfrecord',
                        'img':      img_np,
                        'reg':      reg_np,
                        'lbl':      lbl_np,
                        'seq_id':   idx // 4,
                        'frame_id': idx % 4,
                    })
                    idx += 1
                except tf.errors.OutOfRangeError:
                    break
        return samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        import cv2
        s    = self._samples[idx]
        H, W = self.image_h, self.image_w
        src  = s['source']

        # ── Dummy ──────────────────────────────────────────────────────
        if src == 'dummy':
            rng   = torch.Generator()
            rng.manual_seed(idx * 13 + s['seq_id'] * 97 + s['frame_id'])
            image = torch.rand(4, H, W, generator=rng) * 255.0
            image[3] = torch.rand(H, W, generator=rng)
            label = torch.randint(0, 2, (H, W), generator=rng, dtype=torch.long)

        # ── Image file ────────────────────────────────────────────────
        elif src == 'image':
            bgr = cv2.imread(s['img_path'])
            if bgr is None:
                raise IOError(f"Cannot read image: {s['img_path']}")
            img_np = cv2.cvtColor(
                cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)                             # [H, W, 3]
            reg_np = np.ones((1, H, W), dtype=np.float32)   # full road region
            label  = torch.zeros(H, W, dtype=torch.long)
            image  = torch.from_numpy(
                np.concatenate([img_np.transpose(2, 0, 1), reg_np], axis=0).copy()
            )

        # ── Paired image + mask [+ optional LiDAR depth] ─────────────
        elif src == 'paired':
            bgr = cv2.imread(s['img_path'])
            if bgr is None:
                raise IOError(f"Cannot read image: {s['img_path']}")
            mask_gray = cv2.imread(s['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask_gray is None:
                raise IOError(f"Cannot read mask: {s['mask_path']}")
            img_np = cv2.cvtColor(
                cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            mask_r = cv2.resize(mask_gray, (W, H), interpolation=cv2.INTER_NEAREST)
            lbl_np = (mask_r > 127).astype(np.int64)
            label  = torch.from_numpy(lbl_np)

            # Channel 4: real LiDAR depth if available, else all-ones
            depth_path = s.get('depth_path')
            if depth_path and os.path.exists(depth_path):
                d = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
                if d is not None:
                    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
                    reg_np = (d.astype(np.float32) / 255.0)[np.newaxis]  # [1,H,W]
                else:
                    reg_np = np.ones((1, H, W), dtype=np.float32)
            else:
                reg_np = np.ones((1, H, W), dtype=np.float32)

            image  = torch.from_numpy(
                np.concatenate([img_np.transpose(2, 0, 1), reg_np], axis=0).copy()
            )

        # ── Video frame ───────────────────────────────────────────────
        elif src == 'video':
            bgr    = s['frame']
            img_np = cv2.cvtColor(
                cv2.resize(bgr, (W, H), interpolation=cv2.INTER_LINEAR),
                cv2.COLOR_BGR2RGB,
            ).astype(np.float32)
            reg_np = np.ones((1, H, W), dtype=np.float32)
            label  = torch.zeros(H, W, dtype=torch.long)
            image  = torch.from_numpy(
                np.concatenate([img_np.transpose(2, 0, 1), reg_np], axis=0).copy()
            )

        # ── TFRecord ──────────────────────────────────────────────────
        else:
            img_np = cv2.resize(s['img'], (W, H), interpolation=cv2.INTER_LINEAR)
            reg_ch = cv2.resize(s['reg'][:, :, 0], (W, H),
                                interpolation=cv2.INTER_NEAREST)
            lbl_np = cv2.resize(s['lbl'][:, :, 0].astype(np.uint8), (W, H),
                                interpolation=cv2.INTER_NEAREST).astype(np.int64)
            lbl_np = (lbl_np > 0).astype(np.int64)
            label  = torch.from_numpy(lbl_np)
            image  = torch.from_numpy(
                np.concatenate([
                    img_np.transpose(2, 0, 1),              # [3, H, W]
                    reg_ch[None].astype(np.float32) / 255.0,  # [1, H, W]
                ], axis=0).copy()
            )

        # ── Augmentation (train only) ─────────────────────────────────
        if self.augment:
            # Random horizontal flip
            if torch.rand(1).item() > 0.5:
                image = torch.flip(image, dims=[2])
                label = torch.flip(label, dims=[1])
            # Random brightness/contrast on RGB channels only
            if torch.rand(1).item() > 0.5:
                factor = 0.7 + torch.rand(1).item() * 0.6   # [0.7, 1.3]
                image[:3] = image[:3] * factor

        image = normalize_image(image)

        return {
            'image':    image,
            'label':    label,
            'seq_id':   torch.tensor(s['seq_id'],   dtype=torch.long),
            'frame_id': torch.tensor(s['frame_id'], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Smoke test  (python dataset_pt.py)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    print("Running dataset smoke test with dummy data...")

    ds     = FusionLaneDataset(data_dir=CONFIG['data_dir'], split='train',
                               image_h=CONFIG['image_h'], image_w=CONFIG['image_w'])
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    batch  = next(iter(loader))

    img   = batch['image']
    lbl   = batch['label']
    seq   = batch['seq_id']
    frame = batch['frame_id']

    print(f"  image shape : {img.shape}")
    print(f"  label shape : {lbl.shape}")
    print(f"  image min   : {img.min():.4f}")
    print(f"  image max   : {img.max():.4f}")
    print(f"  seq_ids     : {seq}")
    print(f"  frame_ids   : {frame}")

    assert img.min() < 0, \
        f"Normalization failed: image min={img.min():.4f} must be negative"
    assert img.max() > 1.5, \
        f"Normalization failed: image max={img.max():.4f} must be > 1.5"
    assert list(frame.numpy()) == [0, 1, 2, 3], \
        f"Temporal ordering broken: frame_ids={list(frame.numpy())}"
    assert (seq == seq[0]).all(), \
        f"Sequence grouping broken: seq_ids={list(seq.numpy())}"

    print("Smoke test passed!")
