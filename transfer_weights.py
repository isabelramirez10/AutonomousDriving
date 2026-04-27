"""
Transfer weights from the original TF checkpoint into the PyTorch model.

Usage:
    python transfer_weights.py \
        --tf_ckpt  ~/FusionLane/Fusionlane_model/Fusionlane_model/model.ckpt-Fusionlane \
        --out_dir  ~/FusionLane/model_pt

Produces:  <out_dir>/latest.pt   (drop-in replacement for the PyTorch checkpoint)
"""

import argparse
import os
import numpy as np
import torch

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

from model_pt import FusionLaneModel

parser = argparse.ArgumentParser()
parser.add_argument('--tf_ckpt',  type=str,
    default='/home/ryan/FusionLane/Fusionlane_model/Fusionlane_model/model.ckpt-Fusionlane')
parser.add_argument('--out_dir',  type=str,
    default='/home/ryan/FusionLane/model_pt')


# ---------------------------------------------------------------------------
# Shape converters
# ---------------------------------------------------------------------------

def conv_w(arr):
    """TF conv:      [H, W, in, out] → PyTorch [out, in, H, W]"""
    return np.transpose(arr, (3, 2, 0, 1))

def dw_w(arr):
    """TF depthwise: [H, W, in, 1]  → PyTorch [in, 1, H, W]"""
    return np.transpose(arr, (2, 3, 0, 1))


# ---------------------------------------------------------------------------
# Main transfer
# ---------------------------------------------------------------------------

def transfer(tf_ckpt, out_dir):
    print(f"Loading TF checkpoint: {tf_ckpt}")
    reader = tf.train.load_checkpoint(tf_ckpt)

    def get(name):
        return reader.get_tensor(name)

    # Build fresh PyTorch model
    model = FusionLaneModel(num_classes=7)
    sd    = model.state_dict()          # mutable copy of all params
    ok    = []
    skip  = []

    # Helper: copy numpy array → state dict entry, with shape check
    def cp(pt_key, arr, transform=None):
        if transform:
            arr = transform(arr)
        t = torch.from_numpy(arr.copy())
        if pt_key not in sd:
            skip.append(f"MISSING PyTorch key: {pt_key}")
            return
        if sd[pt_key].shape != t.shape:
            skip.append(f"SHAPE MISMATCH {pt_key}: "
                        f"PyTorch {tuple(sd[pt_key].shape)} vs TF {tuple(t.shape)}")
            return
        sd[pt_key] = t
        ok.append(pt_key)

    def cp_bn_no_gamma(prefix_tf, prefix_pt):
        """BN without TF gamma (Xception style: scale=False)."""
        cp(f'{prefix_pt}.bias',         get(f'{prefix_tf}/beta'))
        cp(f'{prefix_pt}.running_mean', get(f'{prefix_tf}/moving_mean'))
        cp(f'{prefix_pt}.running_var',  get(f'{prefix_tf}/moving_variance'))
        # weight (gamma) stays at PyTorch default of 1.0

    def cp_bn(prefix_tf, prefix_pt):
        """BN with TF gamma (ASPP style: scale=True)."""
        cp(f'{prefix_pt}.weight',       get(f'{prefix_tf}/gamma'))
        cp(f'{prefix_pt}.bias',         get(f'{prefix_tf}/beta'))
        cp(f'{prefix_pt}.running_mean', get(f'{prefix_tf}/moving_mean'))
        cp(f'{prefix_pt}.running_var',  get(f'{prefix_tf}/moving_variance'))

    def cp_sep(tf_scope, pt_prefix):
        """Depthwise-separable conv + BN (Xception style, no gamma)."""
        cp(f'{pt_prefix}.dw.weight',
           get(f'{tf_scope}/depthwise_weights'), dw_w)
        cp(f'{pt_prefix}.pw.weight',
           get(f'{tf_scope}/pointwise_weights'), conv_w)

    # -----------------------------------------------------------------------
    # Xception backbone
    # -----------------------------------------------------------------------
    print("Transferring Xception backbone...")

    # Region branch
    cp('backbone.a_conv1.weight', get('Xception/a_layer1_conv/weights'), conv_w)
    cp_bn_no_gamma('Xception/a_layer1_bn', 'backbone.a_bn1')

    cp('backbone.a_conv2.weight', get('Xception/a_layer2_conv/weights'), conv_w)
    cp_bn_no_gamma('Xception/a_layer2_bn', 'backbone.a_bn2')

    cp_sep('Xception/a_layer3_dws_conv', 'backbone.a_sep3')
    cp_bn_no_gamma('Xception/a_layer3_bn', 'backbone.a_sep3.bn')

    cp_sep('Xception/a_layer4_dws_conv', 'backbone.a_sep4')
    cp_bn_no_gamma('Xception/a_layer4_bn', 'backbone.a_sep4.bn')

    cp_sep('Xception/a_layer5_dws_conv', 'backbone.a_sep5')
    cp_bn_no_gamma('Xception/a_layer5_bn', 'backbone.a_sep5.bn')

    cp_sep('Xception/a_layer6_dws_conv', 'backbone.a_sep6')
    cp_bn_no_gamma('Xception/a_layer6_bn', 'backbone.a_sep6.bn')

    cp_sep('Xception/a_layer7_dws_conv', 'backbone.a_sep7')
    cp_bn_no_gamma('Xception/a_layer7_bn', 'backbone.a_sep7.bn')

    # Main branch entry
    cp('backbone.b_conv1.weight', get('Xception/block0_conv1/weights'), conv_w)
    cp_bn_no_gamma('Xception/block0_bn1', 'backbone.b_bn1')

    cp('backbone.b_conv2.weight', get('Xception/block0_conv2/weights'), conv_w)
    cp_bn_no_gamma('Xception/block0_bn2', 'backbone.b_bn2')

    cp_sep('Xception/block0_dws_conv1', 'backbone.b_sep1')
    cp_bn_no_gamma('Xception/block0_bn3', 'backbone.b_sep1.bn')

    # Residual 0
    cp('backbone.res0_conv.weight', get('Xception/block0_res_conv/weights'), conv_w)
    cp_bn_no_gamma('Xception/block0_res_bn', 'backbone.res0_bn')

    # Block 1 main
    cp_sep('Xception/block1_dws_conv1', 'backbone.b1_sep1')
    cp_bn_no_gamma('Xception/block1_bn1', 'backbone.b1_sep1.bn')

    cp_sep('Xception/block1_dws_conv2', 'backbone.b1_sep2')
    cp_bn_no_gamma('Xception/block1_bn2', 'backbone.b1_sep2.bn')

    cp_sep('Xception/block1_dws_conv3', 'backbone.b1_sep3')
    cp_bn_no_gamma('Xception/block1_bn3', 'backbone.b1_sep3.bn')

    # Residual 1
    cp('backbone.res1_conv.weight', get('Xception/block1_res_conv/weights'), conv_w)
    cp_bn_no_gamma('Xception/block1_res_bn', 'backbone.res1_bn')

    # Block 2 main
    cp_sep('Xception/block2_dws_conv1', 'backbone.b2_sep1')
    cp_bn_no_gamma('Xception/block2_bn1', 'backbone.b2_sep1.bn')

    cp_sep('Xception/block2_dws_conv2', 'backbone.b2_sep2')
    cp_bn_no_gamma('Xception/block2_bn2', 'backbone.b2_sep2.bn')

    cp_sep('Xception/block2_dws_conv3', 'backbone.b2_sep3')
    cp_bn_no_gamma('Xception/block2_bn3', 'backbone.b2_sep3.bn')

    # Middle flow: TF blocks 3-10 → PyTorch middle[0-7]
    for i in range(8):
        tf_b  = i + 3           # block3 … block10
        pt_mi = f'backbone.middle.{i}'
        for j, s in enumerate(['1', '2', '3']):
            cp_sep(f'Xception/block{tf_b}_dws_conv{s}',
                   f'{pt_mi}.{j}')
            cp_bn_no_gamma(f'Xception/block{tf_b}_bn{s}',
                           f'{pt_mi}.{j}.bn')

    # Exit flow
    cp_sep('Xception/block11_dws_conv1', 'backbone.exit_sep1')
    cp_bn_no_gamma('Xception/block11_bn1', 'backbone.exit_sep1.bn')

    cp_sep('Xception/block11_dws_conv2', 'backbone.exit_sep2')
    cp_bn_no_gamma('Xception/block11_bn2', 'backbone.exit_sep2.bn')

    cp_sep('Xception/block11_dws_conv3', 'backbone.exit_sep3')
    cp_bn_no_gamma('Xception/block11_bn3', 'backbone.exit_bn')

    # -----------------------------------------------------------------------
    # ASPP  (has gamma — scale=True)
    # -----------------------------------------------------------------------
    print("Transferring ASPP...")

    cp('aspp.c1x1.0.weight',    get('aspp/conv_1x1/weights'), conv_w)
    cp_bn('aspp/conv_1x1/BatchNorm', 'aspp.c1x1.1')

    cp('aspp.c3x3_r6.0.weight', get('aspp/conv_3x3_1/weights'), conv_w)
    cp_bn('aspp/conv_3x3_1/BatchNorm', 'aspp.c3x3_r6.1')

    cp('aspp.c3x3_r12.0.weight', get('aspp/conv_3x3_2/weights'), conv_w)
    cp_bn('aspp/conv_3x3_2/BatchNorm', 'aspp.c3x3_r12.1')

    cp('aspp.c3x3_r18.0.weight', get('aspp/conv_3x3_3/weights'), conv_w)
    cp_bn('aspp/conv_3x3_3/BatchNorm', 'aspp.c3x3_r18.1')

    cp('aspp.global_br.0.weight',
       get('aspp/image_level_features/conv_1x1/weights'), conv_w)
    cp_bn('aspp/image_level_features/conv_1x1/BatchNorm', 'aspp.global_br.1')

    cp('aspp.proj.0.weight', get('aspp/conv_1x1_concat/weights'), conv_w)
    cp_bn('aspp/conv_1x1_concat/BatchNorm', 'aspp.proj.1')

    # -----------------------------------------------------------------------
    # LSTM
    # -----------------------------------------------------------------------
    print("Transferring LSTM...")

    # 1×1 projection conv (no bias in TF)
    cp('lstm_proj.weight', get('lstm/conv1_1x1/weights'), conv_w)
    # lstm_proj.bias stays at 0 (PyTorch default zeros)

    # ConvLSTM cell gates:
    # TF kernel [H, W, in+hidden, 4*hidden] → PyTorch [4*hidden, in+hidden, H, W]
    def _ijfo_to_ifgo(arr, axis):
        """Reorder TF [i, j, f, o] → PyTorch [i, f, g, o] (j == g, just relabelled)."""
        i, j, f, o = np.split(arr, 4, axis=axis)
        return np.concatenate([i, f, j, o], axis=axis)

    HIDDEN = 64
    tf_kernel = get('lstm/rnn/multi_rnn_cell/cell_0/conv_lstm_cell/kernel')
    tf_bias   = get('lstm/rnn/multi_rnn_cell/cell_0/conv_lstm_cell/biases')

    # Kernel/bias: gate axis is -1
    tf_kernel = _ijfo_to_ifgo(tf_kernel, axis=-1)
    tf_bias   = _ijfo_to_ifgo(tf_bias,   axis=-1)

    # Bake in TF ConvLSTMCell's runtime forget_bias=1.0 (not stored in 'biases').
    # After reordering, f-gate occupies channels [HIDDEN : 2*HIDDEN].
    tf_bias[HIDDEN:2*HIDDEN] += 1.0

    cp('conv_lstm.cell.gates.weight', tf_kernel, conv_w)
    cp('conv_lstm.cell.gates.bias',   tf_bias)

    # -----------------------------------------------------------------------
    # Decoder  (no bias in TF)
    # -----------------------------------------------------------------------
    print("Transferring decoder...")

    cp('ll_conv.weight',
       get('decoder/low_level_features/conv_1x1/weights'), conv_w)

    cp('dec_c1.weight',
       get('decoder/upsampling_logits/conv_3x3_1/weights'), conv_w)
    cp('dec_c2.weight',
       get('decoder/upsampling_logits/conv_3x3_2/weights'), conv_w)
    cp('dec_c3.weight',
       get('decoder/upsampling_logits/conv_3x3_3/weights'), conv_w)
    cp('dec_c4.weight',
       get('decoder/upsampling_logits/conv_3x3_4/weights'), conv_w)
    cp('inp_conv.weight',
       get('decoder/upsampling_logits/low_level_feature_conv_1x1/weights'), conv_w)
    cp('final.weight',
       get('decoder/upsampling_logits/outputs/weights'), conv_w)
    # decoder biases stay at 0 (PyTorch default zeros — no bias in TF)

    # -----------------------------------------------------------------------
    # Load state dict and save
    # -----------------------------------------------------------------------
    model.load_state_dict(sd)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'latest.pt')
    torch.save({
        'global_step': 115544,   # original TF step
        'epoch':       0,
        'model_state': model.state_dict(),
        'optimizer_state': {},
    }, out_path)

    print(f"\n✓ Transferred {len(ok)} tensors")
    if skip:
        print(f"✗ Skipped / mismatched ({len(skip)}):")
        for s in skip:
            print(f"  {s}")
    print(f"\nSaved → {out_path}")
    print("Run inference to verify:")
    print("  python infer_pt.py --data_dir ~/FusionLane/tfrecord/tfrecord "
          "--model_dir ~/FusionLane/model_pt "
          "--output_dir ~/FusionLane/output_images_pretrained")


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    transfer(FLAGS.tf_ckpt, FLAGS.out_dir)
