#!/usr/bin/env python3
"""
ENHANCED Multi-GPU FSE Native FLUXA Training Script - v3.12 (Enhanced Segmentation Loss)

KEY ENHANCEMENTS:
✅ ADDED: Dice loss function for better segmentation
✅ ADDED: Combined Dice+Focal loss option
✅ ADDED: Configurable segmentation loss types (focal, dice, combined)
✅ ADDED: Tunable loss weights for optimal segmentation performance
✅ All other functionality preserved exactly as before
"""

import sys
import os
import time
import io
import json
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
import csv
import math 
from datetime import datetime
from google.cloud import storage # type: ignore
import cv2 
from typing import Dict, List, Optional, Tuple, Any, Union
import functools 

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info("ℹ️ XLA/JIT will be used based on TensorFlow's default behavior or XLA_FLAGS environment variable.")

# --- Mixed Precision Policy ---
try:
    mixed_precision_policy = 'mixed_float16' 
    tf.keras.mixed_precision.set_global_policy(mixed_precision_policy)
    logger.info(f"✅ Global mixed precision policy set to '{mixed_precision_policy}'")
except Exception as e:
    logger.error(f"❌ Failed to set mixed precision policy: {e}", exc_info=True)
    sys.exit(1)

# --- Import FSE Native Components ---
try:
    from fse_native_core import ( 
        FLUXA_FSE_Native, SYNTHA_CONTEXT_WIDTH,
    )
    logger.info("✅ Successfully imported FSE Native components (expected v3.1 with global metrics) from fse_native_core.py")
except ImportError as e:
    logger.error(f"❌ Failed to import FSE components: {e}. Ensure fse_native_core.py (v3.1) is accessible.", exc_info=True)
    sys.exit(1)
except Exception as e: 
    logger.error(f"❌ An unexpected error occurred while importing FSE components: {e}", exc_info=True)
    sys.exit(1)

# --- GPU Configuration ---
def setup_gpu_strategy() -> Tuple[tf.distribute.Strategy, int]:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    num_gpus = len(gpus)
    if num_gpus > 0:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Memory growth enabled for {num_gpus} GPU(s).")
            if num_gpus > 1: strategy = tf.distribute.MirroredStrategy(); logger.info(f"✅ MirroredStrategy for {strategy.num_replicas_in_sync} replicas."); return strategy, strategy.num_replicas_in_sync
            else: logger.info("✅ Single GPU. Default strategy."); return tf.distribute.get_strategy(), 1
        except Exception as e: logger.error(f"❌ GPU setup failed: {e}. CPU fallback.", exc_info=True); return tf.distribute.get_strategy(), 0 
    else: logger.info("⚠️ No GPUs. CPU training."); return tf.distribute.get_strategy(), 0

# --- Batch Size & LR Calc ---
def calculate_optimal_batch_size(num_gpus: int, base_batch_size_per_gpu: int = 8) -> int:
    if num_gpus == 0: logger.info(f"ℹ️ CPU training. Base batch: {base_batch_size_per_gpu}"); return base_batch_size_per_gpu
    global_batch_size = base_batch_size_per_gpu * num_gpus
    logger.info(f"🧠 Batch Calc: Replicas: {num_gpus}, Base/Replica: {base_batch_size_per_gpu}, Global: {global_batch_size}")
    return global_batch_size

def scale_learning_rate(base_lr: float, global_batch_size: int, base_ref_batch_size: int = 256, scaling_method: str = 'linear') -> float:
    if global_batch_size <= 0 or base_ref_batch_size <=0: logger.warning(f"Invalid batch for LR scale. Using base_lr."); return base_lr
    if scaling_method == 'linear': scaled_lr = base_lr * (global_batch_size / base_ref_batch_size)
    elif scaling_method == 'sqrt': scaled_lr = base_lr * (math.sqrt(global_batch_size) / math.sqrt(base_ref_batch_size))
    else: scaled_lr = base_lr
    logger.info(f"📈 LR scale ({scaling_method}): Base {base_lr:.6f}, GlobalBatch {global_batch_size} -> Scaled {scaled_lr:.6f}")
    return scaled_lr

# --- Argument Parsing ---
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='ENHANCED Multi-GPU FSE Native FLUXA Training - v3.12 (Enhanced Segmentation Loss)')
    parser.add_argument('--project_id', type=str, default=os.environ.get('GCP_PROJECT'))
    parser.add_argument('--bucket_name', type=str, required=True)
    parser.add_argument('--job_dir', type=str, default=os.environ.get('AIP_MODEL_DIR', '/tmp/fluxa_fse_output_v3.12'))
    parser.add_argument('--learning_rate', type=float, default=0.001) 
    parser.add_argument('--base_batch_size_per_gpu', type=int, default=3) 
    parser.add_argument('--val_batch_size_per_gpu', type=int, default=3)
    parser.add_argument('--validation_split_fraction', type=float, default=0.15)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr_scaling_method', type=str, default='sqrt', choices=['linear', 'sqrt', 'none'])
    parser.add_argument('--lr_ref_batch_size', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1) 
    parser.add_argument('--gcs_images_path', type=str, default="fluxa/images")
    parser.add_argument('--gcs_labels_base_path', type=str, default="fluxa") 
    parser.add_argument('--max_train_samples', type=int, default=None) 
    parser.add_argument('--max_val_samples', type=int, default=None)   
    parser.add_argument('--img_height', type=int, default=480)
    parser.add_argument('--img_width', type=int, default=640)
    parser.add_argument('--base_channels', type=int, default=24) 
    parser.add_argument('--field_evolution_rate', type=float, default=0.1) 
    parser.add_argument('--max_cses_per_fil', type=int, default=8) 
    parser.add_argument('--enable_syntha', dest='enable_syntha', action='store_true')
    parser.add_argument('--disable_syntha', dest='enable_syntha', action='store_false')
    parser.set_defaults(enable_syntha=True)
    parser.add_argument('--keypoints_loss_weight', type=float, default=2.0)
    parser.add_argument('--segmentation_loss_weight', type=float, default=5.0)  # ENHANCED: Increased default from 2.0 to 5.0
    parser.add_argument('--surface_normals_loss_weight', type=float, default=1.5)
    parser.add_argument('--env_lighting_loss_weight', type=float, default=1.0)
    parser.add_argument('--checkpoint_save_best_monitor', type=str, default='val_fluxa_segmentation_seg_miou')
    parser.add_argument('--checkpoint_save_steps', type=int, default=8000)
    parser.add_argument('--focal_loss_alpha', type=float, default=0.75)  # ENHANCED: Increased from 0.25 to 0.75
    parser.add_argument('--focal_loss_gamma', type=float, default=3.0)   # ENHANCED: Increased from 2.0 to 3.0
    
    # NEW ENHANCED SEGMENTATION LOSS ARGUMENTS
    parser.add_argument('--segmentation_loss_type', type=str, default='combined_dice_focal', 
                       choices=['focal', 'dice', 'combined_dice_focal'],
                       help='Type of loss to use for segmentation: focal, dice, or combined_dice_focal')
    parser.add_argument('--seg_dice_weight', type=float, default=0.6,
                       help='Weight for Dice loss in combined loss (0.0-1.0)')
    parser.add_argument('--seg_focal_weight', type=float, default=0.4,
                       help='Weight for Focal loss in combined loss (0.0-1.0)')
    
    args_parsed = parser.parse_args()
    if not args_parsed.bucket_name: parser.error("--bucket_name required.")
    
    # Ensure paths from args don't have leading/trailing slashes that might interfere with tf.strings.join
    args_parsed.gcs_images_path = args_parsed.gcs_images_path.strip('/')
    args_parsed.gcs_labels_base_path = args_parsed.gcs_labels_base_path.strip('/')

    # Validate combined loss weights
    if args_parsed.segmentation_loss_type == 'combined_dice_focal':
        total_weight = args_parsed.seg_dice_weight + args_parsed.seg_focal_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Combined loss weights sum to {total_weight:.3f}, not 1.0. Normalizing...")
            args_parsed.seg_dice_weight = args_parsed.seg_dice_weight / total_weight
            args_parsed.seg_focal_weight = args_parsed.seg_focal_weight / total_weight
            logger.info(f"Normalized weights: Dice={args_parsed.seg_dice_weight:.3f}, Focal={args_parsed.seg_focal_weight:.3f}")

    if args_parsed.max_train_samples: logger.info(f"Limiting training samples to {args_parsed.max_train_samples} (after val split).")
    if args_parsed.max_val_samples: logger.info(f"Limiting validation samples to {args_parsed.max_val_samples} (from split).")
    return args_parsed
args = parse_arguments()

# --- CUSTOM KERAS METRIC CLASSES ---
class FixedFSENativeMetric(tf.keras.metrics.Metric):
    def __init__(self, name='fixed_fse_field_coherence', **kwargs):
        super().__init__(name=name, **kwargs)
        self.coherence_sum = self.add_weight(name='coherence_sum', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight(name='count', initializer='zeros', dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_f32 = tf.cast(y_pred, tf.float32)
        spatial_var_f32 = tf.math.reduce_variance(y_pred_f32, axis=[1, 2]) 
        if len(spatial_var_f32.shape) > 1 and spatial_var_f32.shape[-1] > 1: spatial_var_mean_f32 = tf.reduce_mean(spatial_var_f32, axis=-1)
        else: spatial_var_mean_f32 = tf.squeeze(spatial_var_f32, axis=-1) if len(spatial_var_f32.shape) > 1 else spatial_var_f32
        coherence_score_f32 = 1.0 / (1.0 + spatial_var_mean_f32 + tf.keras.backend.epsilon())
        if sample_weight is not None:
            sample_weight_f32 = tf.cast(sample_weight, tf.float32)
            coherence_score_f32 *= tf.reshape(sample_weight_f32, tf.shape(coherence_score_f32)) if len(sample_weight_f32.shape) > 0 else sample_weight_f32
        self.coherence_sum.assign_add(tf.reduce_sum(coherence_score_f32))
        self.count.assign_add(tf.cast(tf.size(coherence_score_f32), tf.float32))
    def result(self): 
        count_val = tf.cast(self.count, self.coherence_sum.dtype); return tf.math.divide_no_nan(self.coherence_sum, count_val) 
    def reset_state(self): self.coherence_sum.assign(0.0); self.count.assign(0.0)
    def get_config(self): base_config = super().get_config(); return {**base_config}

class PhysicsGroundedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='physics_grounded_accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.physics_sum = self.add_weight('physics_sum', initializer='zeros', dtype=tf.float32)
        self.count = self.add_weight('count', initializer='zeros', dtype=tf.float32)
    def update_state(self, y_true, y_pred, sample_weight=None):
        surface_normals_f32 = tf.cast(y_pred, tf.float32)
        normal_magnitudes_f32 = tf.norm(surface_normals_f32, axis=-1) + tf.keras.backend.epsilon()
        unit_deviation_f32 = tf.abs(normal_magnitudes_f32 - 1.0)
        unit_normal_score_f32 = tf.reduce_mean(tf.exp(-unit_deviation_f32 * 4.0), axis=[1,2])
        normal_dx_f32 = surface_normals_f32[:, :, 1:, :] - surface_normals_f32[:, :, :-1, :]
        normal_dy_f32 = surface_normals_f32[:, 1:, :, :] - surface_normals_f32[:, :-1, :, :]
        gradient_mag_x_f32 = tf.sqrt(tf.reduce_sum(tf.square(normal_dx_f32), axis=-1) + 1e-8)
        gradient_mag_y_f32 = tf.sqrt(tf.reduce_sum(tf.square(normal_dy_f32), axis=-1) + 1e-8)
        target_gradient_f32 = tf.constant(0.12, dtype=tf.float32)
        mean_grad_x_per_sample_f32 = tf.reduce_mean(gradient_mag_x_f32, axis=[1,2])
        mean_grad_y_per_sample_f32 = tf.reduce_mean(gradient_mag_y_f32, axis=[1,2])
        grad_score_x_f32 = tf.exp(-tf.abs(mean_grad_x_per_sample_f32 - target_gradient_f32) * 10.0)
        grad_score_y_f32 = tf.exp(-tf.abs(mean_grad_y_per_sample_f32 - target_gradient_f32) * 10.0)
        gradient_score_f32 = (grad_score_x_f32 + grad_score_y_f32) * 0.5
        mean_normal_f32 = tf.reduce_mean(surface_normals_f32, axis=[1, 2])
        mean_normal_magnitude_f32 = tf.norm(mean_normal_f32, axis=-1) + tf.keras.backend.epsilon()
        distribution_score_f32 = tf.exp(-(mean_normal_magnitude_f32 - 0.35) ** 2 * 6.0)
        physics_accuracy_f32_per_sample = 0.4 * unit_normal_score_f32 + 0.4 * gradient_score_f32 + 0.2 * distribution_score_f32
        if sample_weight is not None:
            sample_weight_f32 = tf.cast(sample_weight, tf.float32)
            physics_accuracy_f32_per_sample *= tf.reshape(sample_weight_f32, tf.shape(physics_accuracy_f32_per_sample)) if len(sample_weight_f32.shape) > 0 else sample_weight_f32
        self.physics_sum.assign_add(tf.reduce_sum(physics_accuracy_f32_per_sample))
        self.count.assign_add(tf.cast(tf.size(physics_accuracy_f32_per_sample), tf.float32))
    def result(self): 
        count_val = tf.cast(self.count, self.physics_sum.dtype); return tf.math.divide_no_nan(self.physics_sum, count_val) 
    def reset_state(self): self.physics_sum.assign(0.0); self.count.assign(0.0)
    def get_config(self): base_config = super().get_config(); return {**base_config}

# --- ENHANCED LOSS FUNCTIONS ---
class StableFSENativeLoss(tf.keras.losses.Loss): 
    def __init__(self, field_smoothness: float = 0.01, resize_target: bool = True, name: str = 'stable_fse_native_loss', **kwargs):
        super().__init__(name=name, reduction=tf.keras.losses.Reduction.NONE); self.field_smoothness = float(field_smoothness); self.resize_target = resize_target
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor: 
        y_true_f32 = tf.cast(y_true, tf.float32); y_pred_f32 = tf.cast(y_pred, tf.float32)
        if self.resize_target and (y_true_f32.shape[1:3] != y_pred_f32.shape[1:3]): y_pred_f32 = tf.image.resize(y_pred_f32, tf.shape(y_true_f32)[1:3], method='bilinear')
        reduction_axes_spatial = list(range(1, len(y_pred_f32.shape))) 
        if not reduction_axes_spatial : reduction_axes_spatial = None
        base_loss = tf.reduce_mean(tf.square(y_true_f32 - y_pred_f32), axis=reduction_axes_spatial)
        if self.field_smoothness > 0 and len(y_pred_f32.shape) == 4: 
            gy_pred = y_pred_f32[:, 1:, :, :] - y_pred_f32[:, :-1, :, :]; gx_pred = y_pred_f32[:, :, 1:, :] - y_pred_f32[:, :, :-1, :]
            reduction_axes_grad = list(range(1, len(gy_pred.shape)))
            if not reduction_axes_grad: reduction_axes_grad = None
            smoothness_penalty = tf.reduce_mean(tf.square(gy_pred), axis=reduction_axes_grad) + tf.reduce_mean(tf.square(gx_pred), axis=reduction_axes_grad)
            base_loss += self.field_smoothness * smoothness_penalty
        return base_loss
def stable_fse_loss_wrapper_fn(y_true, y_pred, field_smoothness_val=0.01, resize_target_val=True):
    return StableFSENativeLoss(field_smoothness=field_smoothness_val, resize_target=resize_target_val).call(y_true, y_pred)

# NEW: Enhanced segmentation loss functions
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss for segmentation.
    
    Args:
        y_true: Ground truth labels [batch_size, height, width, 1]
        y_pred: Predicted probabilities [batch_size, height, width, 1]
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Per-sample dice loss
    """
    y_true_f32 = tf.cast(y_true, tf.float32)
    y_pred_f32 = tf.cast(y_pred, tf.float32)
    
    # Flatten for each sample
    y_true_flat = tf.reshape(y_true_f32, [tf.shape(y_true_f32)[0], -1])  # [batch_size, H*W*C]
    y_pred_flat = tf.reshape(y_pred_f32, [tf.shape(y_pred_f32)[0], -1])  # [batch_size, H*W*C]
    
    # Calculate intersection and union for each sample
    intersection = tf.reduce_sum(y_true_flat * y_pred_flat, axis=1)  # [batch_size]
    y_true_sum = tf.reduce_sum(y_true_flat, axis=1)  # [batch_size]
    y_pred_sum = tf.reduce_sum(y_pred_flat, axis=1)  # [batch_size]
    
    # Dice coefficient for each sample
    dice_coeff = (2.0 * intersection + smooth) / (y_true_sum + y_pred_sum + smooth)
    
    # Dice loss (1 - dice_coefficient)
    dice_loss_per_sample = 1.0 - dice_coeff
    
    return dice_loss_per_sample

bce_loss_fn_for_focal = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
def focal_loss_binary_segmentation(y_true, y_pred, alpha=0.75, gamma=3.0, epsilon=1e-7):
    """Enhanced focal loss with better default parameters for segmentation"""
    y_true_f32 = tf.cast(y_true, tf.float32); y_pred_f32 = tf.cast(y_pred, tf.float32)
    y_pred_f32 = tf.clip_by_value(y_pred_f32, epsilon, 1.0 - epsilon)
    bce_per_pixel = bce_loss_fn_for_focal(y_true_f32, y_pred_f32) 
    p_t = tf.where(tf.equal(y_true_f32, 1.0), y_pred_f32, 1.0 - y_pred_f32)
    alpha_factor = tf.where(tf.equal(y_true_f32, 1.0), alpha, 1.0 - alpha)
    modulating_factor = tf.pow(1.0 - p_t, gamma)
    if len(bce_per_pixel.shape) == len(alpha_factor.shape) -1 : bce_per_pixel = tf.expand_dims(bce_per_pixel, axis=-1) 
    focal_loss_per_pixel = alpha_factor * modulating_factor * bce_per_pixel
    reduction_axes = list(range(1, len(focal_loss_per_pixel.shape)))
    if not reduction_axes: per_sample_loss = focal_loss_per_pixel
    else: per_sample_loss = tf.reduce_mean(focal_loss_per_pixel, axis=reduction_axes)
    return per_sample_loss

def combined_dice_focal_loss_segmentation(y_true, y_pred, dice_weight=0.6, focal_weight=0.4, 
                                        focal_alpha=0.75, focal_gamma=3.0, dice_smooth=1e-6):
    """
    Combined Dice + Focal loss for segmentation.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities  
        dice_weight: Weight for dice loss component (0.0-1.0)
        focal_weight: Weight for focal loss component (0.0-1.0)
        focal_alpha: Alpha parameter for focal loss
        focal_gamma: Gamma parameter for focal loss
        dice_smooth: Smoothing factor for dice loss
    
    Returns:
        Combined loss per sample
    """
    # Calculate individual losses
    dice_loss_val = dice_loss(y_true, y_pred, smooth=dice_smooth)
    focal_loss_val = focal_loss_binary_segmentation(y_true, y_pred, alpha=focal_alpha, gamma=focal_gamma)
    
    # Combine losses
    combined_loss = dice_weight * dice_loss_val + focal_weight * focal_loss_val
    
    return combined_loss

def per_sample_mse_loss_fn(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true_f32 = tf.cast(y_true, tf.float32); y_pred_f32 = tf.cast(y_pred, tf.float32)
    mse_loss = tf.keras.losses.mean_squared_error(y_true_f32, y_pred_f32)
    if len(mse_loss.shape) > 1: return tf.reduce_mean(mse_loss, axis=list(range(1, len(mse_loss.shape))))
    return mse_loss 

# --- FIXED GCS Client & Data Listing ---
_gcs_storage_client: Optional[storage.Client] = None; _gcs_bucket_obj: Optional[storage.Bucket] = None
def _initialize_gcs_client(): 
    global _gcs_storage_client, _gcs_bucket_obj
    if _gcs_bucket_obj is None and args.bucket_name:
        try: 
            # FIXED: Simple GCS client initialization (retry/timeout handled at operation level)
            _gcs_storage_client = storage.Client(project=args.project_id)
            _gcs_bucket_obj = _gcs_storage_client.bucket(args.bucket_name)
            logger.info(f"✅ GCS client initialized: {args.bucket_name}")
        except Exception as e: logger.error(f"GCS client init failed: {e}", exc_info=True); _gcs_bucket_obj = None
    elif not args.bucket_name: logger.warning("GCS bucket name not provided.")
def _get_gcs_bucket() -> Optional[storage.Bucket]: 
    if _gcs_bucket_obj is None: _initialize_gcs_client()
    return _gcs_bucket_obj

def gcs_list_base_ids(bucket_obj_param: Optional[storage.Bucket], gcs_folder_path: str, file_extension: str) -> List[str]:
    if not bucket_obj_param: logger.warning(f"GCS N/A for {gcs_folder_path}"); return []
    prefix = gcs_folder_path if gcs_folder_path.endswith('/') else gcs_folder_path + '/'
    try:
        blobs = list(bucket_obj_param.list_blobs(prefix=prefix))
        return list(set([os.path.splitext(os.path.basename(b.name))[0] for b in blobs if b.name.endswith(file_extension) and not b.name.endswith('/')]))
    except Exception as e: logger.error(f"Err list GCS: gs://{bucket_obj_param.name}/{prefix} ext {file_extension}: {e}", exc_info=True); return []

def find_main_dataset_ids() -> List[str]:
    gcs_bucket = _get_gcs_bucket()
    if not gcs_bucket: return []
    def _ensure_trailing_slash(path_str): return path_str if path_str.endswith('/') else path_str + '/'
    kp_folder = _ensure_trailing_slash(os.path.join(args.gcs_labels_base_path, "keypoints"))
    mask_folder = _ensure_trailing_slash(os.path.join(args.gcs_labels_base_path, "segmentation_masks"))
    normals_folder = _ensure_trailing_slash(os.path.join(args.gcs_labels_base_path, "surface_normals"))
    kp_ids = set(gcs_list_base_ids(gcs_bucket, kp_folder, '.npy'))
    mask_ids = set(gcs_list_base_ids(gcs_bucket, mask_folder, '.png'))
    norm_ids = set(gcs_list_base_ids(gcs_bucket, normals_folder, '.npy'))
    logger.info(f"Found {len(kp_ids)} keypoint IDs in gs://{args.bucket_name}/{kp_folder}")
    logger.info(f"Found {len(mask_ids)} mask IDs in gs://{args.bucket_name}/{mask_folder}")
    logger.info(f"Found {len(norm_ids)} normal IDs in gs://{args.bucket_name}/{normals_folder}")
    common_ids = list(kp_ids & mask_ids & norm_ids) 
    if not common_ids: logger.error(f"No common sample IDs found in main data paths: gs://{args.bucket_name}/{args.gcs_labels_base_path}/<modality>/")
    else: logger.info(f"Found {len(common_ids)} common sample IDs for train/val split.")
    return common_ids

# --- Data Loading and Preprocessing Functions ---
def _py_load_npy_from_gcs(gcs_path_bytes: bytes, expected_final_dims: int) -> np.ndarray:
    gcs_path_str = gcs_path_bytes.decode('utf-8'); bucket = _get_gcs_bucket()
    default_shape = (args.img_height, args.img_width, expected_final_dims); default_arr = np.zeros(default_shape, dtype=np.float32)
    if "surface_normals" in gcs_path_str and expected_final_dims == 3 : default_arr[...,2] = 1.0
    if not bucket: return default_arr
    if gcs_path_str.startswith(f"gs://{bucket.name}/"): blob_name = gcs_path_str[len(f"gs://{bucket.name}/"):]
    else: logger.error(f"Path {gcs_path_str} does not match bucket {bucket.name}"); return default_arr
    if not blob_name: return default_arr
    try:
        blob = bucket.blob(blob_name)
        if not blob.exists(): return default_arr 
        arr = np.load(io.BytesIO(blob.download_as_bytes())); interp = cv2.INTER_NEAREST if expected_final_dims == 17 else cv2.INTER_LINEAR
        if arr.ndim == 2 and expected_final_dims == 17: arr = np.stack([cv2.resize(arr, (args.img_width, args.img_height), interpolation=interp)]*17, axis=-1)
        elif arr.ndim == 3 and arr.shape[-1] == expected_final_dims:
            if arr.shape[0]!=args.img_height or arr.shape[1]!=args.img_width: arr=cv2.resize(arr,(args.img_width,args.img_height),interpolation=interp)
        elif arr.ndim != 3 or arr.shape[-1] != expected_final_dims: return default_arr
        if "surface_normals" in gcs_path_str and expected_final_dims == 3:
            m, M = np.min(arr), np.max(arr)
            if M > 1.05 or m < -1.05 : 
                if (M-m) > 1e-7: arr = 2.0*(arr-m)/(M-m)-1.0
                else: arr = np.zeros_like(arr); arr[...,2]=1.0
        return arr.astype(np.float32)
    except Exception: return default_arr

def load_and_preprocess_sample(image_id_tensor: tf.Tensor, is_training_flag: bool) -> Tuple[Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], Dict[str, tf.Tensor]]:
    image_id_str = tf.strings.as_string(image_id_tensor) 
    
    # Use args directly as they are now cleaned in parse_arguments
    img_path_prefix = args.gcs_images_path 
    label_base_path = args.gcs_labels_base_path 

    img_gcs_path_jpg = tf.strings.join(["gs://", args.bucket_name, "/", img_path_prefix, "/", image_id_str, ".jpg"])
    img_gcs_path_png = tf.strings.join(["gs://", args.bucket_name, "/", img_path_prefix, "/", image_id_str, ".png"])
    
    img_content = tf.constant(b'') 
    try: img_content = tf.io.read_file(img_gcs_path_jpg)
    except tf.errors.NotFoundError: 
        try: img_content = tf.io.read_file(img_gcs_path_png)
        except tf.errors.NotFoundError: tf.print("WARNING: Image file not found (TF):", image_id_str,". Using zeros.", output_stream=sys.stderr)
    except Exception as e_read_img: tf.print("WARNING: Error reading image file (TF):", image_id_str, "Error:", e_read_img, ". Using zeros.", output_stream=sys.stderr)

    img_decoded_f32 = tf.zeros([args.img_height, args.img_width, 3], dtype=tf.float32)
    if tf.strings.length(img_content) > 0:
        try:
            decoded_img_raw = tf.image.decode_image(img_content, channels=3, expand_animations=False)
            img_decoded_f32 = tf.cast(decoded_img_raw, tf.float32) 
        except Exception as e_decode: 
            tf.print("WARNING: Error decoding image (TF):", image_id_str, "Error:", e_decode, ". Using zeros.", output_stream=sys.stderr)
    
    img_resized_f32 = tf.image.resize(img_decoded_f32, [args.img_height, args.img_width], method='bilinear')
    img = img_resized_f32 / 255.0 
    img.set_shape([args.img_height, args.img_width, 3])
    
    kp_gcs_path = tf.strings.join(["gs://", args.bucket_name, "/", label_base_path, "/keypoints/", image_id_str, ".npy"])
    keypoints = tf.numpy_function(functools.partial(_py_load_npy_from_gcs, expected_final_dims=17), [kp_gcs_path], Tout=tf.float32); keypoints.set_shape([args.img_height, args.img_width, 17])
    
    mask_gcs_path = tf.strings.join(["gs://", args.bucket_name, "/", label_base_path, "/segmentation_masks/", image_id_str, ".png"])
    segmentation_mask = tf.zeros([args.img_height, args.img_width, 1], dtype=tf.float32) 
    mask_content = tf.constant(b'')
    try: mask_content = tf.io.read_file(mask_gcs_path)
    except tf.errors.NotFoundError: tf.print("WARNING: SegMask not found (TF):", image_id_str, ". Using zeros.", output_stream=sys.stderr)
    except Exception as e_read_mask: tf.print("WARNING: Error reading SegMask (TF):", image_id_str, "Error:", e_read_mask, ". Using zeros.", output_stream=sys.stderr)
    if tf.strings.length(mask_content) > 0:
        try:
            decoded_mask_raw = tf.image.decode_png(mask_content, channels=1, dtype=tf.uint8)
            decoded_mask_f32 = tf.cast(decoded_mask_raw, tf.float32)
            mask_resized_f32 = tf.image.resize(decoded_mask_f32, [args.img_height, args.img_width], method='nearest')
            segmentation_mask = mask_resized_f32 / 255.0
        except Exception as e_decode_mask: tf.print("WARNING: Error decoding SegMask (TF):", image_id_str, "Error:", e_decode_mask, ". Using zeros.", output_stream=sys.stderr)
    segmentation_mask.set_shape([args.img_height, args.img_width, 1])

    normals_gcs_path = tf.strings.join(["gs://", args.bucket_name, "/", label_base_path, "/surface_normals/", image_id_str, ".npy"])
    surface_normals = tf.numpy_function(functools.partial(_py_load_npy_from_gcs, expected_final_dims=3), [normals_gcs_path], Tout=tf.float32); surface_normals.set_shape([args.img_height, args.img_width, 3])
    
    img, keypoints, segmentation_mask, surface_normals = apply_augmentation(img, keypoints, segmentation_mask, surface_normals, training=tf.constant(is_training_flag) ) 
    
    syntha_context = tf.zeros([SYNTHA_CONTEXT_WIDTH], dtype=tf.float32)
    if args.enable_syntha:
        mean_c = tf.reduce_mean(img, axis=[0, 1]); std_c = tf.math.reduce_std(img, axis=[0, 1])
        kp_density = tf.reshape(tf.reduce_sum(keypoints) / tf.cast(args.img_height * args.img_width * 17, tf.float32), [1])
        seg_coverage = tf.reshape(tf.reduce_mean(segmentation_mask), [1])
        base_features = tf.concat([tf.reshape(mean_c, [-1]), tf.reshape(std_c, [-1]), kp_density, seg_coverage], axis=0) 
        current_width = tf.shape(base_features)[0]; padding_needed = SYNTHA_CONTEXT_WIDTH - current_width
        if tf.greater(padding_needed, 0): syntha_context = tf.pad(base_features, [[0, padding_needed]])
        elif tf.less(padding_needed, 0): syntha_context = base_features[:SYNTHA_CONTEXT_WIDTH]
        else: syntha_context = base_features
        syntha_context.set_shape([SYNTHA_CONTEXT_WIDTH])
    inputs_tuple = (img, syntha_context) if args.enable_syntha else img
    labels_dict = {'fluxa_keypoints': keypoints,'fluxa_segmentation': segmentation_mask,'fluxa_surface_normals': surface_normals}
    return inputs_tuple, labels_dict

def add_batched_environment_lighting(inputs, labels):
    batch_size = tf.shape(labels['fluxa_keypoints'])[0] 
    env_light = load_environment_lighting() 
    batched_env_light = tf.tile(tf.expand_dims(env_light, 0), [batch_size, 1])
    labels['fluxa_environment_lighting'] = batched_env_light
    return inputs, labels

def create_optimized_tf_dataset(sample_ids, batch_size, is_training_dataset: bool, num_parallel_calls=tf.data.AUTOTUNE):
    if not sample_ids: 
        logger.warning(f"No IDs for {'training' if is_training_dataset else 'validation'}. Empty dataset.")
        img_spec = tf.TensorSpec(shape=(None, args.img_height, args.img_width, 3), dtype=tf.float32); syntha_spec = tf.TensorSpec(shape=(None, SYNTHA_CONTEXT_WIDTH,), dtype=tf.float32)
        input_spec = (img_spec, syntha_spec) if args.enable_syntha else img_spec
        label_spec = {'fluxa_keypoints': tf.TensorSpec(shape=(None,args.img_height, args.img_width,17),dtype=tf.float32),'fluxa_segmentation':tf.TensorSpec(shape=(None,args.img_height,args.img_width,1),dtype=tf.float32),'fluxa_surface_normals':tf.TensorSpec(shape=(None,args.img_height,args.img_width,3),dtype=tf.float32),'fluxa_environment_lighting':tf.TensorSpec(shape=(None,9),dtype=tf.float32)}
        return tf.data.Dataset.from_generator(lambda: iter([]), output_signature=(input_spec, label_spec))
    
    logger.info(f"Optimized dataset for {'training' if is_training_dataset else 'validation'}: {len(sample_ids)} samples, batch: {batch_size}")
    dataset = tf.data.Dataset.from_tensor_slices(sample_ids)
    if is_training_dataset: dataset = dataset.shuffle(buffer_size=len(sample_ids), reshuffle_each_iteration=True)
    
    mapped_dataset = dataset.map(functools.partial(load_and_preprocess_sample, is_training_flag=is_training_dataset), num_parallel_calls=num_parallel_calls)
    
    def is_sample_valid(inputs, labels):
        img_tensor = inputs[0] if isinstance(inputs, tuple) else inputs
        return tf.greater(tf.reduce_sum(tf.abs(img_tensor)), 1e-6) 
    filtered_dataset = mapped_dataset.filter(is_sample_valid)
    
    batched_dataset = filtered_dataset.batch(batch_size, drop_remainder=is_training_dataset) 
    final_dataset = batched_dataset.map(add_batched_environment_lighting, num_parallel_calls=num_parallel_calls) 
    if is_training_dataset: final_dataset = final_dataset.repeat() 
    final_dataset = final_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    logger.info(f"Optimized TF Dataset created for {'training' if is_training_dataset else 'validation'}.")
    return final_dataset

def apply_augmentation(image, keypoints, segmentation, surface_normals, training: tf.Tensor):
    def augment_fn():
        aug_image, aug_keypoints, aug_segmentation, aug_surface_normals = image, keypoints, segmentation, surface_normals
        if tf.random.uniform(()) > 0.5: aug_image=tf.image.flip_left_right(aug_image); aug_keypoints=tf.image.flip_left_right(aug_keypoints); aug_segmentation=tf.image.flip_left_right(aug_segmentation); aug_surface_normals=tf.image.flip_left_right(aug_surface_normals); aug_surface_normals=tf.stack([-aug_surface_normals[...,0],aug_surface_normals[...,1],aug_surface_normals[...,2]],axis=-1)
        aug_image=tf.image.random_brightness(aug_image,max_delta=0.05); aug_image=tf.image.random_contrast(aug_image,lower=0.95,upper=1.05); aug_image=tf.clip_by_value(aug_image,0.0,1.0)
        return aug_image, aug_keypoints, aug_segmentation, aug_surface_normals
    def no_augment_fn(): return image, keypoints, segmentation, surface_normals
    return tf.cond(tf.logical_and(training, tf.random.uniform(()) > 0.7), augment_fn, no_augment_fn)

def load_environment_lighting(): return tf.random.normal(shape=[9], dtype=tf.float32) * 0.1 + 0.5

class GradientAccumulationOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, optimizer: tf.keras.optimizers.Optimizer, accumulation_steps: int = 1, name="GradientAccumulation", **kwargs):
        super().__init__(name=name, **kwargs); self.optimizer = optimizer; self.accumulation_steps = tf.constant(accumulation_steps, dtype=tf.int32)
        if accumulation_steps < 1: raise ValueError("accumulation_steps must be >= 1.")
        self._optimizer_vars: List[tf.Variable] = []; self._var_to_acc_index: Dict[Any, int] = {}; self.accumulated_gradients: List[tf.Variable] = [] 
        self.accumulation_counter = tf.Variable(0, dtype=tf.int32, trainable=False, name="accumulation_counter"); self._built = False 
    def build(self, var_list: List[tf.Variable]): 
        if self._built and self._optimizer_vars == var_list: return             
        self.optimizer.build(var_list) 
        if not self.accumulated_gradients or len(self.accumulated_gradients) != len(var_list) or self._optimizer_vars != var_list:
            self._optimizer_vars = list(var_list); self._var_to_acc_index = {var.ref(): i for i, var in enumerate(self._optimizer_vars)}
            self.accumulated_gradients = [tf.Variable(tf.zeros_like(var, dtype=var.dtype), trainable=False, name=f"acc_grad_{i}_{var.name.replace(':', '_').replace('/', '_')}") for i, var in enumerate(self._optimizer_vars)]
        self._built = True
    def apply_gradients(self, grads_and_vars: List[Tuple[Optional[tf.Tensor], tf.Variable]], name: Optional[str]=None, **kwargs):
        if not self._built: self.build([v for g,v in grads_and_vars])
        for grad, var in grads_and_vars:
            if grad is not None:
                var_ref = var.ref(); acc_idx = self._var_to_acc_index.get(var_ref)
                if acc_idx is not None: self.accumulated_gradients[acc_idx].assign_add(tf.cast(grad, self.accumulated_gradients[acc_idx].dtype))
        new_counter_val = self.accumulation_counter.assign_add(1)
        with tf.control_dependencies([new_counter_val]): is_update_step = tf.equal(new_counter_val, self.accumulation_steps)
        final_grads = [tf.cond(is_update_step, lambda i=i, v=v: tf.cast(tf.cast(self.accumulated_gradients[i], tf.float32) / tf.cast(self.accumulation_steps, tf.float32), v.dtype), lambda v=v: tf.zeros_like(v, dtype=v.dtype)) for i, v in enumerate(self._optimizer_vars)]
        applied_op = self.optimizer.apply_gradients(zip(final_grads, self._optimizer_vars), name=name, **kwargs)
        def _reset_state_fn(): return tf.group([ag.assign(tf.zeros_like(ag)) for ag in self.accumulated_gradients] + [self.accumulation_counter.assign(0)])
        with tf.control_dependencies([applied_op]): reset_op = tf.cond(is_update_step, _reset_state_fn, tf.no_op)
        return tf.group(applied_op, reset_op) 
    @property
    def learning_rate(self): return self.optimizer.learning_rate
    @property
    def iterations(self): return self.optimizer.iterations 
    def get_config(self): cfg = super().get_config(); cfg.update({"optimizer":tf.keras.optimizers.serialize(self.optimizer),"accumulation_steps":int(self.accumulation_steps.numpy())}); return cfg
    @classmethod
    def from_config(cls, cfg, custom_objects=None): opt = tf.keras.optimizers.deserialize(cfg.pop("optimizer"), custom_objects=custom_objects); return cls(opt, **cfg)

# --- FIXED CHECKPOINT MANAGERS ---

class FSEStepCheckpointCallback(tf.keras.callbacks.Callback):
    """FIXED step-based checkpoint callback that tracks steps internally"""
    
    def __init__(self, checkpoint_dir: str, gcs_bucket_name: str, gcs_path: str, 
                 save_every_steps: int = 8000, gcs_project_id: Optional[str] = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_path = gcs_path
        self.save_every_steps = save_every_steps
        self.gcs_project_id = gcs_project_id
        self.step_counter = 0
        self.current_epoch = 0
        
        # Create local directory
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize GCS bucket
        self._gcs_bucket = None
        self._init_gcs_bucket()
        
        logger.info(f"✅ FSE Step Checkpoint initialized: every {save_every_steps} steps")
    
    def _init_gcs_bucket(self):
        """Initialize GCS bucket with proper error handling"""
        try:
            # FIXED: Simple GCS client initialization
            client = storage.Client(project=self.gcs_project_id)
            self._gcs_bucket = client.bucket(self.gcs_bucket_name)
            logger.info(f"✅ Step checkpoint GCS bucket connected: {self.gcs_bucket_name}")
        except Exception as e:
            logger.error(f"❌ Step checkpoint GCS init failed: {e}")
            self._gcs_bucket = None
    
    def on_epoch_begin(self, epoch, logs=None):
        self.current_epoch = epoch
    
    def on_train_batch_end(self, batch, logs=None):
        """FIXED: Track steps internally and save when needed"""
        self.step_counter += 1
        
        if self.step_counter % self.save_every_steps == 0:
            self._save_step_checkpoint(logs)
    
    def _save_step_checkpoint(self, logs=None):
        """Save step-based checkpoint with proper naming"""
        try:
            # FIXED: Generate filename using internal step counter
            filename = f"fluxa_fse_step_e{self.current_epoch+1:04d}_s{self.step_counter:05d}.weights.h5"
            local_path = os.path.join(self.checkpoint_dir, filename)
            
            logger.info(f"💾 Saving step checkpoint: {filename}")
            
            # Save weights locally
            self.model.save_weights(local_path)
            
            # Upload to GCS
            if self._gcs_bucket:
                try:
                    gcs_full_path = os.path.join(self.gcs_path, filename)
                    blob = self._gcs_bucket.blob(gcs_full_path)
                    blob.upload_from_filename(local_path)
                    logger.info(f"✅ Step checkpoint uploaded: gs://{self.gcs_bucket_name}/{gcs_full_path}")
                    
                    # Save metadata
                    self._save_checkpoint_metadata(filename, logs)
                    
                except Exception as e:
                    logger.error(f"❌ Step checkpoint GCS upload failed: {e}")
            else:
                logger.warning("⚠️ GCS not available for step checkpoint upload")
                
        except Exception as e:
            logger.error(f"❌ Step checkpoint save failed: {e}")
    
    def _save_checkpoint_metadata(self, filename: str, logs=None):
        """Save checkpoint metadata to GCS"""
        try:
            current_optimizer_step = self.model.optimizer.iterations.numpy() if hasattr(self.model.optimizer, 'iterations') else self.step_counter
            
            serializable_logs = {}
            if logs:
                serializable_logs = {
                    k: float(v.numpy()) if isinstance(v, tf.Tensor) else (
                        float(v) if isinstance(v, (np.float32, np.float16, float, int)) else str(v)
                    ) for k, v in logs.items()
                }
            
            metadata = {
                'epoch': self.current_epoch + 1,
                'step': self.step_counter,
                'optimizer_step': int(current_optimizer_step),
                'logs': serializable_logs,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': 'step_based'
            }
            
            meta_filename = filename.replace('.weights.h5', '_meta.json')
            meta_path = os.path.join(self.gcs_path, meta_filename)
            
            meta_blob = self._gcs_bucket.blob(meta_path)
            meta_blob.upload_from_string(json.dumps(metadata, indent=2))
            logger.info(f"✅ Step checkpoint metadata saved: {meta_filename}")
            
        except Exception as e:
            logger.error(f"❌ Step checkpoint metadata save failed: {e}")


class FSEBestModelCheckpointCallback(tf.keras.callbacks.Callback):
    """FIXED best model checkpoint callback based on validation metrics"""
    
    def __init__(self, checkpoint_dir: str, gcs_bucket_name: str, gcs_path: str,
                 monitor: str = 'val_loss', mode: str = 'min', gcs_project_id: Optional[str] = None):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.gcs_bucket_name = gcs_bucket_name
        self.gcs_path = gcs_path
        self.monitor = monitor
        self.mode = mode
        self.gcs_project_id = gcs_project_id
        
        # Initialize best value tracking
        if mode == 'min':
            self.best_value = float('inf')
            self.is_better = lambda new, best: new < best
        else:
            self.best_value = float('-inf')
            self.is_better = lambda new, best: new > best
        
        # Create local directory
        if self.checkpoint_dir and not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Initialize GCS bucket
        self._gcs_bucket = None
        self._init_gcs_bucket()
        
        logger.info(f"✅ FSE Best Model Checkpoint initialized: monitor={monitor}, mode={mode}")
    
    def _init_gcs_bucket(self):
        """Initialize GCS bucket with proper error handling"""
        try:
            # FIXED: Simple GCS client initialization
            client = storage.Client(project=self.gcs_project_id)
            self._gcs_bucket = client.bucket(self.gcs_bucket_name)
            logger.info(f"✅ Best model checkpoint GCS bucket connected: {self.gcs_bucket_name}")
        except Exception as e:
            logger.error(f"❌ Best model checkpoint GCS init failed: {e}")
            self._gcs_bucket = None
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint if current model is best"""
        if logs is None:
            return
        
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"⚠️ Monitor metric '{self.monitor}' not found in logs")
            return
        
        # Convert tensor to float if needed
        if isinstance(current_value, tf.Tensor):
            current_value = float(current_value.numpy())
        else:
            current_value = float(current_value)
        
        if self.is_better(current_value, self.best_value):
            self.best_value = current_value
            self._save_best_checkpoint(epoch, logs, current_value)
    
    def _save_best_checkpoint(self, epoch: int, logs: dict, metric_value: float):
        """Save best model checkpoint"""
        try:
            # FIXED: Generate filename with epoch and metric value
            safe_monitor = self.monitor.replace('val_', '').replace('/', '_')
            filename = f"fluxa_fse_best_{safe_monitor}_e{epoch+1:04d}_{metric_value:.6f}.weights.h5"
            local_path = os.path.join(self.checkpoint_dir, filename)
            
            logger.info(f"💾 Saving BEST model checkpoint: {filename}")
            logger.info(f"🎯 New best {self.monitor}: {metric_value:.6f} (was {self.best_value:.6f})")
            
            # Save weights locally
            self.model.save_weights(local_path)
            
            # Upload to GCS
            if self._gcs_bucket:
                try:
                    gcs_full_path = os.path.join(self.gcs_path, filename)
                    blob = self._gcs_bucket.blob(gcs_full_path)
                    blob.upload_from_filename(local_path)
                    logger.info(f"✅ Best model uploaded: gs://{self.gcs_bucket_name}/{gcs_full_path}")
                    
                    # Save metadata
                    self._save_best_metadata(filename, epoch, logs, metric_value)
                    
                except Exception as e:
                    logger.error(f"❌ Best model GCS upload failed: {e}")
            else:
                logger.warning("⚠️ GCS not available for best model upload")
                
        except Exception as e:
            logger.error(f"❌ Best model checkpoint save failed: {e}")
    
    def _save_best_metadata(self, filename: str, epoch: int, logs: dict, metric_value: float):
        """Save best model metadata to GCS"""
        try:
            current_optimizer_step = self.model.optimizer.iterations.numpy() if hasattr(self.model.optimizer, 'iterations') else 0
            
            serializable_logs = {}
            if logs:
                serializable_logs = {
                    k: float(v.numpy()) if isinstance(v, tf.Tensor) else (
                        float(v) if isinstance(v, (np.float32, np.float16, float, int)) else str(v)
                    ) for k, v in logs.items()
                }
            
            metadata = {
                'epoch': epoch + 1,
                'optimizer_step': int(current_optimizer_step),
                'monitor_metric': self.monitor,
                'monitor_value': metric_value,
                'mode': self.mode,
                'logs': serializable_logs,
                'timestamp': datetime.now().isoformat(),
                'checkpoint_type': 'best_model'
            }
            
            meta_filename = filename.replace('.weights.h5', '_meta.json')
            meta_path = os.path.join(self.gcs_path, meta_filename)
            
            meta_blob = self._gcs_bucket.blob(meta_path)
            meta_blob.upload_from_string(json.dumps(metadata, indent=2))
            logger.info(f"✅ Best model metadata saved: {meta_filename}")
            
        except Exception as e:
            logger.error(f"❌ Best model metadata save failed: {e}")


def load_latest_checkpoint_from_gcs(model, gcs_ckpt_dir, local_tmp_dir):
    """FIXED checkpoint loading with mixed precision optimizer compatibility"""
    gcs_bucket = _get_gcs_bucket()
    if not gcs_bucket:
        logger.warning("GCS N/A. Fresh training.")
        return 0
    
    try:
        # Look for checkpoints in the specified directory
        blobs = list(gcs_bucket.list_blobs(prefix=gcs_ckpt_dir + "/"))
        ckpt_files = [b for b in blobs if b.name.endswith((".h5", ".weights.h5")) and "_meta.json" not in b.name]
        
        if not ckpt_files:
            logger.info(f"No GCS checkpoints found in gs://{gcs_bucket.name}/{gcs_ckpt_dir}. Starting fresh.")
            return 0
        
        # Find the latest checkpoint by timestamp
        latest_blob = max(ckpt_files, key=lambda b: b.updated or datetime.min.replace(tzinfo=datetime.timezone.utc))
        
        # Create local temp directory
        if not os.path.exists(local_tmp_dir):
            os.makedirs(local_tmp_dir, exist_ok=True)
        
        # Download checkpoint
        local_path = os.path.join(local_tmp_dir, os.path.basename(latest_blob.name))
        logger.info(f"📥 Downloading checkpoint: gs://{gcs_bucket.name}/{latest_blob.name}")
        logger.info(f"📁 Local path: {local_path}")
        
        latest_blob.download_to_filename(local_path)
        
        # FIXED: Load weights with mixed precision compatibility
        try:
            # Use skip_mismatch and by_name to handle LossScaleOptimizer issues
            model.load_weights(local_path, skip_mismatch=True, by_name=True)
            logger.info(f"✅ Model weights loaded: {os.path.basename(local_path)}")
            logger.info("ℹ️ Note: Optimizer state skipped for mixed precision compatibility")
        except Exception as e_load:
            logger.warning(f"⚠️ Weight loading failed: {e_load}")
            logger.info("🔄 Starting fresh training instead")
            return 0
        
        # Try to extract epoch from metadata
        epoch = 0
        try:
            meta_name = latest_blob.name.replace('.weights.h5', '_meta.json').replace('.h5', '_meta.json')
            meta_blob = gcs_bucket.blob(meta_name)
            
            if meta_blob.exists():
                meta_info = json.loads(meta_blob.download_as_string())
                epoch = int(meta_info.get('epoch', 0))
                logger.info(f"📊 Resuming from epoch {epoch} (from metadata)")
            else:
                # Try to extract epoch from filename
                filename_parts = os.path.basename(local_path).split('_')
                for i, part in enumerate(filename_parts):
                    if part.startswith('e') and part[1:].isdigit():
                        epoch = int(part[1:])
                        break
                    elif part == 'epoch' and i + 1 < len(filename_parts) and filename_parts[i + 1].isdigit():
                        epoch = int(filename_parts[i + 1])
                        break
                
                if epoch > 0:
                    logger.info(f"📊 Resuming from epoch {epoch} (from filename)")
                else:
                    logger.info("📊 Could not determine epoch from checkpoint, starting from 0")
                    
        except Exception as e_meta:
            logger.warning(f"⚠️ Error extracting epoch info: {e_meta}. Starting from epoch 0.")
            epoch = 0
        
        return epoch
        
    except Exception as e:
        logger.error(f"❌ Checkpoint loading failed: {e}. Starting fresh training.", exc_info=True)
        return 0

# --- Main Training Function ---
def run_training(args_obj: argparse.Namespace, strategy: tf.distribute.Strategy, num_replicas: int):
    logger.info("="*80 + f"\n🚀 ENHANCED MULTI-GPU FSE TRAINING INITIATED (v3.12 - Enhanced Segmentation Loss) 🚀\n" + "="*80) 
    _initialize_gcs_client() 
    logger.info("--- Data Preparation (Optimized Pipeline v3.12) ---")
    
    all_available_ids = find_main_dataset_ids()
    if not all_available_ids: logger.error("No common data IDs. Aborting."); return
    random.shuffle(all_available_ids)

    num_val_samples_to_take = int(len(all_available_ids) * args_obj.validation_split_fraction)
    if args_obj.max_val_samples is not None: num_val_samples_to_take = min(num_val_samples_to_take, args_obj.max_val_samples)
    
    min_train_needed = (args_obj.base_batch_size_per_gpu * num_replicas if num_replicas > 0 else args_obj.base_batch_size_per_gpu) or 1
    if len(all_available_ids) - num_val_samples_to_take < min_train_needed:
        num_val_samples_to_take = max(0, len(all_available_ids) - min_train_needed)
        logger.warning(f"Validation split adjusted to {num_val_samples_to_take} for min training samples.")
    
    val_ids = []; train_ids = all_available_ids
    if num_val_samples_to_take > 0 :
        val_ids = all_available_ids[:num_val_samples_to_take]
        train_ids = all_available_ids[num_val_samples_to_take:]
    elif len(all_available_ids) <= min_train_needed : 
        logger.warning("Not enough total samples for a validation split while preserving a training batch. Using all data for training, no validation.")
        val_ids = [] 
    
    logger.info(f"Data split: {len(train_ids)} training IDs, {len(val_ids)} validation IDs.")

    if args_obj.max_train_samples is not None and len(train_ids) > args_obj.max_train_samples:
        train_ids = train_ids[:args_obj.max_train_samples] 
        logger.info(f"Limited training samples to {len(train_ids)} after split.")
    if not train_ids: logger.error("No training IDs after split/max_samples. Aborting."); return

    effective_gpus = num_replicas if num_replicas > 0 else 0
    global_train_bs = calculate_optimal_batch_size(effective_gpus, args_obj.base_batch_size_per_gpu)
    global_val_bs = calculate_optimal_batch_size(effective_gpus, args_obj.val_batch_size_per_gpu) if val_ids else 0
    logger.info(f"Global train batch: {global_train_bs}, Global val batch: {global_val_bs if val_ids else 'N/A'}")

    train_dataset = create_optimized_tf_dataset(train_ids, global_train_bs, is_training_dataset=True)
    val_dataset = create_optimized_tf_dataset(val_ids, global_val_bs, is_training_dataset=False) if val_ids and global_val_bs > 0 else None
    
    logger.info("Verifying training data pipeline...")
    try:
        test_in, test_lbl = next(iter(train_dataset.take(1)))
        if isinstance(test_in,tuple): logger.info(f" Train Img: {test_in[0].shape} {test_in[0].dtype}, Syntha: {test_in[1].shape} {test_in[1].dtype}")
        else: logger.info(f" Train Img: {test_in.shape} {test_in.dtype}")
        for n,t in test_lbl.items(): logger.info(f" Train Lbl '{n}': {t.shape} {t.dtype}")
        logger.info("Training data pipeline OK.")
    except Exception as e_data: 
        logger.error(f"Training data pipeline test FAILED: {e_data}", exc_info=True)
        if "NOT_FOUND" in str(e_data).upper():
            logger.error(">>> GCS File NOT FOUND during data pipeline test. Please verify all data paths and file existence for the listed IDs in `find_main_dataset_ids()` logs.")
        return

    if val_dataset:
        logger.info("Verifying validation data pipeline...")
        try: 
            val_test_in, val_test_lbl = next(iter(val_dataset.take(1)))
            logger.info("Validation data pipeline OK.")
        except Exception as e_data_val: 
            logger.error(f"Validation data pipeline test FAILED: {e_data_val}", exc_info=True)
            if "NOT_FOUND" in str(e_data_val).upper():
                logger.error(">>> GCS File NOT FOUND during validation data pipeline test. Please verify all data paths and file existence for the `val_ids` being used.")
            val_dataset = None; logger.warning("Validation disabled due to pipeline error.")

    if num_replicas > 1: 
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        if val_dataset: val_dataset = strategy.experimental_distribute_dataset(val_dataset)
        logger.info("Datasets distributed.")

    logger.info("--- Model Creation & Compilation (Enhanced Loss Configuration) ---")
    with strategy.scope():
        model = FLUXA_FSE_Native(
            input_shape=(args_obj.img_height,args_obj.img_width,3), base_channels=args_obj.base_channels,
            field_evolution_rate=args_obj.field_evolution_rate, enable_syntha_integration=args_obj.enable_syntha,
            max_cses_per_fil=args_obj.max_cses_per_fil
        )
        policy = tf.keras.mixed_precision.global_policy(); dummy_dtype = tf.float16 if policy.compute_dtype == 'float16' else tf.float32
        dummy_img = tf.zeros((1,args_obj.img_height,args_obj.img_width,3),dtype=dummy_dtype)
        build_feed = (dummy_img, tf.zeros((1,SYNTHA_CONTEXT_WIDTH),dtype=dummy_dtype)) if args_obj.enable_syntha else dummy_img
        try: logger.info(f"Building model with dummy input {dummy_dtype}..."); _ = model(build_feed, training=False); logger.info("Model built.")
        except Exception as e_build: logger.error(f"Model build failed: {e_build}", exc_info=True); return
        
        scaled_lr = scale_learning_rate(args_obj.learning_rate,global_train_bs,args_obj.lr_ref_batch_size,args_obj.lr_scaling_method)
        opt_base = tf.keras.optimizers.Adam(learning_rate=scaled_lr,beta_1=0.9,beta_2=0.999,epsilon=1e-7, clipnorm=1.0)
        if args_obj.gradient_accumulation_steps > 1: logger.info(f"Grad Accum ON: {args_obj.gradient_accumulation_steps} steps."); opt = GradientAccumulationOptimizer(optimizer=opt_base, accumulation_steps=args_obj.gradient_accumulation_steps)
        else: logger.info("Grad Accum OFF."); opt = opt_base
        opt = tf.keras.mixed_precision.LossScaleOptimizer(opt)
        
        # ENHANCED: Configure segmentation loss based on command line arguments
        logger.info(f"🔧 Configuring segmentation loss: {args_obj.segmentation_loss_type}")
        
        if args_obj.segmentation_loss_type == 'focal':
            segmentation_loss_fn = functools.partial(focal_loss_binary_segmentation, 
                                                   alpha=args_obj.focal_loss_alpha, 
                                                   gamma=args_obj.focal_loss_gamma)
            logger.info(f"   Using Focal Loss: alpha={args_obj.focal_loss_alpha}, gamma={args_obj.focal_loss_gamma}")
        
        elif args_obj.segmentation_loss_type == 'dice':
            segmentation_loss_fn = dice_loss
            logger.info(f"   Using Dice Loss")
        
        elif args_obj.segmentation_loss_type == 'combined_dice_focal':
            segmentation_loss_fn = functools.partial(combined_dice_focal_loss_segmentation,
                                                    dice_weight=args_obj.seg_dice_weight,
                                                    focal_weight=args_obj.seg_focal_weight,
                                                    focal_alpha=args_obj.focal_loss_alpha,
                                                    focal_gamma=args_obj.focal_loss_gamma)
            logger.info(f"   Using Combined Dice+Focal Loss: dice_weight={args_obj.seg_dice_weight:.3f}, focal_weight={args_obj.seg_focal_weight:.3f}")
        
        else:
            # Fallback to focal loss
            segmentation_loss_fn = functools.partial(focal_loss_binary_segmentation, 
                                                   alpha=args_obj.focal_loss_alpha, 
                                                   gamma=args_obj.focal_loss_gamma)
            logger.warning(f"Unknown segmentation loss type: {args_obj.segmentation_loss_type}. Using focal loss.")
        
        # Enhanced loss configuration
        losses_cfg = { 
            'fluxa_keypoints': functools.partial(stable_fse_loss_wrapper_fn, field_smoothness_val=0.01),
            'fluxa_segmentation': segmentation_loss_fn,  # ENHANCED: Use configured segmentation loss
            'fluxa_surface_normals': functools.partial(stable_fse_loss_wrapper_fn, field_smoothness_val=0.02), 
            'fluxa_environment_lighting': per_sample_mse_loss_fn
        }
        loss_weights_cfg = {
            'fluxa_keypoints': args_obj.keypoints_loss_weight, 
            'fluxa_segmentation': args_obj.segmentation_loss_weight,  # ENHANCED: Increased default weight
            'fluxa_surface_normals': args_obj.surface_normals_loss_weight, 
            'fluxa_environment_lighting': args_obj.env_lighting_loss_weight
        }
        
        logger.info(f"🎯 Loss Configuration:")
        logger.info(f"   • Keypoints: weight={args_obj.keypoints_loss_weight:.1f}, loss=stable_fse")
        logger.info(f"   • Segmentation: weight={args_obj.segmentation_loss_weight:.1f}, loss={args_obj.segmentation_loss_type}")
        logger.info(f"   • Surface Normals: weight={args_obj.surface_normals_loss_weight:.1f}, loss=stable_fse")
        logger.info(f"   • Environment: weight={args_obj.env_lighting_loss_weight:.1f}, loss=mse")
        
        metrics_cfg = {
            'fluxa_keypoints': [tf.keras.metrics.MeanAbsoluteError(name='kp_mae'), FixedFSENativeMetric(name='kp_fse_coherence')],
            'fluxa_segmentation': [tf.keras.metrics.BinaryAccuracy(name='seg_acc'), tf.keras.metrics.Precision(name='seg_p'), tf.keras.metrics.Recall(name='seg_r'), tf.keras.metrics.MeanIoU(num_classes=2, name='seg_miou')],
            'fluxa_surface_normals': [tf.keras.metrics.MeanAbsoluteError(name='sn_mae'), FixedFSENativeMetric(name='sn_fse_coherence'), PhysicsGroundedAccuracy(name='sn_physics_accuracy')],
            'fluxa_environment_lighting': [tf.keras.metrics.MeanAbsoluteError(name='env_mae')]
        }
        model.compile(optimizer=opt, loss=losses_cfg, loss_weights=loss_weights_cfg, metrics=metrics_cfg)
        logger.info("✅ Model compiled with enhanced segmentation loss."); model.summary(line_length=150, expand_nested=True)

    # FIXED: Setup checkpoint directories and loading
    local_ckpt_dir = "/tmp/fluxa_fse_local_ckpts_v3_12"
    os.makedirs(local_ckpt_dir, exist_ok=True)
    
    # FIXED: Use the specified GCS checkpoint path
    gcs_ckpt_dir_in_bucket = "fluxa_fse_native_enhanced/checkpoints"
    
    # FIXED: Load latest checkpoint with proper epoch tracking
    initial_epoch = load_latest_checkpoint_from_gcs(model, gcs_ckpt_dir_in_bucket, local_ckpt_dir)
    
    # FIXED: Create both checkpoint callbacks
    callbacks_list = []
    
    # FIXED: Step-based checkpoint callback (every 8000 steps)
    step_checkpoint_callback = FSEStepCheckpointCallback(
        checkpoint_dir=local_ckpt_dir,
        gcs_bucket_name=args_obj.bucket_name,
        gcs_path=gcs_ckpt_dir_in_bucket,
        save_every_steps=args_obj.checkpoint_save_steps,
        gcs_project_id=args_obj.project_id
    )
    callbacks_list.append(step_checkpoint_callback)
    
    # FIXED: Best model checkpoint callback (based on validation metric)
    monitor_metric = args_obj.checkpoint_save_best_monitor if val_dataset else 'loss' 
    monitor_mode = 'max' if 'miou' in monitor_metric or 'acc' in monitor_metric else 'min'
    
    best_model_callback = FSEBestModelCheckpointCallback(
        checkpoint_dir=local_ckpt_dir,
        gcs_bucket_name=args_obj.bucket_name,
        gcs_path=gcs_ckpt_dir_in_bucket,
        monitor=monitor_metric,
        mode=monitor_mode,
        gcs_project_id=args_obj.project_id
    )
    callbacks_list.append(best_model_callback)

    callbacks_list.extend([
        tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor_metric, mode=monitor_mode, factor=0.5, patience=5, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor=monitor_metric, mode=monitor_mode, patience=15, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(args_obj.job_dir,"logs_v3_12",datetime.now().strftime("%Y%m%d-%H%M%S")),histogram_freq=0,write_graph=False,profile_batch='100,120' if num_replicas > 0 else 0) 
    ])

    logger.info(f"--- Training Start (Epochs: {args_obj.epochs}, Initial Epoch: {initial_epoch}) ---")
    steps_epoch = len(train_ids)//global_train_bs if train_ids and global_train_bs > 0 else None
    val_steps = len(val_ids)//global_val_bs if val_ids and global_val_bs > 0 and val_dataset else None
    
    if not steps_epoch or steps_epoch == 0: logger.error(f"steps_per_epoch is {steps_epoch}. Abort."); return
    logger.info(f"steps_per_epoch={steps_epoch}, validation_steps={val_steps if val_dataset else 'N/A'}")
    
    # ENHANCED: Log enhanced segmentation configuration
    logger.info("📁 ENHANCED Training Configuration:")
    logger.info(f"   - Segmentation loss type: {args_obj.segmentation_loss_type}")
    logger.info(f"   - Segmentation loss weight: {args_obj.segmentation_loss_weight:.1f}")
    logger.info(f"   - Step checkpoints: Every {args_obj.checkpoint_save_steps} steps")
    logger.info(f"   - Best model checkpoints: Monitor {monitor_metric} ({monitor_mode})")
    logger.info(f"   - GCS path: gs://{args_obj.bucket_name}/{gcs_ckpt_dir_in_bucket}")
    logger.info(f"   - Local dir: {local_ckpt_dir}")
    
    try:
        model.fit(train_dataset,epochs=args_obj.epochs,initial_epoch=initial_epoch,
                  validation_data=val_dataset if val_dataset else None, 
                  callbacks=callbacks_list,steps_per_epoch=steps_epoch,
                  validation_steps=val_steps,verbose=1)
        logger.info("✅🚀 ENHANCED Training (v3.12) completed!")
    except Exception as e_fit: 
        logger.error(f"❌ ENHANCED Training (v3.12) failed: {e_fit}", exc_info=True)
        if "ResourceExhaustedError" in str(e_fit) or "OOM" in str(e_fit).upper(): logger.error(">>> OOM Error")
        elif "Nan" in str(e_fit) or "NaN" in str(e_fit) or "nan" in str(e_fit): logger.error(">>> NaN Detected")
        elif "NOT_FOUND" in str(e_fit).upper(): logger.error(">>> GCS File NOT FOUND error. Check data paths and integrity.")
        return 
    
    # FIXED: Save final model
    final_model_path=os.path.join(local_ckpt_dir,"fluxa_fse_final_v3_12.weights.h5")
    model.save_weights(final_model_path)
    logger.info(f"💾 Final weights (v3.12) saved: {final_model_path}")
    
    gcs_bkt_final=_get_gcs_bucket()
    if gcs_bkt_final:
        gcs_final_path=os.path.join(gcs_ckpt_dir_in_bucket,"fluxa_fse_final_v3_12.weights.h5")
        try: 
            gcs_bkt_final.blob(gcs_final_path).upload_from_filename(final_model_path)
            logger.info(f"✅ Final weights (v3.12) uploaded: gs://{args_obj.bucket_name}/{gcs_final_path}")
        except Exception as e_upload: 
            logger.error(f"❌ Final weights (v3.12) GCS upload fail: {e_upload}",exc_info=True)
    else: 
        logger.warning("GCS N/A for final model (v3.12) upload.")
    
    logger.info("🎉🎉🎉 ENHANCED FSE NATIVE FLUXA TRAINING (v3.12 - Enhanced Segmentation Loss) FINISHED! 🎉🎉🎉")

if __name__ == '__main__':
    _initialize_gcs_client() 
    strategy, num_dev = setup_gpu_strategy()
    logger.info("Effective arguments for this run (v3.12 - Enhanced Segmentation Loss):")
    for arg_n, val in sorted(vars(args).items()): logger.info(f"  --{arg_n}={val}")
    run_training(args, strategy, num_dev)