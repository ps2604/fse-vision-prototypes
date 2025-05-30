#!/usr/bin/env python3
"""
Auralith FLUXA-FSE Training Script - FIXED VERSION
Chunk 1: Setup and Imports
"""

import os
import sys
import time
import io
import json
import random
import logging
import argparse
import numpy as np
import tensorflow as tf
import cv2
from datetime import datetime
from google.cloud import storage
from tensorflow.keras import layers, regularizers

# Import FIXED FSE components
from fse_core import (
    FLIT, CSE, FIL, FSEBlock, 
    create_fse_backbone, create_fse_upsampling_block,
    create_fse_skip_connection
)

# REMOVED mixed precision to avoid dtype issues
# mixed_precision.set_global_policy("mixed_float16")   # COMMENTED OUT

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory optimization settings
for device in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(device, True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

def parse_args():
    parser = argparse.ArgumentParser(description='Train FLUXA-FSE module on Vertex AI')
    
    # Core cloud arguments
    parser.add_argument('--project-id', type=str, default='bright-link-455716-h0')
    parser.add_argument('--bucket-name', type=str, default='auralith')
    parser.add_argument('--base-path', type=str, default='fluxa_fse')
    
    # Training settings - REDUCED for memory efficiency
    parser.add_argument('--batch-size', type=int, default=4)  # Reduced from 8
    parser.add_argument('--val-batch-size', type=int, default=8)  # Reduced from 16
    parser.add_argument('--epochs', type=int, default=50)  # Reduced from 100
    parser.add_argument('--initial-epoch', type=int, default=0)
    parser.add_argument('--learning-rate', type=float, default=0.0002)  # Reduced for stability
    parser.add_argument('--checkpoint-steps', type=int, default=1000)  # Less frequent saves
    
    # Data limits - REDUCED
    parser.add_argument('--max-samples', type=int, default=10000)  # Reduced from 50000
    parser.add_argument('--max-val-samples', type=int, default=1000)  # Reduced from 5000
    
    # FSE-specific parameters - MEMORY OPTIMIZED
    parser.add_argument('--base-filters', type=int, default=16)  # Reduced from 32
    parser.add_argument('--dropout-rate', type=float, default=0.1)  # Reduced from 0.15
    parser.add_argument('--l2-reg', type=float, default=1e-5)  # Slightly reduced
    parser.add_argument('--field-evolution-rate', type=float, default=0.05)  # Reduced for stability
    
    # Loss weights
    parser.add_argument('--keypoints-weight', type=float, default=1.0)
    parser.add_argument('--segmentation-weight', type=float, default=2.0)  # Reduced from 3.0
    parser.add_argument('--surface-normals-weight', type=float, default=0.5)  # Reduced from 0.8
    parser.add_argument('--env-lighting-weight', type=float, default=1.0)  # Reduced from 2.0
    
    # Controls
    parser.add_argument('--early-stopping-patience', type=int, default=10)  # Reduced from 15
    parser.add_argument('--use-validation', action='store_true')
    parser.add_argument('--skip-checkpoint', action='store_true')
    
    return parser.parse_args()

# Parse args once globally
args = parse_args()

# GCS paths and setup - UNCHANGED
PROJECT_ID = args.project_id
GCS_BUCKET_NAME = args.bucket_name
GCS_BASE_PATH = args.base_path
LOCAL_TEMP_DIR = "/tmp/auralith_fse"

GCS_IMAGES_PATH = "fluxa/images"
GCS_KEYPOINTS_PATH = "fluxa/keypoints"
GCS_MASKS_PATH = "fluxa/segmentation_masks"
GCS_SURFACE_NORMALS_PATH = "fluxa/surface_normals"
GCS_ENV_LIGHTING_PATH = "fluxa/environment_lighting"
GCS_CHECKPOINT_DIR = "fluxa/checkpoints"

# Validation paths
GCS_VAL_IMAGES_PATH = "fluxa/val/images"
GCS_VAL_KEYPOINTS_PATH = "fluxa/val/keypoints"
GCS_VAL_MASKS_PATH = "fluxa/val/segmentation_masks"
GCS_VAL_SURFACE_NORMALS_PATH = "fluxa/val/surface_normals"

os.makedirs(LOCAL_TEMP_DIR, exist_ok=True)

# Connect to GCS - UNCHANGED
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(GCS_BUCKET_NAME)
    logger.info(f"✅ Connected to GCS bucket '{GCS_BUCKET_NAME}' successfully")
except Exception as e:
    logger.error(f"❌ Failed to connect to GCS: {e}")
    sys.exit(1)

# =====================================
# Data Loading Functions - UNCHANGED from original
# =====================================

def list_available_samples(prefix_path, file_extension='.npy'):
    """List samples with matching file type in GCS"""
    try:
        blobs = list(bucket.list_blobs(prefix=f"{prefix_path}/"))
        files = {os.path.splitext(os.path.basename(b.name))[0]
                 for b in blobs if b.name.endswith(file_extension)}
        return files
    except Exception as e:
        logger.error(f"❌ Error listing from {prefix_path}: {e}")
        return set()

def find_common_samples(training=True):
    """Find samples that have image, keypoints, mask, normals"""
    if training:
        kp = list_available_samples(GCS_KEYPOINTS_PATH, '.npy')
        mask = list_available_samples(GCS_MASKS_PATH, '.png')
        normals = list_available_samples(GCS_SURFACE_NORMALS_PATH, '.npy')
    else:
        kp = list_available_samples(GCS_VAL_KEYPOINTS_PATH, '.npy')
        mask = list_available_samples(GCS_VAL_MASKS_PATH, '.png')
        normals = list_available_samples(GCS_VAL_SURFACE_NORMALS_PATH, '.npy')

    return list(kp & mask & normals)

def download_image_from_gcs(image_id, training=True):
    prefix = GCS_IMAGES_PATH if training else GCS_VAL_IMAGES_PATH
    for ext in ['.jpg', '.png']:
        blob_path = f"{prefix}/{image_id}{ext}"
        blob = bucket.blob(blob_path)
        if blob.exists():
            try:
                image_bytes = blob.download_as_bytes()
                img = tf.image.decode_image(image_bytes, channels=3)
                img = tf.image.resize(img, (480, 640))
                return tf.cast(img, tf.float32) / 255.0
            except Exception as e:
                logger.warning(f"⚠️ Error decoding image {image_id}: {e}")
    return tf.zeros((480, 640, 3), dtype=tf.float32)

def load_keypoints_from_gcs(image_id, training=True):
    prefix = GCS_KEYPOINTS_PATH if training else GCS_VAL_KEYPOINTS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.npy")
    if blob.exists():
        keypoints_bytes = blob.download_as_bytes()
        keypoints = np.load(io.BytesIO(keypoints_bytes))
        return tf.convert_to_tensor(keypoints, dtype=tf.float32)
    return tf.zeros((480, 640, 17), dtype=tf.float32)

def load_segmentation_mask_from_gcs(image_id, training=True):
    prefix = GCS_MASKS_PATH if training else GCS_VAL_MASKS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.png")
    if blob.exists():
        try:
            mask_bytes = blob.download_as_bytes()
            mask = tf.image.decode_png(mask_bytes, channels=1)
            mask = tf.image.resize(mask, (480, 640))
            return tf.cast(mask, tf.float32) / 255.0
        except Exception as e:
            logger.warning(f"⚠️ Mask decoding error for {image_id}: {e}")
    return tf.zeros((480, 640, 1), dtype=tf.float32)

def load_surface_normals_from_gcs(image_id, training=True):
    prefix = GCS_SURFACE_NORMALS_PATH if training else GCS_VAL_SURFACE_NORMALS_PATH
    blob = bucket.blob(f"{prefix}/{image_id}.npy")
    if blob.exists():
        normals_bytes = blob.download_as_bytes()
        normals = np.load(io.BytesIO(normals_bytes))
        if normals.shape != (480, 640, 3):
            normals = cv2.resize(normals, (640, 480))
        min_val, max_val = np.min(normals), np.max(normals)
        if max_val > 1.0 or min_val < -1.0:
            normals = 2.0 * (normals - min_val) / (max_val - min_val + 1e-8) - 1.0
        return tf.convert_to_tensor(normals, dtype=tf.float32)
    return tf.ones((480, 640, 3), dtype=tf.float32) * tf.constant([0.0, 0.0, 1.0])

def load_environment_lighting_from_gcs():
    """Load random environment lighting (9 SH coefficients)"""
    try:
        blobs = list(bucket.list_blobs(prefix=GCS_ENV_LIGHTING_PATH, max_results=100))
        env_files = [b for b in blobs if b.name.endswith('.json')]
        if not env_files:
            return tf.random.normal((9,), mean=0.5, stddev=0.1)
        random_env = random.choice(env_files)
        env_json = json.loads(random_env.download_as_string())
        sh = np.mean(np.array(env_json['spherical_harmonics']), axis=0)
        sh = (sh - sh.min()) / (sh.max() - sh.min() + 1e-8)
        return tf.convert_to_tensor(sh, dtype=tf.float32)
    except Exception as e:
        logger.warning(f"⚠️ Env lighting load error: {e}")
        return tf.random.normal((9,), mean=0.5, stddev=0.1)

def apply_augmentation(image, keypoints, segmentation, surface_normals):
    """FSE-aware augmentation that preserves continuous field properties"""
    if tf.random.uniform(()) > 0.8:  # Less aggressive augmentation for FSE
        return image, keypoints, segmentation, surface_normals

    # Continuous field-preserving augmentations
    flip_lr = tf.random.uniform(()) > 0.5
    brightness = tf.random.uniform((), -0.03, 0.03)  # Very reduced range
    contrast = tf.random.uniform((), 0.97, 1.03)     # Very reduced range

    if flip_lr:
        image = tf.image.flip_left_right(image)
        keypoints = tf.image.flip_left_right(keypoints)
        segmentation = tf.image.flip_left_right(segmentation)
        surface_normals = tf.image.flip_left_right(surface_normals)
        surface_normals = tf.stack([
            -surface_normals[..., 0],  # flip X
            surface_normals[..., 1],
            surface_normals[..., 2]
        ], axis=-1)

    image = tf.image.adjust_brightness(image, brightness)
    image = tf.image.adjust_contrast(image, contrast)
    return tf.clip_by_value(image, 0.0, 1.0), keypoints, segmentation, surface_normals

def data_generator(sample_ids, batch_size=8, training=True):
    """FIXED generator - prevents infinite restart loops"""
    samples = sample_ids.copy()
    
    def _single_epoch():
        """Generate exactly one epoch of data"""
        if training:
            random.shuffle(samples)
        
        generated_batches = 0
        target_batches = len(samples) // batch_size
        
        logger.info(f"🎯 Target batches for this epoch: {target_batches}")
        
        for i in range(0, len(samples), batch_size):
            if generated_batches >= target_batches:
                break  # CRITICAL: Prevent infinite generation
                
            batch_ids = samples[i:i+batch_size]
            imgs, kps, masks, norms, lights = [], [], [], [], []

            for img_id in batch_ids:
                try:
                    img = download_image_from_gcs(img_id, training)
                    kp = load_keypoints_from_gcs(img_id, training)
                    mask = load_segmentation_mask_from_gcs(img_id, training)
                    norm = load_surface_normals_from_gcs(img_id, training)

                    if training:
                        img, kp, mask, norm = apply_augmentation(img, kp, mask, norm)

                    imgs.append(img)
                    kps.append(kp)
                    masks.append(mask)
                    norms.append(norm)
                except Exception as e:
                    logger.warning(f"⚠️ Failed loading {img_id}: {e}")

            if len(imgs) == 0:
                continue

            env_light = load_environment_lighting_from_gcs()
            lights = [env_light] * len(imgs)

            generated_batches += 1
            
            yield tf.stack(imgs), {
                'fluxa_keypoints': tf.stack(kps),
                'fluxa_segmentation': tf.stack(masks),
                'fluxa_surface_normals': tf.stack(norms),
                'fluxa_environment_lighting': tf.stack(lights)
            }
        
        logger.info(f"✅ Epoch complete: {generated_batches} batches generated")

    # Return single epoch generator - Keras handles epoch repetition
    return _single_epoch()


def create_fluxa_dataset(sample_ids, batch_size=8, training=True):
    """FIXED dataset creation - no infinite loops"""
    
    logger.info(f"🏗️ Creating {'training' if training else 'validation'} dataset")
    logger.info(f"   - Samples: {len(sample_ids)}")
    logger.info(f"   - Batch size: {batch_size}")
    
    # Create dataset from generator
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(sample_ids, batch_size=batch_size, training=training),
        output_signature=(
            tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32),
            {
                'fluxa_keypoints': tf.TensorSpec(shape=(None, 480, 640, 17), dtype=tf.float32),
                'fluxa_segmentation': tf.TensorSpec(shape=(None, 480, 640, 1), dtype=tf.float32),
                'fluxa_surface_normals': tf.TensorSpec(shape=(None, 480, 640, 3), dtype=tf.float32),
                'fluxa_environment_lighting': tf.TensorSpec(shape=(None, 9), dtype=tf.float32)
            }
        )
    )
    
    # Repeat for multiple epochs - this is KEY
    dataset = dataset.repeat()  # This handles epoch repetition
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    logger.info("✅ Dataset created with proper epoch handling")
    return dataset
# =====================================
# FLUXA-FSE Architecture - COMPLETELY FIXED
# =====================================

class FLUXA_FSE(tf.keras.Model):
    """
    FLUXA FSE - FIXED with proper decoder sizing
    """
    
    def __init__(self,
                 input_shape=(480, 640, 3),
                 base_filters=16,
                 l2_reg=1e-5,
                 dropout_rate=0.1,
                 field_evolution_rate=0.05,
                 name="fluxa_fse",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        # Store hyperparameters
        self.input_shape_dims = input_shape
        self.base_filters = base_filters
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.field_evolution_rate = field_evolution_rate

        # Create backbone
        self.backbone = create_fse_backbone(
            input_shape=input_shape,
            base_filters=base_filters,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
        )

        # Create analysis branches
        self.env_perception = self._build_environmental_perception()
        self.keypoint_analysis = self._build_keypoint_analysis()
        self.segmentation_analysis = self._build_segmentation_analysis()
        self.surface_normal_analysis = self._build_surface_normal_analysis()
        self.env_lighting_analysis = self._build_env_lighting_analysis()

        # FIXED: Only 3 decoder stages (not 4)
        # This gives us exactly 8x upsampling to match 8x downsampling from backbone
        self.dec_stage1 = self._make_decoder_stage(
            filters=self.base_filters * 4,
            size=(2, 2),
            name="dec_stage1"
        )
        self.dec_stage2 = self._make_decoder_stage(
            filters=self.base_filters * 2,
            size=(2, 2),
            name="dec_stage2"
        )
        self.dec_stage3 = self._make_decoder_stage(
            filters=self.base_filters,
            size=(2, 2),
            name="dec_stage3"
        )
        # dec_stage4 REMOVED - was causing over-upsampling

        # Output heads remain the same
        self.keypoint_head = layers.Conv2D(
            17, kernel_size=1,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="fluxa_keypoints"
        )
        self.segmentation_head = layers.Conv2D(
            1, kernel_size=1, activation="sigmoid",
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="fluxa_segmentation"
        )
        self.surface_normal_head = layers.Conv2D(
            3, kernel_size=1, activation="tanh",
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="fluxa_surface_normals"
        )

    def _make_decoder_stage(self, filters, size, name):
        """Create decoder stage - all in __init__"""
        return tf.keras.Sequential([
            layers.UpSampling2D(size=size, name=f"{name}_up"),
            FSEBlock(
                filters=filters,
                num_fils=1,  # Single FIL for memory efficiency
                fil_types=['continuous'],
                l2_reg=self.l2_reg,
                dropout_rate=self.dropout_rate if 'stage1' in name else 0.0,
                name=f"{name}_fse"
            ),
            layers.BatchNormalization(name=f"{name}_bn"),
            layers.Activation('relu', name=f"{name}_act"),
        ], name=name)

    def _build_environmental_perception(self):
        """Build environmental perception - SIMPLIFIED"""
        return tf.keras.Sequential([
            FIL(
                num_cses=2,  # Reduced from 8
                cse_config={
                    'num_flits': 2,  # Reduced
                    'flit_channels': self.base_filters * 2,  # Reduced
                    'evolution_rate': self.field_evolution_rate,
                    'coherence_type': 'adaptive'
                },
                field_type='continuous',
                spatial_kernel_size=3,  # Reduced from 5
                l2_reg=self.l2_reg,
                dropout_rate=self.dropout_rate,
                name='env_perception_fil'
            )
        ], name='environmental_perception')

    def _build_keypoint_analysis(self):
        """Build continuous keypoint extraction - SIMPLIFIED"""
        return tf.keras.Sequential([
            FIL(
                num_cses=4,  # Reduced from 17
                cse_config={
                    'num_flits': 2,
                    'flit_channels': self.base_filters,
                    'evolution_rate': self.field_evolution_rate * 1.2,
                    'coherence_type': 'adaptive'
                },
                field_type='continuous',
                spatial_kernel_size=3,  # Reduced from 7
                l2_reg=self.l2_reg,
                name='keypoint_continuous_extraction'
            )
        ], name='keypoint_analysis')

    def _build_segmentation_analysis(self):
        """Build continuous boundary field detection - SIMPLIFIED"""
        return tf.keras.Sequential([
            FIL(
                num_cses=2,  # Reduced from 8
                cse_config={
                    'num_flits': 2,  # Reduced
                    'flit_channels': self.base_filters,
                    'evolution_rate': self.field_evolution_rate,
                    'coherence_type': 'adaptive'
                },
                field_type='continuous',
                spatial_kernel_size=3,  # Reduced from 5
                l2_reg=self.l2_reg,
                dropout_rate=self.dropout_rate * 0.5,
                name='boundary_field_detection'
            )
        ], name='segmentation_analysis')

    def _build_surface_normal_analysis(self):
        """Build continuous surface geometry understanding - SIMPLIFIED"""
        return tf.keras.Sequential([
            FIL(
                num_cses=3,  # Reduced from 12
                cse_config={
                    'num_flits': 2,  # Reduced
                    'flit_channels': self.base_filters,  # Reduced
                    'evolution_rate': self.field_evolution_rate * 0.8,
                    'coherence_type': 'adaptive'
                },
                field_type='continuous',
                spatial_kernel_size=3,  # Reduced from 7
                l2_reg=self.l2_reg,
                name='surface_geometry_field'
            )
        ], name='surface_normal_analysis')

    def _build_env_lighting_analysis(self):
        """Build ambient lighting analysis - SIMPLIFIED"""
        return tf.keras.Sequential([
            # Global pooling and dense layers
            layers.GlobalAveragePooling2D(name='global_field_pool'),
            layers.Dense(
                self.base_filters,  # Reduced
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name='sh_projection_1'
            ),
            layers.Dropout(self.dropout_rate, name='sh_dropout'),
            layers.Dense(
                9,  # 9 SH coefficients
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name='spherical_harmonics_output'
            )
        ], name='env_lighting_analysis')

    def call(self, inputs, training=None):
        """
        Forward pass - CLEAN VERSION (no resizing needed)
        """
        # 1. Backbone features: (480,640,3) → (60,80,channels)
        feat = self.backbone(inputs, training=training)

        # 2. Environmental perception
        env_feat = self.env_perception(feat, training=training)

        # 3. Progressive decoder: (60,80) → (120,160) → (240,320) → (480,640)
        x = self.dec_stage1(feat, training=training)    # → (120,160)
        x = self.dec_stage2(x, training=training)       # → (240,320)  
        dec_out = self.dec_stage3(x, training=training) # → (480,640) ✅

        # 4. Task-specific heads - outputs are already correct size!
        kp_feat = self.keypoint_analysis(dec_out, training=training)
        keypoints = self.keypoint_head(kp_feat, training=training)

        seg_feat = self.segmentation_analysis(dec_out, training=training)
        segmentation = self.segmentation_head(seg_feat, training=training)

        norm_feat = self.surface_normal_analysis(dec_out, training=training)
        surface_normals = self.surface_normal_head(norm_feat, training=training)

        env_lighting = self.env_lighting_analysis(env_feat, training=training)

        # No resizing needed - outputs are naturally (480, 640)!
        return {
            "fluxa_keypoints": keypoints,
            "fluxa_segmentation": segmentation,
            "fluxa_surface_normals": surface_normals,
            "fluxa_environment_lighting": env_lighting,
        }

    def get_config(self):
        return {
            'input_shape': self.input_shape_dims,
            'base_filters': self.base_filters,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            'field_evolution_rate': self.field_evolution_rate,
        }
# =====================================
# FSE-Specific Metrics and Losses - MEMORY OPTIMIZED
# =====================================

class ContinuousFieldLoss(tf.keras.losses.Loss):
    """Custom loss for continuous field representations - MEMORY OPTIMIZED"""
    
    def __init__(self, field_type='keypoint', smoothness_weight=0.05, name='continuous_field_loss'):
        super().__init__(name=name)
        self.field_type = field_type
        self.smoothness_weight = smoothness_weight
    
    def call(self, y_true, y_pred):
        # Resize pred to match true - handle potential size mismatch
        y_true_shape = tf.shape(y_true)
        y_pred_shape = tf.shape(y_pred)
        
        # Only resize if shapes don't match
        if y_true_shape[1] != y_pred_shape[1] or y_true_shape[2] != y_pred_shape[2]:
            y_pred_resized = tf.image.resize(y_pred, size=[y_true_shape[1], y_true_shape[2]])
        else:
            y_pred_resized = y_pred
        
        # Base loss (MSE for continuous fields)
        base_loss = tf.reduce_mean(tf.square(y_true - y_pred_resized))
        
        # Light smoothness regularization - REDUCED for memory
        if self.smoothness_weight > 0:
            # Simple gradient penalty - memory efficient
            dy = y_pred_resized[:, 1:, :, :] - y_pred_resized[:, :-1, :, :]
            dx = y_pred_resized[:, :, 1:, :] - y_pred_resized[:, :, :-1, :]
            
            smoothness = (tf.reduce_mean(tf.square(dy)) + tf.reduce_mean(tf.square(dx))) * 0.5
            return base_loss + self.smoothness_weight * smoothness
        
        return base_loss

    def get_config(self):
        return {
            'field_type': self.field_type,
            'smoothness_weight': self.smoothness_weight
        }


class FieldCoherenceMetric(tf.keras.metrics.Metric):
    """Simplified field coherence metric - MEMORY OPTIMIZED"""
    
    def __init__(self, name='field_coherence', **kwargs):
        super().__init__(name=name, **kwargs)
        self.coherence_sum = self.add_weight(name='coherence_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Simple coherence measure - spatial variance
        spatial_var = tf.math.reduce_variance(y_pred, axis=[1, 2])
        coherence_score = 1.0 / (1.0 + tf.reduce_mean(spatial_var))
        
        self.coherence_sum.assign_add(coherence_score)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.coherence_sum / (self.count + tf.keras.backend.epsilon())
    
    def reset_state(self):
        self.coherence_sum.assign(0.0)
        self.count.assign(0.0)


# =====================================
# FSE Training Callbacks - SIMPLIFIED
# =====================================

class FSECheckpointCallback(tf.keras.callbacks.Callback):
    """FIXED checkpoint callback with guaranteed saving and loading"""
    
    def __init__(self, checkpoint_dir, gcs_dir, save_freq=5):
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.gcs_dir = gcs_dir
        self.save_freq = save_freq
        self.best_val_loss = float('inf')
        
        # Ensure directories exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"📁 Local checkpoint dir: {self.checkpoint_dir}")
        logger.info(f"☁️ GCS checkpoint dir: {self.gcs_dir}")
    
    def on_epoch_end(self, epoch, logs=None):
        """Save checkpoint every N epochs"""
        if (epoch + 1) % self.save_freq == 0:
            self._save_checkpoint(epoch, logs)
    
    def _save_checkpoint(self, epoch, logs=None):
        """Actually save the checkpoint with error handling"""
        try:
            # Create filenames
            filename = f"fse_checkpoint_epoch_{epoch+1:03d}.weights.h5"
            local_path = os.path.join(self.checkpoint_dir, filename)
            gcs_path = f"{self.gcs_dir}/{filename}"
            
            logger.info(f"💾 Saving FSE checkpoint for epoch {epoch+1}...")
            logger.info(f"   Local: {local_path}")
            logger.info(f"   GCS: {gcs_path}")
            
            # Save weights locally first
            self.model.save_weights(local_path)
            logger.info("✅ Local checkpoint saved")
            
            # Upload to GCS
            try:
                blob = bucket.blob(gcs_path)
                blob.upload_from_filename(local_path)
                logger.info("✅ GCS checkpoint uploaded")
            except Exception as e:
                logger.error(f"❌ GCS upload failed: {e}")
                # Continue anyway - we have local backup
            
            # Save metadata
            metadata = {
                'epoch': epoch + 1,
                'model_config': {
                    'base_filters': args.base_filters,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate
                },
                'metrics': logs if logs else {},
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = os.path.join(self.checkpoint_dir, f"metadata_epoch_{epoch+1:03d}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Upload metadata to GCS
            try:
                metadata_gcs = f"{self.gcs_dir}/metadata_epoch_{epoch+1:03d}.json"
                bucket.blob(metadata_gcs).upload_from_filename(metadata_path)
            except:
                pass  # Metadata upload failure is not critical
            
            # Always save as latest
            latest_local = os.path.join(self.checkpoint_dir, "fse_latest.weights.h5")
            latest_metadata = os.path.join(self.checkpoint_dir, "fse_latest_metadata.json")
            
            import shutil
            shutil.copy2(local_path, latest_local)
            shutil.copy2(metadata_path, latest_metadata)
            
            # Upload latest to GCS
            try:
                bucket.blob(f"{self.gcs_dir}/fse_latest.weights.h5").upload_from_filename(latest_local)
                bucket.blob(f"{self.gcs_dir}/fse_latest_metadata.json").upload_from_filename(latest_metadata)
            except:
                pass
            
            # Save best model
            val_loss = logs.get('val_loss', float('inf')) if logs else float('inf')
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_local = os.path.join(self.checkpoint_dir, "fse_best.weights.h5")
                shutil.copy2(local_path, best_local)
                
                try:
                    bucket.blob(f"{self.gcs_dir}/fse_best.weights.h5").upload_from_filename(best_local)
                except:
                    pass
                
                logger.info(f"🏅 Best model updated (val_loss: {val_loss:.4f})")
            
            logger.info(f"✅ FSE checkpoint saved successfully for epoch {epoch+1}")
            
        except Exception as e:
            logger.error(f"❌ CRITICAL: Failed to save FSE checkpoint: {e}")
            import traceback
            logger.error(traceback.format_exc())


class FSELearningRateScheduler(tf.keras.callbacks.Callback):
    """FSE learning rate scheduler - CONSERVATIVE"""
    
    def __init__(self, initial_lr=0.0002, decay_rate=0.98):  # Very conservative
        super().__init__()
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def on_epoch_begin(self, epoch, logs=None):
        # Conservative exponential decay
        lr = self.initial_lr * (self.decay_rate ** epoch)
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        
        if epoch % 5 == 0:  # Log every 5 epochs
            logger.info(f"📊 Learning rate: {lr:.6f}")

class FSEEpochTransitionCallback(tf.keras.callbacks.Callback):
    """Callback to monitor and log epoch transitions"""
    
    def __init__(self):
        super().__init__()
        self.epoch_start_time = None
    
    def on_train_begin(self, logs=None):
        logger.info("🚀 FSE Training Started")
        logger.info(f"📊 Total epochs planned: {self.params.get('epochs', 'unknown')}")
        logger.info(f"📦 Steps per epoch: {self.params.get('steps', 'unknown')}")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()
        logger.info("=" * 60)
        logger.info(f"🚀 EPOCH {epoch + 1}/{self.params.get('epochs', '?')} STARTING")
        logger.info(f"📊 Steps this epoch: {self.params.get('steps', 500)}")
        logger.info("=" * 60)
    
    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else 0
        
        logger.info("=" * 60)
        logger.info(f"✅ EPOCH {epoch + 1} COMPLETED in {epoch_time:.1f}s")
        
        if logs:
            logger.info("📈 Final epoch metrics:")
            for metric, value in logs.items():
                logger.info(f"   - {metric}: {value:.4f}")
        
        logger.info(f"🔄 Preparing for epoch {epoch + 2}...")
        logger.info("=" * 60)
    
    def on_train_end(self, logs=None):
        logger.info("🎉 FSE TRAINING COMPLETED SUCCESSFULLY!")
        if logs:
            logger.info("📊 Final training metrics:")
            for metric, value in logs.items():
                logger.info(f"   - {metric}: {value:.4f}")

def load_latest_checkpoint(model, checkpoint_dir, gcs_dir):
    """Load the latest checkpoint if available"""
    
    logger.info("🔍 Searching for FSE checkpoints...")
    
    latest_epoch = 0
    loaded_checkpoint = None
    
    # Check local checkpoints first
    if os.path.exists(checkpoint_dir):
        local_checkpoints = [f for f in os.listdir(checkpoint_dir) 
                           if f.startswith('fse_checkpoint_epoch_') and f.endswith('.weights.h5')]
        
        if local_checkpoints:
            # Sort by epoch number
            local_checkpoints.sort(key=lambda x: int(x.split('_')[3].split('.')[0]))
            latest_local = local_checkpoints[-1]
            local_path = os.path.join(checkpoint_dir, latest_local)
            
            try:
                model.load_weights(local_path)
                latest_epoch = int(latest_local.split('_')[3].split('.')[0])
                loaded_checkpoint = local_path
                logger.info(f"✅ Loaded local checkpoint: {latest_local}")
                logger.info(f"📊 Resuming from epoch {latest_epoch}")
            except Exception as e:
                logger.error(f"❌ Failed to load local checkpoint: {e}")
    
    # If no local checkpoint, try GCS
    if not loaded_checkpoint:
        try:
            gcs_checkpoints = []
            for blob in bucket.list_blobs(prefix=f"{gcs_dir}/"):
                if blob.name.endswith('.weights.h5') and 'fse_checkpoint_epoch_' in blob.name:
                    gcs_checkpoints.append(blob)
            
            if gcs_checkpoints:
                # Sort by name (which includes epoch number)
                gcs_checkpoints.sort(key=lambda x: x.name)
                latest_gcs = gcs_checkpoints[-1]
                
                # Download to local
                local_path = os.path.join(checkpoint_dir, os.path.basename(latest_gcs.name))
                latest_gcs.download_to_filename(local_path)
                
                # Load weights
                model.load_weights(local_path)
                latest_epoch = int(os.path.basename(latest_gcs.name).split('_')[3].split('.')[0])
                loaded_checkpoint = local_path
                
                logger.info(f"✅ Downloaded and loaded GCS checkpoint: {latest_gcs.name}")
                logger.info(f"📊 Resuming from epoch {latest_epoch}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load GCS checkpoint: {e}")
    
    if not loaded_checkpoint:
        logger.info("ℹ️ No checkpoints found - starting from scratch")
        return 0
    
    return latest_epoch

# =====================================
# Main Training Function - MEMORY OPTIMIZED & FIXED
# =====================================

def train_fluxa_fse(sample_ids, val_ids=None):
    """COMPLETELY FIXED training function with checkpoints and no loops"""
    
    logger.info("🚀 Initializing FLUXA-FSE training pipeline...")
    
    # Handle validation split
    if val_ids is None:
        train_size = int(0.9 * len(sample_ids))
        val_ids = sample_ids[train_size:]
        sample_ids = sample_ids[:train_size]
    
    if len(val_ids) > args.max_val_samples:
        val_ids = val_ids[:args.max_val_samples]
    
    logger.info(f"📊 Training samples: {len(sample_ids)}, Validation samples: {len(val_ids)}")
    
    # Create FIXED datasets
    train_dataset = create_fluxa_dataset(sample_ids, args.batch_size, training=True)
    val_dataset = create_fluxa_dataset(val_ids, args.val_batch_size, training=False)
    
    # Build model
    fluxa_fse = FLUXA_FSE(
        input_shape=(480, 640, 3),
        base_filters=args.base_filters,
        l2_reg=args.l2_reg,
        dropout_rate=args.dropout_rate,
        field_evolution_rate=args.field_evolution_rate
    )
    
    # Build model with dummy input
    logger.info("🔨 Building model variables...")
    dummy_input = tf.zeros((1, 480, 640, 3), dtype=tf.float32)
    try:
        _ = fluxa_fse(dummy_input, training=False)
        logger.info("✅ Model built successfully")
    except Exception as e:
        logger.error(f"❌ Model build failed: {e}")
        raise

    # Compile model
    losses = {
        'fluxa_keypoints': ContinuousFieldLoss(field_type='keypoint', smoothness_weight=0.05),
        'fluxa_segmentation': 'binary_crossentropy',
        'fluxa_surface_normals': ContinuousFieldLoss(field_type='normal', smoothness_weight=0.03),
        'fluxa_environment_lighting': 'mse'
    }
    
    loss_weights = {
        'fluxa_keypoints': args.keypoints_weight,
        'fluxa_segmentation': args.segmentation_weight,
        'fluxa_surface_normals': args.surface_normals_weight,
        'fluxa_environment_lighting': args.env_lighting_weight
    }
    
    metrics = {
        'fluxa_keypoints': ['mae'],
        'fluxa_segmentation': ['accuracy'],
        'fluxa_surface_normals': ['mae'],
        'fluxa_environment_lighting': ['mae']
    }
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        clipnorm=0.5
    )
    
    logger.info("🔧 Compiling FLUXA-FSE model...")
    fluxa_fse.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics=metrics
    )
    
    # Setup checkpoint directories
    checkpoint_dir = os.path.join(LOCAL_TEMP_DIR, "fse_checkpoints")
    gcs_checkpoint_dir = "fluxa/checkpoints"  # FIXED GCS path
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # LOAD EXISTING CHECKPOINT
    initial_epoch = load_latest_checkpoint(fluxa_fse, checkpoint_dir, gcs_checkpoint_dir)
    
    # Setup callbacks
    callbacks = [
        FSECheckpointCallback(
            checkpoint_dir=checkpoint_dir,
            gcs_dir=gcs_checkpoint_dir,
            save_freq=5
        ),
        FSELearningRateScheduler(
            initial_lr=args.learning_rate,
            decay_rate=0.98
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=args.early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
    ]
    
    # Calculate steps
    steps_per_epoch = min(len(sample_ids) // args.batch_size, 500)
    validation_steps = min(len(val_ids) // args.val_batch_size, 100)
    
    logger.info(f"📈 Training configuration:")
    logger.info(f"   - Initial epoch: {initial_epoch}")
    logger.info(f"   - Total epochs: {args.epochs}")
    logger.info(f"   - Steps per epoch: {steps_per_epoch}")
    logger.info(f"   - Validation steps: {validation_steps}")
    
    # Start training
    logger.info("🏃 Starting FLUXA-FSE training...")
    try:
        history = fluxa_fse.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            initial_epoch=initial_epoch,  # RESUME FROM CHECKPOINT
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
    
    return history
# =====================================
# Main Entry Point - ROBUST ERROR HANDLING
# =====================================

def main():
    """Main entry point for FLUXA-FSE training - COMPLETELY FIXED"""
    try:
        logger.info("=" * 60)
        logger.info("🚀 FLUXA-FSE Training Pipeline - FIXED VERSION")
        logger.info("🧬 Float-Native State Elements Architecture")
        logger.info("🔧 Memory Optimized & Lifecycle Compliant")
        logger.info("=" * 60)
        
        # Verify FSE core is available
        try:
            from fse_core import FLIT, CSE, FIL, FSEBlock
            logger.info("✅ FSE core components loaded successfully")
        except ImportError as e:
            logger.error(f"❌ Failed to import FSE core: {e}")
            logger.error("💡 Make sure fse_core.py is packaged with this script!")
            return 1
        
        # Memory diagnostics
        try:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"🎮 Found {len(gpus)} GPU(s)")
                for i, gpu in enumerate(gpus):
                    logger.info(f"   GPU {i}: {gpu.name}")
            else:
                logger.info("💻 Running on CPU")
        except Exception as e:
            logger.warning(f"⚠️ GPU check failed: {e}")
        
        # Find training samples
        logger.info("🔍 Discovering training data...")
        try:
            training_samples = find_common_samples(training=True)
            validation_samples = find_common_samples(training=False) if args.use_validation else None
        except Exception as e:
            logger.error(f"❌ Failed to discover training data: {e}")
            return 1
        
        if not training_samples:
            logger.error("❌ No training samples found!")
            logger.error("💡 Check GCS bucket paths and data availability")
            return 1
        
        # Limit samples for memory efficiency
        original_count = len(training_samples)
        if len(training_samples) > args.max_samples:
            logger.info(f"📊 Limiting training samples: {original_count} → {args.max_samples}")
            training_samples = training_samples[:args.max_samples]
        
        # Shuffle for randomness
        random.shuffle(training_samples)
        
        logger.info(f"📁 Using {len(training_samples)} training samples")
        if validation_samples:
            logger.info(f"📁 Using {len(validation_samples)} validation samples")
        
        # Memory usage estimate
        estimated_memory_gb = (args.batch_size * args.base_filters * 4) / 1024  # Rough estimate
        logger.info(f"📊 Estimated memory usage: ~{estimated_memory_gb:.1f}GB")
        
        if estimated_memory_gb > 12:  # Warning for high memory usage
            logger.warning("⚠️ High memory usage expected!")
            logger.warning("💡 Consider reducing --batch-size or --base-filters")
        
        # Start training with comprehensive error handling
        try:
            history = train_fluxa_fse(
                sample_ids=training_samples,
                val_ids=validation_samples
            )
            
            # Log final metrics if available
            if history and hasattr(history, 'history') and history.history:
                logger.info("📊 Final training metrics:")
                for metric, values in history.history.items():
                    if values:  # Check if list is not empty
                        logger.info(f"   - {metric}: {values[-1]:.4f}")
            
            logger.info("🎉 FLUXA-FSE training completed successfully!")
            return 0
        
        except tf.errors.ResourceExhaustedError as e:
            logger.error("❌ OUT OF MEMORY ERROR!")
            logger.error("💡 SOLUTIONS:")
            logger.error("   1. Reduce --batch-size (try 2 or 1)")
            logger.error("   2. Reduce --base-filters (try 8 or 12)")
            logger.error("   3. Reduce --max-samples")
            logger.error("   4. Use smaller image resolution")
            logger.error(f"📋 Current settings: batch_size={args.batch_size}, base_filters={args.base_filters}")
            return 2
        
        except tf.errors.InvalidArgumentError as e:
            logger.error("❌ TENSOR SHAPE/TYPE ERROR!")
            logger.error(f"📋 Error details: {e}")
            logger.error("💡 This usually indicates a bug in FSE layer implementation")
            return 3
        
        except Exception as e:
            logger.error(f"❌ UNEXPECTED TRAINING ERROR: {e}")
            import traceback
            logger.error("📋 Full traceback:")
            logger.error(traceback.format_exc())
            return 4
        
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"❌ FATAL ERROR in main(): {e}")
        import traceback
        logger.error("📋 Full traceback:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    """Entry point with proper exit code handling"""
    exit_code = main()
    
    # Log final status
    if exit_code == 0:
        logger.info("✅ FLUXA-FSE training pipeline completed successfully!")
    elif exit_code == 2:
        logger.error("💥 Training failed due to memory constraints")
    elif exit_code == 3:
        logger.error("💥 Training failed due to tensor/shape errors")
    elif exit_code == 130:
        logger.info("⏹️ Training was interrupted")
    else:
        logger.error("💥 Training failed with unexpected error")
    
    logger.info("=" * 60)
    sys.exit(exit_code)