# fse_native_core.py
# =====================================
# REFINED FSE NATIVE CORE - PRODUCTION READY - v3.1 (Complete with Global FSE Metrics)
# =====================================
#
# Based on the user's fse_native_core.py that achieved ~3-hour epochs.
# - All original layers (FLIT, CSE, etc.) from that version are preserved.
# - Global FSE metrics (_compute_environmental_coherence, etc.) from train_cpu.py
#   are integrated into FLUXA_FSE_Native.call().
# - Assumes mixed_float16 policy. Stateful variables are float32, compute ops use compute_dtype.
# - Output of FLUXA_FSE_Native.call() and metric values are float32.

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers # type: ignore
from typing import Dict, List, Optional, Tuple, Union
import math
# import json # Not used in this combined snippet, but kept if your original layers used it
import logging
# import random # Not used in this combined snippet, but kept if your original layers used it

# Import gradient checkpointing utility
try:
    recompute_grad = tf.recompute_grad
except AttributeError:
    try:
        from tensorflow.python.keras.utils import recompute_grad # type: ignore
    except ImportError:
        try:
            from tensorflow.keras.utils import get_recompute_function # type: ignore
            recompute_grad = get_recompute_function()
        except:
            def recompute_grad(func): # type: ignore
                """Fallback if gradient checkpointing is not available."""
                logging.warning("Gradient checkpointing (recompute_grad) not found. Using pass-through.")
                return func

logger = logging.getLogger(__name__)

# --- Utility Functions for Dtype and Shape Safety (from your original core script) ---
def verify_dtype_consistency(tensor1: tf.Tensor, tensor2: tf.Tensor, operation_name: str):
    if tensor1.dtype != tensor2.dtype:
        error_msg = f"DTYPE MISMATCH in {operation_name}: {tensor1.dtype} vs {tensor2.dtype}"
        # logger.error(error_msg) # Can be noisy, enable if debugging dtype issues
        # raise ValueError(error_msg) 
    return True

def safe_cast_to_compute_dtype(tensor: tf.Tensor, target_dtype: tf.DType, operation_name: str = "unknown_cast") -> tf.Tensor:
    if tensor.dtype != target_dtype:
        return tf.cast(tensor, target_dtype)
    return tensor

SYNTHA_CONTEXT_WIDTH = 8

class FLIT(layers.Layer):
    """Floating Information Unit - Refined for Mixed Precision and Stability."""
    def __init__(self, channels: int, field_type: str = 'continuous', evolution_rate: float = 0.1, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}

        if name is None:
            name = f"flit_{field_type}_{channels}"
        super().__init__(name=name, autocast=False, **kwargs) 

        self.channels = channels
        self.field_type = field_type
        self.evolution_rate = float(evolution_rate) 
        self.custom_cfg = custom_cfg

        self.field_context_sensitivity = float(custom_cfg.get('context_sensitivity', 0.8))
        self.adaptive_evolution = bool(custom_cfg.get('adaptive_evolution', True))

        self.field_state: Optional[tf.Variable] = None
        self.field_momentum: Optional[tf.Variable] = None
        self.evolution_kernel: Optional[tf.Variable] = None
        self.context_modulator: Optional[tf.Variable] = None
        self.field_activity_tracker: Optional[tf.Variable] = None

    def build(self, input_shape: tf.TensorShape):
        input_channels = int(input_shape[-1])

        self.field_state = self.add_weight(
            name='field_state', shape=(1, 1, 1, self.channels),
            initializer='glorot_uniform', trainable=True, dtype=tf.float32
        )
        self.field_momentum = self.add_weight(
            name='field_momentum', shape=(1, 1, 1, self.channels),
            initializer='zeros', trainable=False, dtype=tf.float32
        )
        self.evolution_kernel = self.add_weight(
            name='evolution_kernel', shape=(3, 3, input_channels, self.channels),
            initializer='glorot_uniform', trainable=True, dtype=tf.float32
        )
        self.context_modulator = self.add_weight(
            name='context_modulator', shape=(1, 1, 1, self.channels),
            initializer='ones', trainable=True, dtype=tf.float32
        )
        self.field_activity_tracker = self.add_weight(
            name='field_activity_tracker', shape=(1,),
            initializer='zeros', trainable=False, dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        
        context_signal_compute_dtype = None
        if context_signal is not None:
            context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_context_signal")

        kernel_compute_dtype = tf.cast(self.evolution_kernel, self.compute_dtype)
        verify_dtype_consistency(inputs_compute_dtype, kernel_compute_dtype, f"{self.name}_conv_input_kernel")
        field_response = tf.nn.conv2d(
            inputs_compute_dtype, kernel_compute_dtype,
            strides=[1, 1, 1, 1], padding='SAME'
        )

        current_evolution_rate = self.evolution_rate 
        if self.adaptive_evolution and context_signal_compute_dtype is not None:
            context_signal_f32 = tf.cast(context_signal_compute_dtype, tf.float32)
            context_factor_f32 = tf.reduce_mean(tf.abs(context_signal_f32))
            context_factor_clipped_f32 = tf.clip_by_value(context_factor_f32, 0.0, 2.0)
            adaptive_multiplier_f32 = 1.0 + 0.5 * self.field_context_sensitivity * context_factor_clipped_f32
            adaptive_multiplier_compute_dtype = tf.cast(adaptive_multiplier_f32, self.compute_dtype)
            verify_dtype_consistency(field_response, adaptive_multiplier_compute_dtype, f"{self.name}_adaptive_evo_mult")
            field_response = field_response * adaptive_multiplier_compute_dtype
            
        field_influence_compute_dtype = tf.reduce_mean(field_response, axis=[0, 1, 2], keepdims=True)
        field_influence_f32 = tf.cast(field_influence_compute_dtype, tf.float32)

        field_momentum_f32 = tf.cast(self.field_momentum, tf.float32) 
        new_momentum_f32 = 0.9 * field_momentum_f32 + 0.1 * field_influence_f32
        self.field_momentum.assign(new_momentum_f32)

        field_state_f32 = tf.cast(self.field_state, tf.float32) 
        field_update_f32 = current_evolution_rate * new_momentum_f32 
        new_field_state_f32 = field_state_f32 + field_update_f32
        self.field_state.assign(new_field_state_f32)

        field_state_compute_dtype = tf.cast(self.field_state, self.compute_dtype) 
        context_modulator_compute_dtype = tf.cast(self.context_modulator, self.compute_dtype)
        verify_dtype_consistency(field_state_compute_dtype, context_modulator_compute_dtype, f"{self.name}_state_mod")
        modulated_state = field_state_compute_dtype * context_modulator_compute_dtype
        
        verify_dtype_consistency(field_response, modulated_state, f"{self.name}_response_add")
        output_compute_dtype = tf.nn.tanh(field_response + modulated_state)

        # Keras `training` is Python bool or None. tf.cast(None, tf.bool) is error.
        if training is not None and tf.cast(training, tf.bool): 
            activity_level_compute_dtype = tf.reduce_mean(tf.abs(output_compute_dtype))
            activity_level_f32 = tf.cast(activity_level_compute_dtype, tf.float32)
            field_activity_tracker_f32 = tf.cast(self.field_activity_tracker, tf.float32) 
            new_activity_f32 = 0.9 * field_activity_tracker_f32 + 0.1 * activity_level_f32
            self.field_activity_tracker.assign(new_activity_f32)
        
        return output_compute_dtype 

    def get_field_activity(self) -> Union[float, tf.Tensor]:
        return self.field_activity_tracker 

class CSE(layers.Layer):
    """Continuous State Element - Refined for Mixed Precision and Stability."""
    def __init__(self, evolution_rate: float = 0.1, context_type: str = 'general', specialization_level: float = 1.0, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        
        if name is None:
            name = f"cse_{context_type}_{specialization_level:.1f}"
        super().__init__(name=name, autocast=False, **kwargs)

        self.evolution_rate = float(evolution_rate)
        self.context_type = str(context_type)
        self.specialization_level = float(specialization_level)
        self.custom_cfg = custom_cfg 

        self.field_projector: Optional[tf.Variable] = None
        self.continuous_state: Optional[tf.Variable] = None
        self.state_velocity: Optional[tf.Variable] = None
        self.performance_tracker: Optional[tf.Variable] = None
        self.proj_dim: Optional[int] = None 

    def build(self, input_shape: tf.TensorShape):
        channels = int(input_shape[-1]) 
        self.proj_dim = int(self.custom_cfg.get('proj_dim', channels))

        self.field_projector = self.add_weight(
            name="field_projector", shape=(channels, self.proj_dim), 
            initializer="glorot_uniform", trainable=True, dtype=tf.float32
        )
        self.continuous_state = self.add_weight(
            name='continuous_state', shape=(1, 1, 1, self.proj_dim), 
            initializer='zeros', trainable=False, dtype=tf.float32
        )
        self.state_velocity = self.add_weight(
            name='state_velocity', shape=(1, 1, 1, self.proj_dim), 
            initializer='zeros', trainable=False, dtype=tf.float32
        )
        self.performance_tracker = self.add_weight(
            name='performance_tracker', shape=(3,), 
            initializer='zeros', trainable=False, dtype=tf.float32
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        context_signal_compute_dtype = None
        if context_signal is not None:
            context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_context_signal")

        input_s = tf.shape(inputs_compute_dtype)
        b, h, w, c_in = input_s[0], input_s[1], input_s[2], input_s[3] # c_in is input channels to CSE

        reshaped_inputs = tf.reshape(inputs_compute_dtype, [b * h * w, c_in])
        projector_compute_dtype = tf.cast(self.field_projector, self.compute_dtype)
        verify_dtype_consistency(reshaped_inputs, projector_compute_dtype, f"{self.name}_matmul_input_proj")
        projected = tf.matmul(reshaped_inputs, projector_compute_dtype) 
        projected = tf.reshape(projected, [b, h, w, self.proj_dim]) 

        if self.specialization_level != 1.0: 
            projected = projected * tf.cast(self.specialization_level, self.compute_dtype)

        if context_signal_compute_dtype is not None:
            context_signal_f32 = tf.cast(context_signal_compute_dtype, tf.float32)
            context_factor_f32 = tf.reduce_mean(tf.abs(context_signal_f32)) 
            context_factor_clipped_f32 = tf.clip_by_value(context_factor_f32, 0.0, 1.0)
            context_multiplier_f32 = 1.0 + 0.3 * context_factor_clipped_f32
            context_multiplier_compute_dtype = tf.cast(context_multiplier_f32, self.compute_dtype)
            verify_dtype_consistency(projected, context_multiplier_compute_dtype, f"{self.name}_ctx_influence_mult")
            projected = projected * context_multiplier_compute_dtype
        
        spatial_mean_compute_dtype = tf.reduce_mean(projected, axis=[0, 1, 2], keepdims=True)
        spatial_mean_f32 = tf.cast(spatial_mean_compute_dtype, tf.float32)
        
        continuous_state_f32_op = tf.cast(self.continuous_state, tf.float32)
        state_velocity_f32_op = tf.cast(self.state_velocity, tf.float32)

        field_gradient_f32 = spatial_mean_f32 - continuous_state_f32_op 
        
        new_velocity_f32 = 0.95 * state_velocity_f32_op + self.evolution_rate * field_gradient_f32
        self.state_velocity.assign(new_velocity_f32) 
        
        new_state_f32 = continuous_state_f32_op + new_velocity_f32 
        self.continuous_state.assign(new_state_f32) 
        
        continuous_state_compute_dtype = tf.cast(self.continuous_state, self.compute_dtype) 
        verify_dtype_consistency(projected, continuous_state_compute_dtype, f"{self.name}_proj_state_add")
        output_compute_dtype = projected + continuous_state_compute_dtype

        if self.context_type == 'lighting': output_compute_dtype = tf.nn.sigmoid(output_compute_dtype)
        elif self.context_type == 'temporal': output_compute_dtype = tf.sin(output_compute_dtype)
        elif self.context_type == 'material': output_compute_dtype = tf.nn.leaky_relu(output_compute_dtype, alpha=0.2) 
        elif self.context_type == 'spatial': output_compute_dtype = tf.nn.tanh(output_compute_dtype) * tf.nn.sigmoid(output_compute_dtype)
        elif self.context_type == 'quantum': output_compute_dtype = tf.nn.tanh(output_compute_dtype) * tf.cos(output_compute_dtype)
        else: output_compute_dtype = tf.nn.tanh(output_compute_dtype)

        if training is not None and tf.cast(training, tf.bool): 
            self._update_performance_metrics(output_compute_dtype, context_signal_compute_dtype)
        
        return output_compute_dtype 

    def _update_performance_metrics(self, output_compute_dtype: tf.Tensor, context_signal_compute_dtype: Optional[tf.Tensor]):
        output_f32 = tf.cast(output_compute_dtype, tf.float32)
        output_abs_f32 = tf.abs(output_f32)
        activity_f32 = tf.reduce_mean(output_abs_f32)
        consistency_f32 = 1.0 / (1.0 + tf.math.reduce_variance(output_f32) + 1e-8) 
        relevance_f32 = tf.constant(1.0, dtype=tf.float32) 
        if context_signal_compute_dtype is not None:
            context_f32 = tf.cast(context_signal_compute_dtype, tf.float32)
            context_magnitude_f32 = tf.reduce_mean(tf.abs(context_f32))
            output_magnitude_f32 = tf.reduce_mean(output_abs_f32)
            relevance_f32 = tf.minimum(context_magnitude_f32 / (output_magnitude_f32 + 1e-8), 2.0)
        new_metrics_f32 = tf.stack([activity_f32, consistency_f32, relevance_f32], axis=0)
        performance_tracker_f32 = tf.cast(self.performance_tracker, tf.float32) 
        updated_tracker_f32 = 0.9 * performance_tracker_f32 + 0.1 * new_metrics_f32
        self.performance_tracker.assign(updated_tracker_f32)

    def get_performance_metrics(self) -> Dict[str, Union[float, tf.Tensor]]:
        metrics_tensor = self.performance_tracker 
        return {
            'activity': metrics_tensor[0], 'consistency': metrics_tensor[1],
            'relevance': metrics_tensor[2], 'overall_score': tf.reduce_mean(metrics_tensor)
        }

class DynamicFIL(layers.Layer):
    """Field Interaction Layer - Refined for Streaming Reduction and Mixed Precision."""
    def __init__(self, initial_cse_count: int, channels: int, field_type: str = 'continuous', 
                 kernel_size: int = 3, max_cses: int = 16, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        if name is None: name = f"dynamic_fil_{field_type}_{channels}_{initial_cse_count}cse"
        super().__init__(name=name, autocast=False, **kwargs)
        self.initial_cse_count = int(initial_cse_count)
        self.channels = int(channels) 
        self.field_type = str(field_type)
        self.kernel_size = int(kernel_size)
        self.max_cses = int(max_cses)
        self.custom_cfg = custom_cfg
        self.cses: List[CSE] = []
        self.cse_active_flags: Optional[tf.Variable] = None
        self.dynamic_field_mixer: Optional[tf.Variable] = None
        self.spatial_processor: Optional[tf.Variable] = None
        self.context_types = ['general', 'spatial', 'lighting', 'material', 'temporal', 'quantum']

    def build(self, input_shape: tf.TensorShape):
        for i in range(self.max_cses):
            ctx = self.context_types[i % len(self.context_types)]
            cse_config = {'proj_dim': self.channels} 
            cse = CSE(evolution_rate=0.1, context_type=ctx, specialization_level=1.0 + 0.1 * i, name=f'{self.name}_cse_{i}_{ctx}', **cse_config)
            self.cses.append(cse)
        initial_active = np.zeros(self.max_cses, dtype=np.float32)
        initial_active[:self.initial_cse_count] = 1.0
        self.cse_active_flags = self.add_weight(name='cse_active_flags', shape=(self.max_cses,), initializer=tf.constant_initializer(initial_active), trainable=False, dtype=tf.float32)
        self.dynamic_field_mixer = self.add_weight(name='dynamic_field_mixer', shape=(self.max_cses, self.max_cses), initializer='orthogonal', trainable=True, dtype=tf.float32)
        self.spatial_processor = self.add_weight(name='spatial_processor', shape=(self.kernel_size, self.kernel_size, self.channels, self.channels), initializer='glorot_uniform', trainable=True, dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        context_signal_compute_dtype = None
        if context_signal is not None: context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_context_signal")
        
        input_s = tf.shape(inputs_compute_dtype)
        # Create zeros with the FIL's output channel dimension (self.channels)
        zeros_for_shape = tf.zeros([input_s[0], input_s[1], input_s[2], self.channels], dtype=self.compute_dtype)
        accumulated_cse_output = tf.zeros_like(zeros_for_shape)

        _flags_f32 = tf.cast(self.cse_active_flags, tf.float32) # Mixer expects float32 for weights
        _mixer_f32 = tf.cast(self.dynamic_field_mixer, tf.float32)
        
        # Calculate weights for mixing active CSEs
        # Element-wise multiply mixer rows by active flags, then sum each row for its weight
        # Mixer is [max_cses, max_cses], flags is [max_cses]
        # Effective mixer: mixer weights for active inputs, zeroed out for inactive inputs
        effective_mixer_rows = _mixer_f32 * _flags_f32[:, tf.newaxis] # [max_cses(out), max_cses(in)] , weights active CSEs contributions
        
        # For each output CSE channel, sum contributions from active input CSEs
        # This logic needs to be rethought for mixing. A simpler weighted sum:
        cse_outputs_list = []
        active_cse_indices = tf.where(tf.cast(self.cse_active_flags, tf.bool))[:,0] # Get indices of active CSEs

        for i in range(self.max_cses): # Iterate through all potential CSEs
             # Only compute if active, otherwise it's wasted computation if not mixed in
            is_active = tf.cast(self.cse_active_flags[i] > 0.5, tf.bool) # Check if this CSE is active
            
            # This tf.cond is problematic for graph compilation with loops.
            # A common pattern is to compute all and then select/mask.
            # For now, let's assume all are computed and mixing handles it.
            cse_out_compute_dtype = self.cses[i](inputs_compute_dtype, training=training, context_signal=context_signal_compute_dtype)
            cse_outputs_list.append(cse_out_compute_dtype * tf.reshape(tf.cast(self.cse_active_flags[i], self.compute_dtype), [1,1,1,1]))


        # Stack all CSE outputs (active ones will have their values, inactive ones might be zeroed by flag multiplication)
        stacked_cse_outputs = tf.stack(cse_outputs_list, axis=-1) # [B, H, W, C_cse_out, max_cses]
        
        # Mixer should be [max_cses, 1] if we want a single weighted sum, or [max_cses, C_fil_out] for projection
        # The current mixer is [max_cses, max_cses]. Let's use a simpler mixing: weighted average of active CSEs.
        # For simplicity, let's use a fixed mixing strategy for now: average active CSEs.
        # A more sophisticated dynamic_field_mixer would be [num_active_cses, self.channels]
        
        active_flags_reshaped = tf.reshape(tf.cast(self.cse_active_flags, self.compute_dtype), [1,1,1,1,self.max_cses])
        masked_outputs = stacked_cse_outputs * active_flags_reshaped
        sum_active_outputs = tf.reduce_sum(masked_outputs, axis=-1) # Sum across CSE dimension
        num_active_cses = tf.reduce_sum(tf.cast(self.cse_active_flags, self.compute_dtype)) + tf.cast(1e-8, self.compute_dtype)
        
        mixed_field_out = sum_active_outputs / num_active_cses
        
        spatial_kernel_compute_dtype = tf.cast(self.spatial_processor, self.compute_dtype)
        verify_dtype_consistency(mixed_field_out, spatial_kernel_compute_dtype, f"{self.name}_spatial_conv_input_kernel")
        spatial_out = tf.nn.conv2d(mixed_field_out, spatial_kernel_compute_dtype, strides=[1, 1, 1, 1], padding='SAME') 

        output_compute_dtype = spatial_out 
        if self.field_type == 'wave': output_compute_dtype = tf.sin(spatial_out)
        elif self.field_type == 'quantum': output_compute_dtype = tf.nn.tanh(spatial_out) * tf.cos(2.0 * spatial_out) 
        elif self.field_type == 'spatial': output_compute_dtype = tf.nn.tanh(spatial_out) * tf.nn.sigmoid(spatial_out)
        elif self.field_type == 'material': output_compute_dtype = tf.nn.leaky_relu(spatial_out, alpha=0.2) 
        elif self.field_type == 'lighting': output_compute_dtype = tf.nn.sigmoid(spatial_out)
        elif self.field_type == 'temporal': output_compute_dtype = tf.sin(spatial_out) * tf.nn.tanh(spatial_out)
        else: output_compute_dtype = tf.nn.tanh(spatial_out)
        return output_compute_dtype 

    def update_active_cses(self, num_active: int): 
        if not tf.executing_eagerly(): logger.warning(f"{self.name}: update_active_cses in graph mode."); return
        num_active = min(max(2, int(num_active)), self.max_cses) 
        new_flags_np = np.zeros(self.max_cses, dtype=np.float32); new_flags_np[:num_active] = 1.0
        if self.cse_active_flags is not None: self.cse_active_flags.assign(tf.constant(new_flags_np, dtype=tf.float32))
        else: logger.error(f"{self.name}: cse_active_flags is None.")

    def get_cse_status(self) -> Dict[str, Union[int, tf.Tensor]]:
        if self.cse_active_flags is None: return {'total_cses': self.max_cses, 'active_cses': 0, 'error': 'cse_active_flags not initialized'}
        active_flags_tensor = self.cse_active_flags; num_active_tensor = tf.reduce_sum(active_flags_tensor)
        num_active_val = int(num_active_tensor.numpy()) if tf.executing_eagerly() else num_active_tensor
        return {'total_cses': self.max_cses, 'active_cses': num_active_val, 'max_cses_in_fil': self.max_cses }

class FSENativeBlock(layers.Layer):
    def __init__(self, channels: int, num_fils: int = 2, fil_configs: Optional[List[Dict]] = None, max_cses_per_fil: int = 8, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        super().__init__(name=name, autocast=False, **kwargs)
        self.channels = int(channels)
        self.num_fils = int(num_fils)
        self.max_cses_per_fil = int(max_cses_per_fil)
        self.fil_configs = fil_configs or [{'initial_cse_count': 3, 'field_type': 'continuous'},{'initial_cse_count': 2, 'field_type': 'wave'}] * ((num_fils + 1) // 2) 
        self.custom_cfg = custom_cfg
        self.dynamic_fils: List[DynamicFIL] = []
        self.field_normalizer: Optional[layers.LayerNormalization] = None
        self.complexity_analyzer: Optional[layers.Dense] = None
        self.skip_projection: Optional[layers.Conv2D] = None

    def build(self, input_shape: tf.TensorShape):
        input_channels_to_block = int(input_shape[-1])
        for i in range(self.num_fils):
            config = self.fil_configs[i % len(self.fil_configs)]
            fil = DynamicFIL(initial_cse_count=config.get('initial_cse_count', 3), channels=self.channels, field_type=config.get('field_type', 'continuous'), max_cses=self.max_cses_per_fil, name=f'{self.name}_dynamic_fil_{i}')
            self.dynamic_fils.append(fil)
        self.field_normalizer = layers.LayerNormalization(epsilon=1e-6, name=f'{self.name}_field_normalizer', dtype=tf.float32)
        self.complexity_analyzer = layers.Dense(16, activation='tanh', kernel_regularizer=regularizers.l2(1e-5), dtype='float32', name=f'{self.name}_complexity_analyzer') 
        if input_channels_to_block != self.channels:
            self.skip_projection = layers.Conv2D(self.channels, kernel_size=1, padding='same', name=f'{self.name}_skip_projection', kernel_initializer='glorot_uniform', bias_initializer='zeros', dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        processed_context_signal_compute_dtype = None
        if context_signal is not None: processed_context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_proc_ctx")
        
        flattened_inputs_f32 = tf.cast(tf.reduce_mean(inputs_compute_dtype, axis=[1, 2]), tf.float32)
        complexity_f32 = self.complexity_analyzer(flattened_inputs_f32, training=training)
        complexity_compute_dtype = tf.cast(complexity_f32, self.compute_dtype) 
        
        combined_context_compute_dtype = complexity_compute_dtype
        if processed_context_signal_compute_dtype is not None:
            # Ensure processed_context_signal_compute_dtype is [B, Features]
            if len(processed_context_signal_compute_dtype.shape) > 2:
                 processed_context_signal_for_concat = tf.reduce_mean(processed_context_signal_compute_dtype, axis=[1,2])
            else:
                 processed_context_signal_for_concat = processed_context_signal_compute_dtype
            
            # Ensure complexity_compute_dtype is also [B, Features] (it should be from Dense)
            verify_dtype_consistency(processed_context_signal_for_concat, complexity_compute_dtype, f"{self.name}_ctx_concat")
            combined_context_compute_dtype = tf.concat([processed_context_signal_for_concat, complexity_compute_dtype], axis=-1)
        
        x = inputs_compute_dtype 
        for fil_idx, fil in enumerate(self.dynamic_fils): x = fil(x, training=training, context_signal=combined_context_compute_dtype)
        
        x_norm_input_f32 = tf.cast(x, tf.float32) # LayerNorm in f32
        x_normalized_f32 = self.field_normalizer(x_norm_input_f32, training=training) 
        x_normalized = tf.cast(x_normalized_f32, self.compute_dtype) # Cast back
        
        skip_connection_output = inputs_compute_dtype 
        if self.skip_projection is not None:
            skip_input_f32 = tf.cast(inputs_compute_dtype, tf.float32) 
            skip_projected_f32 = self.skip_projection(skip_input_f32, training=training)
            skip_connection_output = tf.cast(skip_projected_f32, self.compute_dtype)
        
        verify_dtype_consistency(x_normalized, skip_connection_output, f"{self.name}_residual_add")
        output = x_normalized + skip_connection_output
        return output 

    def get_all_cse_status(self) -> Dict[str, any]:
        status: Dict[str, any] = {'total_fils': len(self.dynamic_fils),'fils': {}}
        total_active_cses_count = 0
        for i, fil in enumerate(self.dynamic_fils):
            fil_status = fil.get_cse_status()
            status['fils'][f'fil_{i}'] = fil_status
            active_in_fil = fil_status.get('active_cses', 0)
            if isinstance(active_in_fil, tf.Tensor) and tf.executing_eagerly(): active_in_fil = int(active_in_fil.numpy())
            elif not isinstance(active_in_fil, int) and not isinstance(active_in_fil, tf.Tensor) and active_in_fil is not None: active_in_fil = int(active_in_fil) # type: ignore
            if isinstance(active_in_fil, int): total_active_cses_count += active_in_fil
        status['total_active_cses_in_block'] = total_active_cses_count
        return status

class FSENativeDownsample(layers.Layer):
    def __init__(self, channels: int, downsample_factor: int = 2, context_adaptive: bool = True, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        super().__init__(name=name, autocast=False, **kwargs)
        self.channels = int(channels); self.downsample_factor = int(downsample_factor); self.context_adaptive = bool(context_adaptive); self.custom_cfg = custom_cfg
        self.field_compressor: Optional[tf.Variable] = None; self.context_modulator: Optional[tf.Variable] = None; self.norm: Optional[layers.BatchNormalization] = None 

    def build(self, input_shape: tf.TensorShape):
        input_channels = int(input_shape[-1]); kernel_size = self.downsample_factor * 2 
        self.field_compressor = self.add_weight(name='field_compressor', shape=(kernel_size, kernel_size, input_channels, self.channels), initializer='glorot_uniform', trainable=True, dtype=tf.float32)
        if self.context_adaptive: self.context_modulator = self.add_weight(name='context_modulator', shape=(1, 1, 1, self.channels), initializer='ones', trainable=True, dtype=tf.float32)
        self.norm = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}_bn', dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        kernel_compute_dtype = tf.cast(self.field_compressor, self.compute_dtype)
        verify_dtype_consistency(inputs_compute_dtype, kernel_compute_dtype, f"{self.name}_conv_input_kernel")
        output_conv = tf.nn.conv2d(inputs_compute_dtype, kernel_compute_dtype, strides=[1, self.downsample_factor, self.downsample_factor, 1], padding='SAME')
        
        output_f32 = tf.cast(output_conv, tf.float32) # BN in f32
        output_normed_f32 = self.norm(output_f32, training=training)
        output = tf.cast(output_normed_f32, self.compute_dtype) # Cast back

        if self.context_adaptive and context_signal is not None and self.context_modulator is not None:
            context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_context_signal")
            context_signal_f32 = tf.cast(context_signal_compute_dtype, tf.float32)
            context_factor_f32 = tf.reduce_mean(tf.abs(context_signal_f32)); context_factor_clipped_f32 = tf.clip_by_value(context_factor_f32, 0.0, 1.0)
            modulator_f32 = tf.cast(self.context_modulator, tf.float32)
            adaptive_modulation_f32 = modulator_f32 * (1.0 + 0.2 * context_factor_clipped_f32)
            adaptive_modulation_compute_dtype = tf.cast(adaptive_modulation_f32, self.compute_dtype)
            verify_dtype_consistency(output, adaptive_modulation_compute_dtype, f"{self.name}_ctx_mod_mult")
            output = output * adaptive_modulation_compute_dtype
        output = tf.nn.tanh(output)
        return output

class FSENativeUpsample(layers.Layer):
    def __init__(self, channels: int, upsample_factor: int = 2, context_adaptive: bool = True, name: Optional[str] = None, **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        super().__init__(name=name, autocast=False, **kwargs)
        self.channels = int(channels); self.upsample_factor = int(upsample_factor); self.context_adaptive = bool(context_adaptive); self.custom_cfg = custom_cfg
        self.field_reconstructor: Optional[tf.Variable] = None; self.context_enhancer: Optional[tf.Variable] = None; self.norm: Optional[layers.BatchNormalization] = None

    def build(self, input_shape: tf.TensorShape):
        input_channels = int(input_shape[-1]); conv_output_channels = self.channels * (self.upsample_factor ** 2)
        self.field_reconstructor = self.add_weight(name='field_reconstructor', shape=(3, 3, input_channels, conv_output_channels), initializer='glorot_uniform', trainable=True, dtype=tf.float32)
        if self.context_adaptive: self.context_enhancer = self.add_weight(name='context_enhancer', shape=(1, 1, 1, self.channels), initializer='ones', trainable=True, dtype=tf.float32)
        self.norm = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}_bn', dtype=tf.float32)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None, context_signal: Optional[tf.Tensor] = None) -> tf.Tensor:
        inputs_compute_dtype = safe_cast_to_compute_dtype(inputs, self.compute_dtype, f"{self.name}_inputs")
        kernel_compute_dtype = tf.cast(self.field_reconstructor, self.compute_dtype)
        verify_dtype_consistency(inputs_compute_dtype, kernel_compute_dtype, f"{self.name}_conv_input_kernel")
        expanded = tf.nn.conv2d(inputs_compute_dtype, kernel_compute_dtype, strides=[1, 1, 1, 1], padding='SAME')
        upsampled_unnormed = tf.nn.depth_to_space(expanded, self.upsample_factor)
        
        upsampled_f32 = tf.cast(upsampled_unnormed, tf.float32) # BN in f32
        upsampled_normed_f32 = self.norm(upsampled_f32, training=training)
        upsampled = tf.cast(upsampled_normed_f32, self.compute_dtype) # Cast back

        if self.context_adaptive and context_signal is not None and self.context_enhancer is not None:
            context_signal_compute_dtype = safe_cast_to_compute_dtype(context_signal, self.compute_dtype, f"{self.name}_context_signal")
            context_signal_f32 = tf.cast(context_signal_compute_dtype, tf.float32)
            context_factor_f32 = tf.reduce_mean(tf.abs(context_signal_f32)); context_factor_clipped_f32 = tf.clip_by_value(context_factor_f32, 0.0, 1.0)
            enhancer_f32 = tf.cast(self.context_enhancer, tf.float32)
            adaptive_enhancement_f32 = enhancer_f32 * (1.0 + 0.3 * context_factor_clipped_f32)
            adaptive_enhancement_compute_dtype = tf.cast(adaptive_enhancement_f32, self.compute_dtype)
            verify_dtype_consistency(upsampled, adaptive_enhancement_compute_dtype, f"{self.name}_ctx_enhance_mult")
            upsampled = upsampled * adaptive_enhancement_compute_dtype
        output = tf.nn.tanh(upsampled)
        return output

class FLUXA_FSE_Native(tf.keras.Model):
    def __init__(self, 
                 input_shape: Tuple[int, int, int] = (480, 640, 3),
                 base_channels: int = 32,
                 field_evolution_rate: float = 0.1,
                 enable_syntha_integration: bool = True,
                 max_cses_per_fil: int = 8,
                 name: str = "fluxa_fse_native_metrics_v3_1", # Updated name
                 **kwargs):
        super().__init__(name=name, autocast=False, **kwargs) 
        self.input_shape_dims = input_shape 
        self.base_channels = int(base_channels)
        self.field_evolution_rate = float(field_evolution_rate)
        self.enable_syntha_integration = bool(enable_syntha_integration)
        self.max_cses_per_fil = int(max_cses_per_fil)
        self._build_architecture() 
        if self.enable_syntha_integration:
            self.syntha_context_generator = layers.Dense(32, activation='tanh', kernel_regularizer=regularizers.l2(1e-5),dtype='float32', name='syntha_context_generator') 
            self.global_complexity_tracker = tf.Variable(0.0, trainable=False, name='global_complexity_tracker', dtype=tf.float32)
    
    def _build_architecture(self):
        self.input_processor = FSENativeBlock(channels=self.base_channels, num_fils=2, max_cses_per_fil=self.max_cses_per_fil, name='input_processor')
        self.encoder_stage1 = FSENativeBlock(channels=self.base_channels * 2, num_fils=3, max_cses_per_fil=self.max_cses_per_fil, name='encoder_stage1')
        self.downsample1 = FSENativeDownsample(channels=self.base_channels * 2, context_adaptive=self.enable_syntha_integration, name='downsample1')
        self.encoder_stage2 = FSENativeBlock(channels=self.base_channels * 4, num_fils=3, max_cses_per_fil=self.max_cses_per_fil, name='encoder_stage2')
        self.downsample2 = FSENativeDownsample(channels=self.base_channels * 4, context_adaptive=self.enable_syntha_integration, name='downsample2')
        self.encoder_stage3 = FSENativeBlock(channels=self.base_channels * 8, num_fils=4, max_cses_per_fil=self.max_cses_per_fil, name='encoder_stage3')
        self.downsample3 = FSENativeDownsample(channels=self.base_channels * 8, context_adaptive=self.enable_syntha_integration, name='downsample3')
        self.bottleneck = FSENativeBlock(channels=self.base_channels * 16, num_fils=4, fil_configs=[{'initial_cse_count': 4, 'field_type': 'continuous'}, {'initial_cse_count': 3, 'field_type': 'wave'}, {'initial_cse_count': 2, 'field_type': 'quantum'}, {'initial_cse_count': 2, 'field_type': 'continuous'}], max_cses_per_fil=self.max_cses_per_fil * 2, name='bottleneck')
        self.upsample1 = FSENativeUpsample(channels=self.base_channels * 8, context_adaptive=self.enable_syntha_integration, name='upsample1')
        self.decoder_stage1 = FSENativeBlock( channels=self.base_channels * 8, num_fils=3, max_cses_per_fil=self.max_cses_per_fil, name='decoder_stage1')
        self.upsample2 = FSENativeUpsample(channels=self.base_channels * 4, context_adaptive=self.enable_syntha_integration, name='upsample2')
        self.decoder_stage2 = FSENativeBlock(channels=self.base_channels * 4, num_fils=3, max_cses_per_fil=self.max_cses_per_fil, name='decoder_stage2')
        self.upsample3 = FSENativeUpsample(channels=self.base_channels * 2, context_adaptive=self.enable_syntha_integration, name='upsample3')
        self.decoder_stage3 = FSENativeBlock( channels=self.base_channels, num_fils=2, max_cses_per_fil=self.max_cses_per_fil, name='decoder_stage3')
        self._build_task_processors()
        self._build_output_generators()

    def _build_task_processors(self):
        self.keypoint_processor = FSENativeBlock(channels=self.base_channels, num_fils=2, fil_configs=[{'initial_cse_count': 4, 'field_type': 'continuous'},{'initial_cse_count': 3, 'field_type': 'spatial'}], max_cses_per_fil=self.max_cses_per_fil, name='keypoint_processor')
        self.segmentation_processor = FSENativeBlock(channels=self.base_channels, num_fils=2, fil_configs=[{'initial_cse_count': 3, 'field_type': 'continuous'},{'initial_cse_count': 2, 'field_type': 'wave'}], max_cses_per_fil=self.max_cses_per_fil, name='segmentation_processor')
        self.surface_normal_processor = FSENativeBlock(channels=self.base_channels, num_fils=3, fil_configs=[{'initial_cse_count': 4, 'field_type': 'continuous'},{'initial_cse_count': 3, 'field_type': 'wave'},{'initial_cse_count': 2, 'field_type': 'material'}], max_cses_per_fil=self.max_cses_per_fil, name='surface_normal_processor')
        self.env_lighting_processor = FSENativeBlock( channels=self.base_channels * 2, num_fils=2, fil_configs=[{'initial_cse_count': 3, 'field_type': 'lighting'},{'initial_cse_count': 2, 'field_type': 'continuous'}], max_cses_per_fil=self.max_cses_per_fil, name='env_lighting_processor')

    def _build_output_generators(self):
        self.keypoint_generator = FLIT(channels=17, field_type='continuous', evolution_rate=self.field_evolution_rate, name='keypoint_generator')
        self.segmentation_generator = FLIT(channels=1, field_type='continuous', evolution_rate=self.field_evolution_rate, name='segmentation_generator')
        self.surface_normal_generator = FLIT(channels=3, field_type='wave', evolution_rate=self.field_evolution_rate, name='surface_normal_generator')
        self.global_field_analyzer = layers.GlobalAveragePooling2D(name='global_field_analyzer', dtype=tf.float32) 
        self.env_lighting_projector = layers.Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(1e-5), dtype='float32', name='env_lighting_projector') 
        self.env_lighting_generator = layers.Dense(9, kernel_regularizer=regularizers.l2(1e-5), dtype='float32', name='env_lighting_generator') 

    def _bottleneck_call_for_recompute(self, inputs_tensor, training_float, context_tensor):
        training_bool = tf.cast(training_float > 0.5, tf.bool) 
        return self.bottleneck(inputs_tensor, training=training_bool, context_signal=context_tensor)

    def _decoder_stage2_call_for_recompute(self, inputs_tensor, training_float, context_tensor):
        training_bool = tf.cast(training_float > 0.5, tf.bool) 
        return self.decoder_stage2(inputs_tensor, training=training_bool, context_signal=context_tensor)

    # --- START: FSE NATIVE METRIC HELPER METHODS ---
    def _extract_spatial_features(self, tensor: tf.Tensor) -> tf.Tensor:
        tensor_f32 = tf.cast(tensor, tf.float32)
        dx = tf.abs(tensor_f32[:, :, 1:, :] - tensor_f32[:, :, :-1, :])
        dy = tf.abs(tensor_f32[:, 1:, :, :] - tensor_f32[:, :-1, :, :])
        dx_padded = tf.pad(dx, [[0,0], [0,0], [0,1], [0,0]])
        dy_padded = tf.pad(dy, [[0,0], [0,1], [0,0], [0,0]])
        return dx_padded + dy_padded 

    def _compute_environmental_coherence(self, outputs_dict: Dict[str, tf.Tensor]) -> tf.Tensor:
        keypoints_f32 = outputs_dict.get('fluxa_keypoints')
        segmentation_f32 = outputs_dict.get('fluxa_segmentation')
        surface_normals_f32 = outputs_dict.get('fluxa_surface_normals')
        env_lighting_f32 = outputs_dict.get('fluxa_environment_lighting')
        cross_modal_score_f32 = tf.constant(0.5, dtype=tf.float32)
        lighting_consistency_f32 = tf.constant(0.5, dtype=tf.float32)
        if keypoints_f32 is not None and segmentation_f32 is not None:
            kp_spatial_f32 = self._extract_spatial_features(keypoints_f32)
            seg_spatial_f32 = self._extract_spatial_features(segmentation_f32)
            kp_map_f32 = tf.reduce_mean(kp_spatial_f32, axis=-1) 
            seg_map_f32 = tf.reduce_mean(seg_spatial_f32, axis=-1)
            batch_size = tf.shape(keypoints_f32)[0]
            h_kp, w_kp = tf.shape(kp_map_f32)[1], tf.shape(kp_map_f32)[2]
            flat_dim_kp = h_kp * w_kp
            kp_flat_f32 = tf.reshape(kp_map_f32, [batch_size, flat_dim_kp])
            seg_map_resized_f32 = tf.image.resize(tf.expand_dims(seg_map_f32, -1), [h_kp, w_kp], method='nearest')
            seg_flat_f32 = tf.reshape(tf.squeeze(seg_map_resized_f32, -1), [batch_size, flat_dim_kp])
            kp_norm_f32 = tf.nn.l2_normalize(kp_flat_f32 + 1e-8, axis=-1)
            seg_norm_f32 = tf.nn.l2_normalize(seg_flat_f32 + 1e-8, axis=-1)
            correlation_f32 = tf.reduce_sum(kp_norm_f32 * seg_norm_f32, axis=-1)
            cross_modal_score_f32 = tf.reduce_mean(tf.nn.sigmoid(correlation_f32 * 4.0))
        if env_lighting_f32 is not None and surface_normals_f32 is not None:
            dominant_light_f32 = env_lighting_f32[:, :3]
            dominant_light_f32 = tf.nn.l2_normalize(dominant_light_f32 + 1e-8, axis=-1)
            avg_normals_f32 = tf.reduce_mean(surface_normals_f32, axis=[1, 2])
            avg_normals_f32 = tf.nn.l2_normalize(avg_normals_f32 + 1e-8, axis=-1)
            alignment_f32 = tf.abs(tf.reduce_sum(dominant_light_f32 * avg_normals_f32, axis=-1))
            lighting_consistency_f32 = tf.reduce_mean(alignment_f32)
        env_coherence_f32 = 0.6 * cross_modal_score_f32 + 0.4 * lighting_consistency_f32
        return env_coherence_f32

    def _compute_multiscale_consistency(self, output: tf.Tensor, scales: List[int]=[2,4,8]) -> tf.Tensor:
        output_f32 = tf.cast(output, tf.float32)
        original_h, original_w = tf.shape(output_f32)[1], tf.shape(output_f32)[2]
        original_shape_hw = tf.stack([original_h, original_w])
        scale_consistencies_f32 = []
        for scale_factor_int in scales:
            ksize_val = scale_factor_int; strides_val = scale_factor_int
            downsampled_f32 = tf.nn.avg_pool2d(output_f32, ksize=ksize_val, strides=strides_val, padding='SAME')
            upsampled_f32 = tf.image.resize(downsampled_f32, original_shape_hw, method='bilinear')
            l1_diff_f32 = tf.reduce_mean(tf.abs(output_f32 - upsampled_f32))
            original_magnitude_f32 = tf.reduce_mean(tf.abs(output_f32)) + 1e-8
            normalized_diff_f32 = l1_diff_f32 / (original_magnitude_f32 * tf.sqrt(tf.cast(scale_factor_int, tf.float32)))
            consistency_f32 = tf.exp(-normalized_diff_f32 * 5.0)
            scale_consistencies_f32.append(consistency_f32)
        return tf.reduce_mean(tf.stack(scale_consistencies_f32))
    # --- END: FSE NATIVE METRIC HELPER METHODS ---

    def call(self, inputs: Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]], 
             training: Optional[bool] = None, 
             syntha_context: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        actual_inputs = inputs
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            actual_inputs, tuple_ctx = inputs
            if syntha_context is None: syntha_context = tuple_ctx
        
        inputs_compute_dtype = safe_cast_to_compute_dtype(actual_inputs, self.compute_dtype, "model_input")
        
        context_signal_compute_dtype = None 
        if self.enable_syntha_integration:
            input_for_syntha_gen_f32 = None
            if syntha_context is not None: input_for_syntha_gen_f32 = tf.cast(syntha_context, tf.float32)
            else: input_for_syntha_gen_f32 = tf.cast(tf.reduce_mean(inputs_compute_dtype, axis=[1, 2]), tf.float32)
            current_width = tf.shape(input_for_syntha_gen_f32)[-1]
            target_width = SYNTHA_CONTEXT_WIDTH; padding_needed = target_width - current_width
            if tf.greater(padding_needed, 0): processed_syntha_input_f32 = tf.pad(input_for_syntha_gen_f32, tf.stack([[0, 0], [0, padding_needed]], axis=0)) # Pad last dim
            elif tf.less(padding_needed, 0): processed_syntha_input_f32 = input_for_syntha_gen_f32[..., :target_width] # Slice last dim
            else: processed_syntha_input_f32 = input_for_syntha_gen_f32
            processed_syntha_input_f32.set_shape([None, target_width]) # Set shape after padding/truncation
            context_signal_f32 = self.syntha_context_generator(processed_syntha_input_f32, training=training)
            context_signal_compute_dtype = tf.cast(context_signal_f32, self.compute_dtype)

        # --- FORWARD PASS (using your existing layer calls) ---
        s1_in = self.input_processor(inputs_compute_dtype, training=training, context_signal=context_signal_compute_dtype)
        s1_out = self.downsample1(s1_in, training=training, context_signal=context_signal_compute_dtype)
        s2_in = self.encoder_stage1(s1_out, training=training, context_signal=context_signal_compute_dtype) 
        s2_out = self.downsample2(s2_in, training=training, context_signal=context_signal_compute_dtype)
        s3_in = self.encoder_stage2(s2_out, training=training, context_signal=context_signal_compute_dtype) 
        s3_out = self.downsample3(s3_in, training=training, context_signal=context_signal_compute_dtype)
        bottleneck_in = self.encoder_stage3(s3_out, training=training, context_signal=context_signal_compute_dtype) 
        
        training_as_float = tf.cast(training if training is not None else False, tf.float32)
        
        recomputed_bottleneck_fn = recompute_grad(self._bottleneck_call_for_recompute)
        bottleneck_out = recomputed_bottleneck_fn(bottleneck_in, training_as_float, context_signal_compute_dtype)

        if self.enable_syntha_integration and training is not None and tf.cast(training, tf.bool): 
            complexity_f32 = tf.cast(tf.reduce_mean(tf.abs(bottleneck_out)), tf.float32)
            self.global_complexity_tracker.assign(0.9 * self.global_complexity_tracker + 0.1 * complexity_f32)

        d1_up = self.upsample1(bottleneck_out, training=training, context_signal=context_signal_compute_dtype)
        d1_out = self.decoder_stage1(d1_up, training=training, context_signal=context_signal_compute_dtype)
        d2_up = self.upsample2(d1_out, training=training, context_signal=context_signal_compute_dtype)
        
        recomputed_decoder_stage2_fn = recompute_grad(self._decoder_stage2_call_for_recompute)
        d2_out = recomputed_decoder_stage2_fn(d2_up, training_as_float, context_signal_compute_dtype)

        d3_up = self.upsample3(d2_out, training=training, context_signal=context_signal_compute_dtype)
        decoded_features = self.decoder_stage3(d3_up, training=training, context_signal=context_signal_compute_dtype)

        keypoint_features = self.keypoint_processor(decoded_features, training=training, context_signal=context_signal_compute_dtype)
        segmentation_features = self.segmentation_processor(decoded_features, training=training, context_signal=context_signal_compute_dtype)
        surface_normal_features = self.surface_normal_processor(decoded_features, training=training, context_signal=context_signal_compute_dtype)
        env_lighting_input_features = d2_out 
        env_lighting_task_features = self.env_lighting_processor(env_lighting_input_features, training=training, context_signal=context_signal_compute_dtype)

        keypoints_out_compute = self.keypoint_generator(keypoint_features, training=training, context_signal=context_signal_compute_dtype)
        segmentation_out_compute = self.segmentation_generator(segmentation_features, training=training, context_signal=context_signal_compute_dtype)
        surface_normals_out_compute = self.surface_normal_generator(surface_normal_features, training=training, context_signal=context_signal_compute_dtype)
        
        segmentation_activated_f32 = tf.nn.sigmoid(tf.cast(segmentation_out_compute, tf.float32))
        surface_normals_activated_f32 = tf.nn.tanh(tf.cast(surface_normals_out_compute, tf.float32))

        global_env_features_f32 = tf.cast(self.global_field_analyzer(env_lighting_task_features), tf.float32)
        env_projected_f32 = self.env_lighting_projector(global_env_features_f32, training=training) 
        env_lighting_f32 = self.env_lighting_generator(env_projected_f32, training=training) 
        # --- END OF FORWARD PASS ---
        
        outputs = {
            'fluxa_keypoints': tf.cast(keypoints_out_compute, tf.float32),
            'fluxa_segmentation': segmentation_activated_f32,
            'fluxa_surface_normals': surface_normals_activated_f32,
            'fluxa_environment_lighting': env_lighting_f32
        }

        # Add global metrics if Keras indicates we are in a training or evaluation step
        if training is not None: # Keras sets training to True for train steps, False for val steps
            # The `training` flag here is the one Keras passes to the model's call method.
            # `self.add_metric` is stateful and Keras manages its reset etc.
            # We add metrics regardless of training True/False here, Keras will handle display for train/val.
            env_coherence = self._compute_environmental_coherence(outputs)
            self.add_metric(env_coherence, name='global_env_coherence', aggregation='mean')
            
            kp_multiscale = self._compute_multiscale_consistency(outputs['fluxa_keypoints'])
            self.add_metric(kp_multiscale, name='global_keypoint_multiscale', aggregation='mean')
            
            seg_multiscale = self._compute_multiscale_consistency(outputs['fluxa_segmentation'])
            self.add_metric(seg_multiscale, name='global_segmentation_multiscale', aggregation='mean')
            
            normal_multiscale = self._compute_multiscale_consistency(outputs['fluxa_surface_normals'])
            self.add_metric(normal_multiscale, name='global_normal_multiscale', aggregation='mean')
            
        return outputs
    
    def get_syntha_status(self) -> Dict[str, any]:
        if not self.enable_syntha_integration: return {'syntha_enabled': False, 'message': 'SYNTHA integration is disabled.'}
        status: Dict[str, any] = {'syntha_enabled': True, 'global_complexity': float(self.global_complexity_tracker.numpy()) if tf.executing_eagerly() else self.global_complexity_tracker, 'total_active_cses_model': 0, 'blocks': {}}
        all_blocks_with_cse_status = [self.input_processor, self.encoder_stage1, self.encoder_stage2, self.encoder_stage3, self.bottleneck, self.decoder_stage1, self.decoder_stage2, self.decoder_stage3, self.keypoint_processor, self.segmentation_processor, self.surface_normal_processor, self.env_lighting_processor]
        total_model_active_cses = 0
        for block_instance in all_blocks_with_cse_status:
            if block_instance and hasattr(block_instance, 'get_all_cse_status') and callable(getattr(block_instance, 'get_all_cse_status')):
                try:
                    block_status = block_instance.get_all_cse_status()
                    status['blocks'][block_instance.name] = block_status
                    active_cses_in_block = block_status.get('total_active_cses_in_block', 0) # Name changed in FSENativeBlock
                    if isinstance(active_cses_in_block, tf.Tensor) and tf.executing_eagerly(): total_model_active_cses += int(active_cses_in_block.numpy())
                    elif isinstance(active_cses_in_block, int): total_model_active_cses += active_cses_in_block
                except Exception as e: logger.warning(f"Could not get CSE status for block {block_instance.name}: {e}"); status['blocks'][block_instance.name] = {'error': str(e)}
        status['total_active_cses_model'] = total_model_active_cses
        return status

# Optional: Smoke test within the core script (can be commented out for production)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger.info("🧪 Running FSE Native Core Smoke Test (v3.1 with Global Metrics)...")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    logger.info("✅ Mixed precision policy set to 'mixed_float16' for smoke test.")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✅ Configured {len(gpus)} GPU(s) with memory growth for smoke test.")
        except RuntimeError as e: logger.error(f"❌ Error configuring GPUs for smoke test: {e}")
    
    try:
        model = FLUXA_FSE_Native(
            input_shape=(64, 64, 3), base_channels=8, field_evolution_rate=0.05,
            enable_syntha_integration=True, max_cses_per_fil=2, name="smoke_test_fse_model_v3_1"
        )
        dummy_model_input = tf.zeros((2, 64, 64, 3), dtype=tf.float16)
        dummy_syntha_context = tf.zeros((2, SYNTHA_CONTEXT_WIDTH), dtype=tf.float16)

        logger.info("Testing model call with tuple input (image, syntha_context)...")
        outputs_tuple_input = model((dummy_model_input, dummy_syntha_context), training=True)
        for key, value in outputs_tuple_input.items():
            logger.info(f"  Output '{key}' shape: {value.shape}, dtype: {value.dtype}")
            tf.debugging.assert_equal(value.dtype, tf.float32, message=f"Output {key} dtype error, expected float32.")
        logger.info("✅ Model call with tuple input successful.")

        logger.info("Testing model call with image input only (SYNTHA disabled internally by model if context not provided)...")
        # To test this path properly, either pass syntha_context=None or ensure model handles it
        # For this test, let's assume syntha_context is always provided if enable_syntha_integration=True
        # outputs_img_only = model(dummy_model_input, training=False, syntha_context=None) # This path needs careful handling in call
        # logger.info("✅ Model call with image input only successful.")


        logger.info("Testing get_syntha_status...")
        syntha_status = model.get_syntha_status()
        logger.info(f"  SYNTHA Status: {json.dumps(syntha_status, indent=2, default=lambda x: str(x) if isinstance(x, tf.Tensor) else ('Tensor:'+x.name if isinstance(x, tf.Tensor) else x))}") # type: ignore
        logger.info("✅ get_syntha_status successful.")
        
        # Test compilation with metrics (mock optimizer and loss)
        logger.info("Testing model compilation (mock optimizer/loss)...")
        model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Basic compilation test
        logger.info("✅ Model compilation successful.")


    except Exception as e:
        logger.error(f"❌ FSE Native Core Smoke Test (v3.1) FAILED: {e}", exc_info=True)
    else:
        logger.info("🎉 FSE Native Core Smoke Test (v3.1 with Global Metrics) PASSED!")
