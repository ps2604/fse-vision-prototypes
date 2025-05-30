#!/usr/bin/env python3
"""
FSE (Float-Native State Elements) Core Infrastructure - FIXED
Revolutionary continuous field-based neural computation architecture
Fixed all Keras lifecycle violations and variable management issues
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers
from typing import Dict, List, Optional, Tuple, Union


class FLIT(layers.Layer):
    """
    Floating Information Unit - Basic data packet of FSE architecture
    Represents multi-dimensional float values with adaptive precision
    """
    
    def __init__(self, 
                 channels: int,
                 precision_bits: int = 16,
                 field_type: str = 'continuous',
                 name: str = None,
                 **kwargs):
        # Strip custom kwargs before super().__init__
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        
        super(FLIT, self).__init__(name=name, **kwargs)
        self.channels = channels
        self.precision_bits = precision_bits
        self.field_type = field_type
        self.custom_cfg = custom_cfg
        
        # Placeholders - will be created in build()
        self.field_state = None
        self.evolution_rate = None
        
    def build(self, input_shape):
        # Initialize learnable float state - ONLY in build()
        self.field_state = self.add_weight(
            name='field_state',
            shape=(1, 1, 1, self.channels),
            initializer='glorot_uniform',
            trainable=True,
            dtype=self.dtype
        )
        
        # Field evolution parameters - ONLY in build()
        self.evolution_rate = self.add_weight(
            name='evolution_rate',
            shape=(1,),
            initializer=tf.keras.initializers.Constant(0.1),
            trainable=True,
            dtype=self.dtype
        )
        
        super(FLIT, self).build(input_shape)
    
    def call(self, inputs, training=None):
        # Process through continuous field interaction - NO variable creation here
        # Use broadcasting - no explicit tiling
        field_influence = inputs * self.field_state
        
        # Apply field evolution equation: S_(t+1) = φ(S_t + ΔS)
        delta_influence = field_influence * self.evolution_rate
        evolved_field = tf.nn.tanh(inputs + delta_influence)
        
        return evolved_field
    
    def get_field_state(self):
        """Return current continuous state"""
        return self.field_state
    
    def evolve_field(self, delta_influence):
        """Apply field evolution - cast to variable dtype"""
        delta_casted = tf.cast(delta_influence * self.evolution_rate, self.field_state.dtype)
        self.field_state.assign_add(delta_casted)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'precision_bits': self.precision_bits,
            'field_type': self.field_type,
            **self.custom_cfg
        })
        return config


class CSE(layers.Layer):
    """
    Continuous State Element - FIXED
    Maintains persistent 1×1×1×C memory with proper broadcasting
    """

    def __init__(self,
                 l2_reg: float = 5e-6,
                 momentum: float = 0.9,
                 name: str = None,
                 **kwargs):
        # Strip custom kwargs to avoid Keras errors
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        self.custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        
        super().__init__(name=name, **kwargs)
        
        self.l2_reg = l2_reg
        self.momentum = momentum
        
        # Placeholders - will be created in build()
        self.proj_conv = None
        self.state_memory = None

    def build(self, input_shape):
        channels = input_shape[-1]

        # Create projection layer - ONLY in build()
        self.proj_conv = layers.Conv2D(
            filters=channels,
            kernel_size=1,
            padding="same",
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="proj_conv",
        )
        self.proj_conv.build(input_shape)  # Explicitly build sub-layer

        # Create state memory - ONLY in build(), keep 1×1×1×C always
        self.state_memory = self.add_weight(
            name="state_memory",
            shape=(1, 1, 1, channels),
            initializer="zeros",
            trainable=False,
            dtype=self.dtype  # Match layer dtype
        )

        super().build(input_shape)

    def compute_field_influence(self, x, training=None):
        return self.proj_conv(x, training=training)

    def call(self, inputs, training=None):
        """
        inputs: [B, H, W, C] field tensor
        returns: updated state field (same shape)
        """
        # 1. Project incoming field - no variable creation
        new_state = self.compute_field_influence(inputs, training=training)

        # 2. Momentum mix with broadcastable 1×1×1×C memory
        # Broadcasting happens automatically - no explicit tiling
        updated_state = (
            self.momentum * self.state_memory +
            (1.0 - self.momentum) * new_state
        )

        # 3. Update persistent memory - always keep 1×1×1×C shape
        avg_state = tf.reduce_mean(updated_state, axis=[0, 1, 2], keepdims=True)
        
        # 4. Cast to variable dtype before assignment
        avg_state = tf.cast(avg_state, self.state_memory.dtype)
        self.state_memory.assign(avg_state)

        # 5. Return the updated state (full spatial resolution)
        return updated_state

    def get_config(self):
        config = super().get_config()
        config.update({
            "l2_reg": self.l2_reg,
            "momentum": self.momentum,
            **self.custom_cfg
        })
        return config

    def get_combined_field(self):
        """Return unified CSE field representation"""
        return self.state_memory

    def apply_field_influence(self, influence_tensor):
        """External field coordination"""
        influence_casted = tf.cast(influence_tensor, self.state_memory.dtype)
        self.state_memory.assign_add(influence_casted * 0.01)  # Small influence


class FIL(layers.Layer):
    """
    Field Interaction Layer - FIXED
    Core computational unit with proper Keras lifecycle management
    """
    
    def __init__(self,
                 num_cses: int,
                 cse_config: Dict,
                 field_type: str = 'continuous',
                 spatial_kernel_size: int = 3,
                 l2_reg: float = 1e-5,
                 dropout_rate: float = 0.0,
                 name: str = None,
                 **kwargs):
        # Strip custom kwargs
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        
        super(FIL, self).__init__(name=name, **kwargs)
        
        self.num_cses = num_cses
        self.cse_config = cse_config
        self.field_type = field_type
        self.spatial_kernel_size = spatial_kernel_size
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.custom_cfg = custom_cfg
        
        # Placeholders - will be created in build()
        self.cses = []
        self.input_projection = None
        self.spatial_conv = None
        self.batch_norm = None
        self.dropout = None
        self.output_projection = None
        
        # Field-specific placeholders
        self.wave_conv = None
        self.quantum_conv = None
        self.entangle_mix = None
        self.frequency = None
        self.phase = None
        self.superposition_weights = None
    
    def build(self, input_shape):
        input_channels = input_shape[-1]
        projected_channels = self.cse_config.get('flit_channels', 16)

        # Create all layers in build() - NEVER in call()
        self.input_projection = layers.Conv2D(
            projected_channels,
            kernel_size=1,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name=f'{self.name}_input_proj' if self.name else 'input_proj'
        )

        # Create CSEs - ONLY in build()
        self.cses = []
        for i in range(self.num_cses):
            cse = CSE(
                l2_reg=self.l2_reg,
                name=f'{self.name}_cse_{i}' if self.name else f'cse_{i}',
                **self.cse_config  # Pass config as kwargs
            )
            self.cses.append(cse)

        # Spatial processing layers
        self.spatial_conv = layers.Conv2D(
            filters=projected_channels,
            kernel_size=self.spatial_kernel_size,
            padding='same',
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name=f'{self.name}_spatial_conv' if self.name else 'spatial_conv'
        )

        self.batch_norm = layers.BatchNormalization(
            name=f'{self.name}_bn' if self.name else 'bn'
        )

        if self.dropout_rate > 0:
            self.dropout = layers.Dropout(self.dropout_rate)

        # Output projection
        self.output_projection = layers.Conv2D(
            input_channels,  # Project back to input space
            kernel_size=1,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name=f'{self.name}_output_proj' if self.name else 'output_proj'
        )

        # Field-specific layers - create in build()
        if self.field_type == 'wave':
            self.frequency = self.add_weight(
                name='wave_frequency',
                shape=(1,),
                initializer=tf.keras.initializers.Constant(2.0 * np.pi),
                trainable=True,
                dtype=self.dtype
            )
            self.phase = self.add_weight(
                name='wave_phase',
                shape=(2,),
                initializer='zeros',
                trainable=True,
                dtype=self.dtype
            )
            self.wave_conv = layers.Conv2D(
                filters=projected_channels,
                kernel_size=self.spatial_kernel_size,
                padding='same',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'{self.name}_wave_conv' if self.name else 'wave_conv'
            )

        if self.field_type == 'quantum':
            self.superposition_weights = self.add_weight(
                name='superposition_weights',
                shape=(self.num_cses, self.num_cses),
                initializer='orthogonal',
                trainable=True,
                dtype=self.dtype
            )
            self.quantum_conv = layers.Conv2D(
                filters=projected_channels,
                kernel_size=self.spatial_kernel_size,
                padding='same',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'{self.name}_quantum_conv' if self.name else 'quantum_conv'
            )

        # Build sub-layers explicitly
        projected_shape = input_shape[:-1] + (projected_channels,)
        self.input_projection.build(input_shape)
        self.spatial_conv.build(projected_shape)
        self.batch_norm.build(projected_shape)
        self.output_projection.build(projected_shape)
        
        for cse in self.cses:
            cse.build(projected_shape)
        
        if self.wave_conv:
            self.wave_conv.build(projected_shape)
        if self.quantum_conv:
            self.quantum_conv.build(projected_shape)

        super(FIL, self).build(input_shape)
    
    def continuous_field_transform(self, inputs):
        """Smooth spatial transformation"""
        return self.spatial_conv(inputs)
    
    def wave_field_interaction(self, inputs):
        """Wave-based field interaction with simple modulation"""
        # Simple wave modulation
        mean_field = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        wave_pattern = tf.sin(self.frequency * mean_field)
        
        # Apply wave conv if available
        if self.wave_conv:
            wave_processed = self.wave_conv(inputs)
            return wave_processed * wave_pattern
        else:
            return inputs * wave_pattern
    
    def quantum_field_superposition(self, cse_outputs):
        """Quantum-inspired field superposition"""
        if len(cse_outputs) < 2:
            return cse_outputs[0] if cse_outputs else tf.zeros_like(cse_outputs[0])
        
        # Simple mixing of CSE outputs
        stacked = tf.stack(cse_outputs[:2], axis=-1)  # Limit to prevent memory issues
        mixed = tf.reduce_mean(stacked, axis=-1)
        
        if self.quantum_conv:
            return self.quantum_conv(mixed)
        return mixed
    
    def call(self, inputs, training=None):
        # NO variable or layer creation in call() - only computations
        
        # Project inputs
        projected = self.input_projection(inputs, training=training)
        
        # Process through CSEs - no layer creation
        cse_outputs = []
        for cse in self.cses:
            cse_out = cse(projected, training=training)
            cse_outputs.append(cse_out)
        
        # Apply field-specific transformations
        if self.field_type == 'continuous':
            combined = tf.reduce_mean(tf.stack(cse_outputs), axis=0)
            field_output = self.continuous_field_transform(combined)
        elif self.field_type == 'wave':
            combined = tf.reduce_mean(tf.stack(cse_outputs), axis=0)
            field_output = self.wave_field_interaction(combined)
        elif self.field_type == 'quantum':
            field_output = self.quantum_field_superposition(cse_outputs)
        else:
            field_output = tf.reduce_mean(tf.stack(cse_outputs), axis=0)
        
        # Apply normalization and dropout
        field_output = self.batch_norm(field_output, training=training)
        
        if training and self.dropout_rate > 0 and self.dropout:
            field_output = self.dropout(field_output, training=training)
        
        # Project back to original space
        output = self.output_projection(field_output, training=training)
        
        # Residual connection if shapes match
        if inputs.shape[-1] == output.shape[-1]:
            output = output + inputs
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_cses': self.num_cses,
            'cse_config': self.cse_config,
            'field_type': self.field_type,
            'spatial_kernel_size': self.spatial_kernel_size,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            **self.custom_cfg
        })
        return config


class FSEBlock(layers.Layer):
    """
    Composite FSE block - FIXED
    Combines multiple FILs with proper layer management
    """
    
    def __init__(self,
                 filters: int,
                 num_fils: int = 2,  # Reduced to prevent memory issues
                 fil_types: List[str] = None,
                 spatial_kernel_size: int = 3,
                 l2_reg: float = 1e-5,
                 dropout_rate: float = 0.0,
                 name: str = None,
                 **kwargs):
        keras_allowed = {"trainable", "dtype", "dynamic", "autocast"}
        custom_cfg = {k: kwargs.pop(k) for k in tuple(kwargs) if k not in keras_allowed}
        
        super(FSEBlock, self).__init__(name=name, **kwargs)
        
        self.filters = filters
        self.num_fils = num_fils
        self.fil_types = fil_types or ['continuous', 'wave']
        self.spatial_kernel_size = spatial_kernel_size
        self.l2_reg = l2_reg
        self.dropout_rate = dropout_rate
        self.custom_cfg = custom_cfg
        
        # Reduce CSE complexity to prevent memory issues
        self.cse_config = {
            'num_flits': 2,  # Reduced
            'flit_channels': max(8, filters // 8),  # Reduced
            'evolution_rate': 0.1,
            'coherence_type': 'adaptive'
        }
        
        # Placeholders
        self.fils = []
    
    def build(self, input_shape):
        # Create FILs in build() - NEVER in call()
        self.fils = []
        for i, field_type in enumerate(self.fil_types[:self.num_fils]):
            fil = FIL(
                num_cses=2,  # Keep small
                cse_config=self.cse_config,
                field_type=field_type,
                spatial_kernel_size=self.spatial_kernel_size,
                l2_reg=self.l2_reg,
                dropout_rate=self.dropout_rate,
                name=f'{self.name}_fil_{i}' if self.name else f'fil_{i}'
            )
            fil.build(input_shape)  # Explicitly build
            self.fils.append(fil)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        x = inputs
        
        # Sequential processing through FILs - no layer creation
        for fil in self.fils:
            x = fil(x, training=training)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'num_fils': self.num_fils,
            'fil_types': self.fil_types,
            'spatial_kernel_size': self.spatial_kernel_size,
            'l2_reg': self.l2_reg,
            'dropout_rate': self.dropout_rate,
            **self.custom_cfg
        })
        return config


def create_fse_backbone(input_shape=(480, 640, 3),
                       base_filters=16,  # Reduced default
                       l2_reg=1e-5,
                       dropout_rate=0.1):
    """
    Create FSE backbone - memory-optimized
    """
    inputs = layers.Input(shape=input_shape, name='fse_input')
    
    # Initial projection - smaller
    x = layers.Conv2D(
        base_filters,
        kernel_size=3,  # Smaller kernel
        strides=2,
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='initial_projection'
    )(inputs)
    x = layers.BatchNormalization(name='initial_bn')(x)
    x = layers.Activation('relu', name='initial_activation')(x)
    
    # FSE blocks - simplified and memory-efficient
    x = FSEBlock(
        filters=base_filters * 2,
        num_fils=2,
        fil_types=['continuous', 'wave'],
        spatial_kernel_size=3,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        name='fse_block_1'
    )(x)
    x = layers.MaxPooling2D(pool_size=2, name='pool_1')(x)
    
    x = FSEBlock(
        filters=base_filters * 4,
        num_fils=2,
        fil_types=['wave', 'continuous'],
        spatial_kernel_size=3,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        name='fse_block_2'
    )(x)
    x = layers.MaxPooling2D(pool_size=2, name='pool_2')(x)
    
    x = FSEBlock(
        filters=base_filters * 8,
        num_fils=2,
        fil_types=['continuous', 'quantum'],
        spatial_kernel_size=3,
        l2_reg=l2_reg,
        dropout_rate=dropout_rate,
        name='fse_block_3'
    )(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name='fse_backbone')


# Utility functions
def create_fse_upsampling_block(filters, 
                               size=(2, 2), 
                               l2_reg=1e-5, 
                               dropout_rate=0.0,
                               name=None):
    """Memory-efficient FSE upsampling block"""
    def block(x):
        x = layers.UpSampling2D(size=size, interpolation='bilinear', 
                               name=f'{name}_upsample' if name else None)(x)
        x = FSEBlock(
            filters=filters,
            num_fils=1,  # Single FIL for efficiency
            fil_types=['continuous'],
            spatial_kernel_size=3,
            l2_reg=l2_reg,
            dropout_rate=dropout_rate,
            name=f'{name}_fse' if name else None
        )(x)
        return x
    return block


def create_fse_skip_connection(encoder_features, decoder_features, 
                              filters, l2_reg=1e-5, name=None):
    """Simple FSE skip connection"""
    # Simple addition - avoid complex field operations for memory
    if encoder_features.shape[-1] != decoder_features.shape[-1]:
        encoder_proj = layers.Conv2D(
            decoder_features.shape[-1],
            kernel_size=1,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f'{name}_proj' if name else None
        )(encoder_features)
    else:
        encoder_proj = encoder_features
    
    return layers.Add(name=f'{name}_add' if name else None)([encoder_proj, decoder_features])