#!/usr/bin/env python3
"""
Robust Augmentation Pipeline for Grayscale Images
Simplified version for single channel grayscale training data
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import json
import os
import random
from pathlib import Path
import argparse

class RealisticRainGeneration:
    """Rain generation for grayscale images"""
    
    def __init__(self):
        self.rain_types = ['light', 'medium', 'heavy', 'torrential']
        
    def generate_base_rain_streaks(self, H, W, rain_density='medium'):
        """Generate rain streaks for grayscale images"""
        rain_image = Image.new('L', (W, H), 0)
        draw = ImageDraw.Draw(rain_image)
        
        density_params = {
            'light': {'num_streaks': (20, 50), 'length': (15, 40), 'width': (1, 2), 'intensity': (80, 150)},
            'medium': {'num_streaks': (50, 120), 'length': (25, 60), 'width': (1, 3), 'intensity': (100, 200)},
            'heavy': {'num_streaks': (120, 250), 'length': (40, 80), 'width': (2, 4), 'intensity': (150, 255)},
            'torrential': {'num_streaks': (200, 400), 'length': (50, 100), 'width': (2, 5), 'intensity': (180, 255)}
        }
        
        params = density_params[rain_density]
        num_streaks = random.randint(*params['num_streaks'])
        
        for _ in range(num_streaks):
            length = random.randint(*params['length'])
            width = random.randint(*params['width'])
            intensity = random.randint(*params['intensity'])
            
            angle = random.uniform(-15, 15)
            start_x = random.randint(0, W-1)
            start_y = random.randint(0, max(1, H//3))
            
            dx = length * np.sin(np.radians(angle))
            dy = length * np.cos(np.radians(angle))
            
            end_x = int(start_x + dx)
            end_y = int(start_y + dy)
            
            end_x = np.clip(end_x, 0, W-1)
            end_y = np.clip(end_y, 0, H-1)
            
            for w in range(width):
                try:
                    draw.line([(start_x + w//2, start_y), (end_x + w//2, end_y)], 
                             fill=intensity, width=1)
                except:
                    continue
        
        rain_map = np.array(rain_image).astype(np.float32) / 255.0
        return rain_map
    
    def apply_rain_transformations(self, rain_map):
        """Apply transformations to rain map"""
        H, W = rain_map.shape
        rain_pil = Image.fromarray((rain_map * 255).astype(np.uint8))
        
        # Rotation
        if random.random() < 0.8:
            angle = random.uniform(-10, 10)
            rain_pil = rain_pil.rotate(angle, fillcolor=0)
        
        # Scaling
        if random.random() < 0.7:
            scale_factor = random.uniform(0.9, 1.1)
            new_size = (int(W * scale_factor), int(H * scale_factor))
            rain_pil = rain_pil.resize(new_size).resize((W, H))
        
        # Translation
        if random.random() < 0.6:
            shift_x = random.randint(-W//20, W//20)
            shift_y = random.randint(-H//20, H//20)
            new_rain = Image.new('L', (W, H), 0)
            new_rain.paste(rain_pil, (shift_x, shift_y))
            rain_pil = new_rain
        
        transformed_rain = np.array(rain_pil).astype(np.float32) / 255.0
        return transformed_rain
    
    def generate_rainmix_augmentation(self, image):
        """RainMix for grayscale images"""
        img_array = np.array(image)
        H, W = img_array.shape
        
        rain_density = random.choice(self.rain_types)
        base_rain = self.generate_base_rain_streaks(H, W, rain_density)
        
        # Generate multiple rain patterns and mix them
        rain_augmentations = []
        for i in range(3):
            aug_rain = self.apply_rain_transformations(base_rain.copy())
            rain_augmentations.append(aug_rain)
        
        # Simple weighted combination
        weights = np.random.dirichlet([1, 1, 1])
        mixed_rain = np.zeros_like(base_rain)
        for i, (aug_rain, weight) in enumerate(zip(rain_augmentations, weights)):
            mixed_rain += weight * aug_rain
        
        # Blend with base
        beta = np.random.beta(2, 5)
        final_rain = beta * base_rain + (1 - beta) * mixed_rain
        
        # Add rain to image
        rainy_image = self.blend_rain_with_grayscale_image(img_array, final_rain)
        
        return Image.fromarray(rainy_image.astype(np.uint8)), f"RainMix: {rain_density}"
    
    def blend_rain_with_grayscale_image(self, image_array, rain_map):
        """Blend rain with grayscale image"""
        H, W = image_array.shape
        rainy_image = image_array.copy().astype(np.float32)
        
        # Add rain streaks (brighten image where rain falls)
        rain_addition = rain_map * 80
        
        # Add atmospheric effect for heavy rain
        rain_intensity = np.mean(rain_map)
        if rain_intensity > 0.3:
            # Dim the image slightly for heavy rain atmosphere
            rainy_image *= (0.9 - rain_intensity * 0.2)
            # Add atmospheric haze
            haze = np.ones_like(rainy_image) * 180 * rain_intensity * 0.5
            alpha = rain_map * 0.7
            rainy_image = rainy_image * (1 - alpha) + haze * alpha
        
        # Apply rain streaks
        rainy_image += rain_addition
        rainy_image = np.clip(rainy_image, 0, 255)
        
        return rainy_image

class WeatherSimulation:
    """Weather effects for grayscale images"""
    
    def __init__(self):
        self.weather_types = ['night', 'realistic_rain', 'fog', 'overcast']
        self.rain_generator = RealisticRainGeneration()
    
    def apply_night_effect(self, image):
        """Make image darker (night effect)"""
        enhancer = ImageEnhance.Brightness(image)
        dark_image = enhancer.enhance(0.6)
        return dark_image
    
    def apply_realistic_rain_effect(self, image):
        """Apply realistic rain"""
        rainy_image, info = self.rain_generator.generate_rainmix_augmentation(image)
        return rainy_image
    
    def apply_fog_effect(self, image):
        """Add fog effect to grayscale image"""
        img_array = np.array(image).astype(np.float32)
        H, W = img_array.shape
        
        # Create fog intensity map
        center_x, center_y = W//2, H//2
        y_coords, x_coords = np.ogrid[:H, :W]
        
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        fog_intensity = (distances / max_distance) * 0.4 + 0.1
        
        # Apply fog (blend with gray color)
        fog_color = 200  # Light gray for grayscale
        fogged = img_array * (1 - fog_intensity) + fog_color * fog_intensity
        fogged = np.clip(fogged, 0, 255)
        
        return Image.fromarray(fogged.astype(np.uint8))
    
    def apply_overcast_effect(self, image):
        """Overcast effect - dim the image"""
        enhancer = ImageEnhance.Brightness(image)
        dimmed = enhancer.enhance(0.75)
        return dimmed

class ColorJitterAugmentation:
    """Brightness/contrast jittering for grayscale"""
    
    def __init__(self, brightness_range=(0.7, 1.4), contrast_range=(0.7, 1.4)):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
    
    def apply_jitter(self, image):
        """Apply brightness and contrast jittering"""
        jittered = image.copy()
        
        # Random brightness
        brightness_factor = random.uniform(*self.brightness_range)
        enhancer = ImageEnhance.Brightness(jittered)
        jittered = enhancer.enhance(brightness_factor)
        
        # Random contrast
        contrast_factor = random.uniform(*self.contrast_range)
        enhancer = ImageEnhance.Contrast(jittered)
        jittered = enhancer.enhance(contrast_factor)
        
        return jittered, f"B:{brightness_factor:.2f},C:{contrast_factor:.2f}"

class DynamicMaskingProcess:
    """Dynamic masking for grayscale images"""
    
    def __init__(self, patch_size=8):
        self.patch_size = patch_size
        
    def apply_to_pil(self, image):
        """Apply masking to PIL grayscale image"""
        img_array = np.array(image)
        H, W = img_array.shape  # Grayscale only has 2 dimensions
        
        # Random mask ratio
        mask_ratio = random.uniform(0, 0.6)
        
        # Calculate patches
        patch_h = H // self.patch_size
        patch_w = W // self.patch_size
        total_patches = patch_h * patch_w
        num_masked = int(total_patches * mask_ratio)
        
        if num_masked == 0:
            return image, f"mask_ratio: 0"
            
        # Select random patches
        masked_patches = random.sample(range(total_patches), num_masked)
        
        # Apply masking
        masked_array = img_array.copy()
        for patch_id in masked_patches:
            row = (patch_id // patch_w) * self.patch_size
            col = (patch_id % patch_w) * self.patch_size
            masked_array[row:row+self.patch_size, col:col+self.patch_size] = 0
            
        return Image.fromarray(masked_array), f"mask_ratio: {mask_ratio:.2f}"

class RobustAugmentationPipeline:
    """Combined augmentation pipeline for grayscale images"""
    
    def __init__(self, patch_size=8):
        self.weather_sim = WeatherSimulation()
        self.color_jitter = ColorJitterAugmentation()
        self.dynamic_masking = DynamicMaskingProcess(patch_size)
        
    def apply_combined_augmentation(self, image):
        """Apply combined augmentations to grayscale image"""
        augmented = image.copy()
        aug_info = []
        applied_effects = []
        
        # Weather effects (choose one)
        weather_chance = random.random()
        if weather_chance < 0.15:  # 15% night
            augmented = self.weather_sim.apply_night_effect(augmented)
            applied_effects.append("dark")
            aug_info.append("Weather: night")
        elif weather_chance < 0.35:  # 20% rain
            augmented = self.weather_sim.apply_realistic_rain_effect(augmented)
            applied_effects.append("rain")
            aug_info.append("Weather: rain")
        elif weather_chance < 0.50:  # 15% fog
            augmented = self.weather_sim.apply_fog_effect(augmented)
            applied_effects.append("fog")
            aug_info.append("Weather: fog")
        elif weather_chance < 0.65:  # 15% overcast
            augmented = self.weather_sim.apply_overcast_effect(augmented)
            applied_effects.append("dim")
            aug_info.append("Weather: overcast")
        
        # Brightness/contrast jittering
        jitter_chance = 0.6 if "dark" in applied_effects or "dim" in applied_effects else 0.8
        if random.random() < jitter_chance:
            if "dark" in applied_effects:
                # Lighter jittering for already dark images
                light_jitter = ColorJitterAugmentation(
                    brightness_range=(0.9, 1.3),
                    contrast_range=(0.8, 1.2)
                )
                augmented, jitter_info = light_jitter.apply_jitter(augmented)
                aug_info.append("Light Jitter")
            else:
                augmented, jitter_info = self.color_jitter.apply_jitter(augmented)
                aug_info.append("Jitter")
        
        # Masking
        mask_chance = 0.4 if len(applied_effects) > 0 else 0.6
        if random.random() < mask_chance:
            augmented, mask_info = self.dynamic_masking.apply_to_pil(augmented)
            aug_info.append("Masking")
            
        return augmented, " + ".join(aug_info) if aug_info else "No augmentation"