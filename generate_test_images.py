#!/usr/bin/env python3
# Generate test images for image algebra operations
import cv2
import numpy as np
import os
from pathlib import Path

def create_output_dir(dir_name="test_images"):
    """Create output directory for test images"""
    output_dir = Path(dir_name)
    output_dir.mkdir(exist_ok=True)
    return output_dir

def generate_gradient_image(size=(256, 256), horizontal=True):
    """Generate a horizontal or vertical gradient image"""
    if horizontal:
        gradient = np.linspace(0, 255, size[1], dtype=np.uint8)
        gradient = np.tile(gradient, (size[0], 1))
    else:
        gradient = np.linspace(0, 255, size[0], dtype=np.uint8)
        gradient = np.tile(gradient[:, np.newaxis], (1, size[1]))
    
    return gradient

def generate_checkerboard(size=(256, 256), tile_size=32):
    """Generate a checkerboard pattern image"""
    x, y = np.indices(size)
    checkerboard = ((x // tile_size) % 2 == 0) ^ ((y // tile_size) % 2 == 0)
    return checkerboard.astype(np.uint8) * 255

def generate_circle_pattern(size=(256, 256), num_circles=5):
    """Generate a pattern with concentric circles"""
    circle_img = np.zeros(size, dtype=np.uint8)
    center = (size[1] // 2, size[0] // 2)
    
    max_radius = min(size) // 2
    step = max_radius // num_circles
    
    for i in range(num_circles):
        radius = (i + 1) * step
        cv2.circle(circle_img, center, radius, 255 - (i * 255 // num_circles), -1 if i % 2 == 0 else 2)
    
    return circle_img

def generate_text_image(size=(256, 256), text="DIP"):
    """Generate an image with text"""
    text_img = np.zeros(size, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 2, 5)[0]
    
    # Calculate position to center the text
    position = ((size[1] - text_size[0]) // 2, (size[0] + text_size[1]) // 2)
    
    cv2.putText(text_img, text, position, font, 2, 255, 5)
    
    return text_img

def generate_noise_image(size=(256, 256), noise_type="gaussian", intensity=50):
    """Generate a noise image"""
    if noise_type == "gaussian":
        # Create a base grayscale image (e.g., mid-gray)
        base_img = np.ones(size, dtype=np.uint8) * 128
        
        # Generate Gaussian noise
        noise = np.random.normal(0, intensity, size).astype(np.int16)
        
        # Add noise to base image and clip to valid range
        noisy_img = np.clip(base_img + noise, 0, 255).astype(np.uint8)
        return noisy_img
    
    elif noise_type == "salt_pepper":
        # Create a base grayscale image (e.g., mid-gray)
        base_img = np.ones(size, dtype=np.uint8) * 128
        
        # Add salt and pepper noise
        noise_mask = np.random.random(size)
        
        # Salt noise
        salt_mask = noise_mask > (1.0 - intensity/255)
        base_img[salt_mask] = 255
        
        # Pepper noise
        pepper_mask = noise_mask < (intensity/255)
        base_img[pepper_mask] = 0
        
        return base_img
    
    elif noise_type == "random":
        # Pure random noise
        return np.random.randint(0, 256, size, dtype=np.uint8)
    
    return None

def main():
    output_dir = create_output_dir()
    
    # Generate different test images
    size = (256, 256)
    
    # Generate and save images
    images = {
        "horizontal_gradient.png": generate_gradient_image(size, horizontal=True),
        "vertical_gradient.png": generate_gradient_image(size, horizontal=False),
        "checkerboard.png": generate_checkerboard(size),
        "circles.png": generate_circle_pattern(size),
        "text.png": generate_text_image(size),
        "gaussian_noise.png": generate_noise_image(size, "gaussian"),
        "salt_pepper_noise.png": generate_noise_image(size, "salt_pepper"),
        "random_noise.png": generate_noise_image(size, "random"),
    }
    
    # Save all generated images
    for filename, img in images.items():
        filepath = output_dir / filename
        cv2.imwrite(str(filepath), img)
        print(f"Saved {filepath}")
    
    print(f"Generated {len(images)} test images in '{output_dir}'")

if __name__ == "__main__":
    main()