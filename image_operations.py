#!/usr/bin/env python3
# Basic Image Algebra Operations - Homework 1
# CSC 741-E001: Digital Image Processing

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_images(img_a_path, img_b_path=None):
    """
    Load one or two grayscale images for processing
    
    Args:
        img_a_path: Path to first image
        img_b_path: Optional path to second image
        
    Returns:
        Tuple of loaded images (img_a, img_b) or just (img_a, None)
    """
    img_a = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
    
    if img_a is None:
        print(f"Error loading image: {img_a_path}")
        return None, None
    
    img_b = None
    if img_b_path:
        img_b = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)
        if img_b is None:
            print(f"Error loading image: {img_b_path}")
            return img_a, None
            
    return img_a, img_b

def display_images(images_dict, figsize=(15, 10)):
    """
    Display multiple images using matplotlib
    
    Args:
        images_dict: Dictionary with format {title: image_array}
        figsize: Size of the figure
    """
    n = len(images_dict)
    if n == 0:
        return
        
    # Calculate rows and columns for subplot grid
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=figsize)
    
    for i, (title, img) in enumerate(images_dict.items(), 1):
        plt.subplot(rows, cols, i)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_results(results_dict, output_dir):
    """
    Save result images to the specified directory
    
    Args:
        results_dict: Dictionary with format {filename: image_array}
        output_dir: Directory to save images
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for filename, img in results_dict.items():
        filepath = output_path / filename
        cv2.imwrite(str(filepath), img)
        print(f"Saved: {filepath}")

# --- ARITHMETIC OPERATIONS ---

def addition_operations(img_a, img_b=None, k=50):
    """
    Perform addition operations on images
    
    Args:
        img_a: First input image
        img_b: Optional second image for image-to-image addition
        k: Constant value for image-to-constant addition
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    # Add constant K to image A
    results["Add_Constant"] = cv2.add(img_a, k)
    
    # Add two images if img_b is provided
    if img_b is not None:
        results["Add_Images"] = cv2.add(img_a, img_b)
    
    return results

def subtraction_operations(img_a, img_b=None, k=50):
    """
    Perform subtraction operations on images
    
    Args:
        img_a: First input image
        img_b: Optional second image for image-to-image subtraction
        k: Constant value for image-to-constant subtraction
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    # Subtract constant K from image A
    results["Subtract_Constant"] = cv2.subtract(img_a, k)
    
    # Subtract images if img_b is provided
    if img_b is not None:
        results["Subtract_Images"] = cv2.subtract(img_a, img_b)
        results["Absolute_Difference"] = cv2.absdiff(img_a, img_b)
    
    return results

def multiplication_operations(img_a, img_b=None, k=0.8):
    """
    Perform multiplication operations on images
    
    Args:
        img_a: First input image
        img_b: Optional second image for image-to-image multiplication
        k: Constant value for image-to-constant multiplication
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    # Multiply by constant K (adjusts contrast/brightness)
    result_mul_k = np.clip(img_a.astype(np.float32) * k, 0, 255).astype(np.uint8)
    results["Multiply_Constant"] = result_mul_k
    
    # Multiply by another image if provided
    if img_b is not None:
        # Convert to float, multiply, clip, convert back
        result_mul = np.clip(
            img_a.astype(np.float32) * img_b.astype(np.float32) / 255.0, 
            0, 255
        ).astype(np.uint8)
        results["Multiply_Images"] = result_mul
        
        # Create a binary mask (for demonstration)
        _, mask = cv2.threshold(img_b, 127, 255, cv2.THRESH_BINARY)
        mask_float = mask.astype(np.float32) / 255.0
        masked_img = np.clip(
            img_a.astype(np.float32) * mask_float, 
            0, 255
        ).astype(np.uint8)
        results["Masking"] = masked_img
    
    return results

def division_operations(img_a, img_b=None, k=2.0):
    """
    Perform division operations on images
    
    Args:
        img_a: First input image
        img_b: Optional second image for image-to-image division
        k: Constant value for image-to-constant division
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    # Division by constant K
    epsilon = 1e-5  # Avoid division by zero
    result_div_k = np.clip(
        img_a.astype(np.float32) / (k + epsilon), 
        0, 255
    ).astype(np.uint8)
    results["Divide_Constant"] = result_div_k
    
    # Division by another image if provided
    if img_b is not None:
        # Add epsilon to avoid division by zero
        result_div = np.clip(
            255.0 * img_a.astype(np.float32) / (img_b.astype(np.float32) + epsilon),
            0, 255
        ).astype(np.uint8)
        results["Divide_Images"] = result_div
    
    return results

def inversion_operation(img_a):
    """
    Perform image inversion (negative)
    
    Args:
        img_a: Input image
        
    Returns:
        Dictionary with result image
    """
    results = {}
    
    # Using OpenCV's bitwise_not function
    results["Negative"] = cv2.bitwise_not(img_a)
    
    return results

def nonlinear_operations(img_a):
    """
    Perform nonlinear transformations (log, exp, sqrt, trig)
    
    Args:
        img_a: Input image
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    # Log Transform (enhances dark regions)
    # Add 1 to input to avoid log(0) error
    log_input = img_a.astype(np.float32) + 1
    result_log = np.log(log_input)
    # Scale result to 0-255 range for display
    result_log_disp = cv2.normalize(result_log, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Log_Transform"] = result_log_disp

    # Exponential Transform (enhances bright regions)
    # Normalize input to [0,1] range first
    exp_input = img_a.astype(np.float32) / 255.0
    result_exp = np.exp(exp_input)
    # Scale result to 0-255 range for display
    result_exp_disp = cv2.normalize(result_exp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Exp_Transform"] = result_exp_disp

    # Square Root Transform
    sqrt_input = img_a.astype(np.float32)
    result_sqrt = np.sqrt(sqrt_input)
    # Scale result to 0-255 range for display
    result_sqrt_disp = cv2.normalize(result_sqrt, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Sqrt_Transform"] = result_sqrt_disp
    
    # Trigonometric Transforms
    # Normalize input to [0, 2π] range for periodic functions
    trig_input = (img_a.astype(np.float32) / 255.0) * 2 * np.pi
    
    # Sine Transform
    result_sin = np.sin(trig_input)
    result_sin_disp = cv2.normalize(result_sin, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Sin_Transform"] = result_sin_disp
    
    # Cosine Transform
    result_cos = np.cos(trig_input)
    result_cos_disp = cv2.normalize(result_cos, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Cos_Transform"] = result_cos_disp
    
    # Tangent Transform (with clipping to handle infinity)
    result_tan = np.clip(np.tan(trig_input), -100, 100)  # Clip extreme values
    result_tan_disp = cv2.normalize(result_tan, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    results["Tan_Transform"] = result_tan_disp
    
    return results

def scaled_operations(img_a, img_b):
    """
    Perform scaled operations (normalized domain)
    
    Args:
        img_a: First input image
        img_b: Second input image
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    if img_b is None:
        print("Second image required for scaled operations")
        return results
    
    # Normalize images to [0.0, 1.0] float range
    img_a_norm = img_a.astype(np.float32) / 255.0
    img_b_norm = img_b.astype(np.float32) / 255.0

    # Scaled Addition (Values are added in [0,1] range, then scaled back)
    result_scaled_add_norm = np.clip((img_a_norm + img_b_norm), 0.0, 1.0)
    result_scaled_add = (result_scaled_add_norm * 255.0).astype(np.uint8)
    results["Scaled_Addition"] = result_scaled_add

    # Scaled Multiplication (Values multiplied in [0,1] range, then scaled back)
    result_scaled_mul_norm = img_a_norm * img_b_norm
    result_scaled_mul = (result_scaled_mul_norm * 255.0).astype(np.uint8)
    results["Scaled_Multiplication"] = result_scaled_mul
    
    return results

def application_example(img_a, img_b):
    """
    Demonstrate application examples
    
    Args:
        img_a: First input image (e.g., current frame)
        img_b: Second input image (e.g., previous frame)
        
    Returns:
        Dictionary of result images
    """
    results = {}
    
    if img_b is None:
        print("Second image required for application examples")
        return results
    
    # Simple Motion Detection
    diff_frame = cv2.absdiff(img_a, img_b)
    results["Difference"] = diff_frame
    
    # Threshold the difference to get a binary motion mask
    threshold_value = 30
    _, motion_mask = cv2.threshold(diff_frame, threshold_value, 255, cv2.THRESH_BINARY)
    results["Motion_Mask"] = motion_mask
    
    # Demonstrate simple image blending (weighted addition)
    alpha = 0.7  # Weight for the first image
    beta = 0.3   # Weight for the second image
    blended = cv2.addWeighted(img_a, alpha, img_b, beta, 0)
    results["Blended_Images"] = blended
    
    return results

def main():
    """
    Main function to run all operations
    """
    # Get input images paths from user
    img_a_path = input("Enter path to first image: ")
    img_b_path = input("Enter path to second image (leave empty to skip): ")
    
    if not img_b_path:
        img_b_path = None
    
    # Load images
    img_a, img_b = load_images(img_a_path, img_b_path)
    if img_a is None:
        print("Could not load the first image. Exiting.")
        return
        
    # Create output directory
    output_dir = "results"
    
    # Dictionary to store all results for saving
    all_results = {}
    
    # Display input images
    display_dict = {"Image A": img_a}
    if img_b is not None:
        display_dict["Image B"] = img_b
    display_images(display_dict)
    
    # Perform and display operations
    operations = [
        # Each function returns a dictionary of results
        ("Addition", lambda: addition_operations(img_a, img_b)),
        ("Subtraction", lambda: subtraction_operations(img_a, img_b)),
        ("Multiplication", lambda: multiplication_operations(img_a, img_b)),
        ("Division", lambda: division_operations(img_a, img_b)),
        ("Inversion", lambda: inversion_operation(img_a)),
        ("Nonlinear", lambda: nonlinear_operations(img_a)),
        ("Scaled", lambda: scaled_operations(img_a, img_b) if img_b is not None else {}),
        ("Application", lambda: application_example(img_a, img_b) if img_b is not None else {})
    ]
    
    # Run each operation and display/save results
    for op_name, op_func in operations:
        print(f"Performing {op_name} operations...")
        results = op_func()
        
        if results:
            display_images(results)
            
            # Add to all_results for saving
            for name, img in results.items():
                filename = f"{op_name}_{name}.png"
                all_results[filename] = img
    
    # Ask user if they want to save results
    save_choice = input("Do you want to save all result images? (y/n): ")
    if save_choice.lower() == 'y':
        save_results(all_results, output_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()