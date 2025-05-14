# Basic Image Algebra Operations - CSC 741 Homework 1

This repository contains code to demonstrate fundamental pixel-wise arithmetic and logical image processing operations on grayscale images as part of the Digital Image Processing course homework.

## Files

- `image_operations.py`: Main script implementing all basic image algebra operations
- `generate_test_images.py`: Script to generate test images for demonstration
- `presentation_notes.md`: Detailed notes to help with the presentation

## Setup and Usage

### Prerequisites

The code requires Python with the following libraries:
- OpenCV (cv2)
- NumPy
- Matplotlib

Install them using pip:

```bash
pip install opencv-python numpy matplotlib
```

### Generating Test Images

If you don't have your own test images or want to use synthetic images for demonstration:

```bash
python generate_test_images.py
```

This will create a `test_images` directory with various grayscale test images.

### Running Image Operations

To run the image operations:

```bash
python image_operations.py
```

The script will prompt you to enter the paths to your input images. If you generated test images, you can use paths like:

- First image: `test_images/horizontal_gradient.png`
- Second image: `test_images/circles.png`

For each operation, the script will:
1. Display the resulting images
2. Ask if you want to save the results (to a `results` directory)

## Operations Implemented

1. Addition
2. Subtraction
3. Multiplication
4. Division
5. Inversion (Negative)
6. Nonlinear operations (Log, Square Root)
7. Scaled operations in normalized [0,1] domain
8. Application examples (motion detection, blending)

## Presentation

For your presentation, you can:

1. Use the generated test images
2. Run operations and save results
3. Use the saved images in your presentation slides
4. Refer to `presentation_notes.md` for key points to discuss

## Customization

Feel free to modify the code to:
- Add more operations
- Change parameters
- Use your own images
- Generate different test patterns

## Notes

- All operations handle proper data type conversion
- Results are clipped to valid grayscale range [0, 255]
- The code demonstrates both OpenCV built-in functions and manual NumPy implementations