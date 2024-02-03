import cv2
from scipy.signal import medfilt
import numpy as np
import gradio as gr

# Function to apply median filter to a given column of pixels


def apply_median_filter(column, kernel_size):
    # Apply median filter with the specified kernel size
    return medfilt(column, kernel_size)

# Function to update the mask based on the indices of first occurrences of zeros and ones


def update_mask_based_on_indices(mask, first_zero, first_one, col):
    # Check if the first zero index is greater than 20
    if first_zero > 20:
        # Set the region between the first one and the first zero to 1 (sky)
        mask[first_one:first_zero, col] = 1
        # Set the region after the first zero to 0 (not sky)
        mask[first_zero:, col] = 0
        # Set the region before the first one to 0 (not sky)
        mask[:first_one, col] = 0

# Function to refine the mask by processing each column


def refine_mask(mask):
    # Iterate through each column in the mask
    for col in range(mask.shape[1]):
        # Extract the column data
        column_data = mask[:, col]
        # Apply median filter to the column data
        filtered_data = apply_median_filter(column_data, 19)

        # Find the indices of the first occurrences of zeros and ones
        zero_indices, one_indices = np.where(filtered_data == 0)[
            0], np.where(filtered_data == 1)[0]
        # Update the mask based on these indices if both are present
        if zero_indices.size and one_indices.size:
            update_mask_based_on_indices(
                mask, zero_indices[0], one_indices[0], col)
    # Return the refined mask
    return mask

# Function to preprocess the image for sky detection


def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to the grayscale image
    blurred = cv2.GaussianBlur(gray, (9, 3), 0)
    # Apply Laplacian operator to detect edges
    laplacian = cv2.Laplacian(blurred, cv2.CV_8U)
    # Create a binary mask where lower gradient values are set to 1
    return (laplacian < 6).astype(np.uint8)

# Main function to extract the sky region from an image


def extract_skyline(image):
    # Preprocess the image to get the gradient mask
    gradient_mask = preprocess_image(image)
    # Define a structuring element for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 3))
    # Erode the gradient mask to refine it
    eroded_mask = cv2.erode(gradient_mask, kernel)

    # Refine the mask further by processing each column
    skyline_mask = refine_mask(eroded_mask)
    # Apply the refined mask to the original image to extract the sky region
    return cv2.bitwise_and(image, image, mask=skyline_mask)


def process_image(input_image):
    output_image = extract_skyline(input_image)
    return output_image


iface = gr.Interface(
    fn=process_image,  # function to call
    inputs=gr.Image(),  # input type
    outputs=gr.Image(type="pil"),  # output type
    title="Sky Detection App",
    description="Upload an image to extract the sky region."
)

iface.launch(share=True)
