#!/usr/bin/env python3
"""
Image preprocessing module for text extraction
"""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
import os

class ImagePreprocessor:
    """Class for preprocessing images for better text extraction."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        pass
        
    def process_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Process the image to enhance text for extraction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            Tuple containing:
                - original image
                - processed image
                - dictionary of intermediate processing steps
        """
        # Load the image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Unable to load image: {image_path}")
            
        # Store processing steps for visualization or debugging
        steps = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        steps['grayscale'] = gray
        
        # Apply adaptive histogram equalization for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        steps['enhanced'] = enhanced
        
        # Check if inversion is needed based on brightness analysis
        inverted = self._apply_smart_inversion(enhanced)
        steps['inverted'] = inverted
        
        # Apply traditional denoising
        denoised = cv2.fastNlMeansDenoising(inverted, None, 10, 7, 21)
        steps['denoised'] = denoised
        
        # Apply adaptive thresholding for binarization
        binary = cv2.adaptiveThreshold(
            denoised, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 
            2
        )
        steps['binary'] = binary
        
        # Apply morphological operations to clean up the image
        kernel = np.ones((1, 1), np.uint8)
        morphology = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        steps['morphology'] = morphology
        
        # Sharpen the image to enhance text edges
        sharpened = self._sharpen_image(morphology)
        steps['sharpened'] = sharpened
        
        # Convert back to RGB for compatibility with deep learning models
        processed_rgb = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2RGB)
        
        return original, processed_rgb, steps
        
    def _apply_smart_inversion(self, image: np.ndarray) -> np.ndarray:
        """
        Smartly determine if color inversion is needed.
        
        Args:
            image (np.ndarray): Grayscale image
            
        Returns:
            np.ndarray: Inverted image if needed, otherwise original
        """
        # Calculate average brightness
        avg_brightness = np.mean(image)
        
        # Check if the image is predominantly dark
        if avg_brightness < 127:
            # Invert colors for dark background, light text
            return cv2.bitwise_not(image)
        
        return image
        
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening to enhance text edges.
        
        Args:
            image (np.ndarray): Image to sharpen
            
        Returns:
            np.ndarray: Sharpened image
        """
        kernel = np.array([[-1, -1, -1], 
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
        
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """
        Detect and correct skew (rotation) in the image.
        
        Args:
            image (np.ndarray): Image to process
            
        Returns:
            np.ndarray: Deskewed image
        """
        # Convert to binary if not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Apply dilation to make lines more visible
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
        dilated = cv2.dilate(thresh, kernel, iterations=5)
        
        # Find all contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find largest contour
        largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)
        
        if not largest_contour:
            return image
            
        # Calculate angle
        angles = []
        for contour in largest_contour[:5]:  # Use top 5 largest contours
            rect = cv2.minAreaRect(contour)
            angle = rect[2]
            if angle < -45:
                angle = 90 + angle
            angles.append(angle)
        
        # Use median angle for robustness
        if angles:
            angle = np.median(angles)
        else:
            return image
            
        # Rotate the image to correct skew
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return rotated
        
    def remove_background(self, image: np.ndarray) -> np.ndarray:
        """
        Remove background noise from the image.
        
        Args:
            image (np.ndarray): Image to process
            
        Returns:
            np.ndarray: Image with background removed
        """
        # Convert to grayscale if not already
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use adaptive thresholding to identify text
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
        # Apply morphology to reduce noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Dilate to connect components
        dilated = cv2.dilate(opening, kernel, iterations=1)
        
        # Invert to get text in black on white background
        clean = cv2.bitwise_not(dilated)
        
        return clean
        
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast using advanced techniques.
        
        Args:
            image (np.ndarray): Image to enhance
            
        Returns:
            np.ndarray: Enhanced image
        """
        # Convert to LAB color space
        if len(image.shape) > 2:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image.copy()
            
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge channels if color image
        if len(image.shape) > 2:
            merged = cv2.merge((cl, a, b))
            enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        else:
            enhanced = cl
            
        return enhanced
