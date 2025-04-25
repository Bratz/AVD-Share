#!/usr/bin/env python3
"""
Core text extraction module using deep learning techniques
"""

import cv2
import numpy as np
import torch
import os
from PIL import Image
import torchvision.transforms as T
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import easyocr
import pytesseract
from typing import List, Tuple, Optional, Union, Dict, Any

# Import custom modules
from preprocessing import ImagePreprocessor
from text_detection import TextDetector
from utils import visualization

class DeepTextExtractor:
    """Main class for extracting text from images using deep learning techniques."""
    
    def __init__(self, debug: bool = False):
        """
        Initialize the text extractor with necessary models.
        
        Args:
            debug (bool): Enable debug mode with verbose logging
        """
        self.debug = debug
        
        if self.debug:
            print("Initializing Deep Text Extractor...")
        
        # Initialize preprocessing module
        self.preprocessor = ImagePreprocessor()
        
        # Initialize text detector
        self.text_detector = TextDetector()
        
        # Initialize EasyOCR reader
        if self.debug:
            print("Loading EasyOCR model...")
        self.reader = easyocr.Reader(['en'])
        
        # Initialize TrOCR model
        if self.debug:
            print("Loading TrOCR model...")
        try:
            self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            
            # Check if GPU is available
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.debug:
                print(f"Using device: {self.device}")
            self.trocr_model.to(self.device)
            self.has_trocr = True
        except Exception as e:
            if self.debug:
                print(f"Failed to load TrOCR model: {e}")
            self.has_trocr = False
        
        # Initialize denoising model if available
        try:
            self.denoising_model = torch.hub.load('pytorch/vision', 'denoising_model', pretrained=True)
            self.denoising_model.to(self.device)
            self.has_denoiser = True
            if self.debug:
                print("Denoising model loaded successfully")
        except Exception as e:
            if self.debug:
                print(f"Failed to load denoising model: {e}")
            self.has_denoiser = False

    def extract_with_easyocr(self, image: np.ndarray) -> str:
        """
        Extract text using EasyOCR deep learning model.
        
        Args:
            image (np.ndarray): Image to extract text from
            
        Returns:
            str: Extracted text
        """
        if image is None or image.size == 0:
            return ""
            
        try:
            results = self.reader.readtext(image)
            text = " ".join([result[1] for result in results])
            return text
        except Exception as e:
            if self.debug:
                print(f"EasyOCR error: {e}")
            return ""

    def extract_with_trocr(self, image: Union[np.ndarray, Image.Image]) -> str:
        """
        Extract text using TrOCR transformer model.
        
        Args:
            image (Union[np.ndarray, Image.Image]): Image to extract text from
            
        Returns:
            str: Extracted text
        """
        if not self.has_trocr:
            if self.debug:
                print("TrOCR model not available, skipping")
            return ""
            
        try:
            if isinstance(image, np.ndarray):
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = image
                
            # Preprocess the image
            pixel_values = self.trocr_processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            # Generate text
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            if self.debug:
                print(f"TrOCR error: {e}")
            return ""

    def extract_with_tesseract(self, image: np.ndarray) -> str:
        """
        Extract text using Tesseract OCR.
        
        Args:
            image (np.ndarray): Image to extract text from
            
        Returns:
            str: Extracted text
        """
        try:
            # Apply custom configuration to improve accuracy
            custom_config = r'--oem 3 --psm 6'
            text = pytesseract.image_to_string(image, config=custom_config)
            return text
        except Exception as e:
            if self.debug:
                print(f"Tesseract error: {e}")
            return ""

    def extract_text(self, image_path: str, method: str = 'ensemble', visualize: bool = False) -> str:
        """
        Main method to extract text from an image.
        
        Args:
            image_path (str): Path to the image file
            method (str): OCR method to use ('ensemble', 'easyocr', 'trocr', 'tesseract')
            visualize (bool): Whether to visualize preprocessing steps
            
        Returns:
            str: Extracted text
        """
        # Load and preprocess the image
        original, processed_image, steps = self.preprocessor.process_image(image_path)
        
        if visualize:
            visualization.visualize_steps(original, steps)
        
        # Apply deep denoising if available
        if self.has_denoiser:
            processed_image = self.apply_deep_denoising(processed_image)
            
            if visualize:
                cv2.imshow("Deep Denoised", processed_image)
                cv2.waitKey(0)
        
        # Detect text regions
        try:
            text_regions = self.text_detector.detect_text_regions(processed_image)
            
            if visualize:
                visualization.visualize_text_regions(original, text_regions)
                
            if not text_regions:
                # If no regions detected, use the whole image
                text_regions = [processed_image]
        except Exception as e:
            if self.debug:
                print(f"Text region detection failed: {e}")
            text_regions = [processed_image]  # Use full image if detection fails
        
        # Extract text based on selected method
        all_text = ""
        
        if method == 'easyocr':
            for region in text_regions:
                text = self.extract_with_easyocr(region)
                if text:
                    all_text += text + " "
                    
        elif method == 'trocr':
            for region in text_regions:
                text = self.extract_with_trocr(region)
                if text:
                    all_text += text + " "
                    
        elif method == 'tesseract':
            for region in text_regions:
                text = self.extract_with_tesseract(region)
                if text:
                    all_text += text + " "
                    
        elif method == 'ensemble':
            # Use multiple models and combine results
            for region in text_regions:
                easyocr_text = self.extract_with_easyocr(region)
                trocr_text = self.extract_with_trocr(region)
                tesseract_text = self.extract_with_tesseract(region)
                
                # Simple ensemble: choose the best result based on length and confidence
                texts = [t for t in [easyocr_text, trocr_text, tesseract_text] if t]
                if texts:
                    # Choose the longest non-empty text as it likely captured more
                    best_text = max(texts, key=len)
                    all_text += best_text + " "
        
        return all_text.strip()
        
    def apply_deep_denoising(self, image: np.ndarray) -> np.ndarray:
        """
        Apply deep learning based denoising to the image.
        
        Args:
            image (np.ndarray): Image to denoise
            
        Returns:
            np.ndarray: Denoised image
        """
        if not self.has_denoiser:
            return image
            
        try:
            # Convert to tensor
            transform = T.Compose([
                T.ToTensor(),
            ])
            
            # Ensure image is RGB
            if len(image.shape) == 2:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image_rgb = image
                
            image_tensor = transform(image_rgb).to(self.device)
            
            # Apply denoising
            with torch.no_grad():
                denoised_tensor = self.denoising_model(image_tensor.unsqueeze(0)).squeeze(0)
                
            # Convert back to numpy
            denoised_image = (denoised_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            
            return denoised_image
        except Exception as e:
            if self.debug:
                print(f"Deep denoising failed: {e}")
            return image
