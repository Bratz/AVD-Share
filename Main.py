#!/usr/bin/env python3
"""
Main module for Deep Text Extraction from Images
This is the entry point for the text extraction system
"""

import argparse
import os
from text_extractor import DeepTextExtractor

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Extract text from images using deep learning')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--method', type=str, default='ensemble', 
                        choices=['ensemble', 'easyocr', 'trocr', 'tesseract'],
                        help='OCR method to use')
    parser.add_argument('--output', type=str, default=None, 
                        help='Output text file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize preprocessing steps')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main function to run the text extractor."""
    args = parse_arguments()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' does not exist.")
        return 1
        
    print(f"Processing image: {args.image_path}")
    print(f"Method: {args.method}")
    
    # Initialize text extractor
    extractor = DeepTextExtractor(debug=args.debug)
    
    # Extract text
    extracted_text = extractor.extract_text(
        args.image_path, 
        method=args.method,
        visualize=args.visualize
    )
    
    # Print extracted text
    print("\nExtracted Text:")
    print("-" * 50)
    print(extracted_text)
    print("-" * 50)
    
    # Save to file if specified
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"Text saved to: {args.output}")
    
    return 0

if __name__ == "__main__":
    exit(main())
