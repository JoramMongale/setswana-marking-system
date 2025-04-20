# app/utils/pdf_processor.py
"""
PDF processing module for the Setswana Marking System.
This module handles extracting text from scanned PDFs using OCR.
"""

import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import numpy as np
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set the path to tesseract if it's not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

class PDFProcessor:
    """Class for processing PDF files, including OCR for handwritten text."""
    
    def __init__(self, language='eng'):
        """
        Initialize the PDF processor.
        
        Args:
            language (str): Language for OCR. Use 'eng' for English or 'tsn' for Setswana 
                           if you have the Tesseract language pack installed.
        """
        self.language = language
        logger.info(f"Initialized PDF processor with language: {language}")
    
    def extract_text_from_pdf(self, pdf_path, use_ocr=True):
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            use_ocr (bool): Whether to use OCR for text extraction
            
        Returns:
            list: List of extracted text by page
        """
        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            text_by_page = []
            
            logger.info(f"Processing PDF with {len(doc)} pages")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                if use_ocr:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    
                    # Apply OCR
                    text = pytesseract.image_to_string(img, lang=self.language)
                else:
                    # Use built-in text extraction (works for digital PDFs)
                    text = page.get_text()
                
                text_by_page.append(text)
                logger.info(f"Processed page {page_num+1}/{len(doc)}")
            
            return text_by_page
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            raise
    
    def extract_student_answers(self, pdf_path, question_count=None):
        """
        Extract student answers from a PDF file and organize by question.
        
        Args:
            pdf_path (str): Path to the PDF file
            question_count (int, optional): Expected number of questions
            
        Returns:
            dict: Dictionary of question numbers to answers
        """
        text_by_page = self.extract_text_from_pdf(pdf_path)
        combined_text = "\n".join(text_by_page)
        
        # Simple approach: Split by question numbers
        # This is a basic implementation and may need refinement based on actual PDF formats
        answers = {}
        current_question = None
        current_answer = []
        
        for line in combined_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new question (simple heuristic)
            # Adjust the regex pattern based on your question numbering format
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                # Save the previous question's answer if there was one
                if current_question is not None:
                    answers[current_question] = '\n'.join(current_answer)
                
                # Parse question number
                parts = line.split('.', 1)
                current_question = int(parts[0])
                
                # Start collecting the new answer
                if len(parts) > 1 and parts[1].strip():
                    current_answer = [parts[1].strip()]
                else:
                    current_answer = []
            elif current_question is not None:
                current_answer.append(line)
        
        # Save the last question's answer
        if current_question is not None and current_answer:
            answers[current_question] = '\n'.join(current_answer)
            
        # Validate if we got the expected number of questions
        if question_count and len(answers) != question_count:
            logger.warning(f"Expected {question_count} questions but found {len(answers)}")
            
        return answers
    
    def extract_memo_answers(self, pdf_path, question_count=None):
        """
        Extract memo (correct answers) from a PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            question_count (int, optional): Expected number of questions
            
        Returns:
            dict: Dictionary of question numbers to correct answers
        """
        # This implementation is similar to extract_student_answers
        # In a real system, you might want to add specific processing for memo formats
        return self.extract_student_answers(pdf_path, question_count)
    
    def preprocess_image(self, img):
        """
        Preprocess an image to improve OCR accuracy.
        
        Args:
            img (PIL.Image): Input image
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Convert to grayscale
        img = img.convert('L')
        
        # Increase contrast
        img = Image.fromarray(np.array(img) * 1.5)
        
        # Binarize the image (convert to black and white)
        threshold = 150  # Adjust this value as needed
        img = img.point(lambda p: p > threshold and 255)
        
        return img
    
    def ocr_with_preprocessing(self, img):
        """
        Apply OCR with preprocessing for better results.
        
        Args:
            img (PIL.Image): Input image
            
        Returns:
            str: Extracted text
        """
        processed_img = self.preprocess_image(img)
        text = pytesseract.image_to_string(processed_img, lang=self.language)
        return text

# Example usage
if __name__ == "__main__":
    processor = PDFProcessor(language='eng')  # or 'tsn' if you have the Setswana language pack
    
    # Example: Process a student answer sheet
    try:
        # This is just an example - replace with your actual paths
        student_pdf = "../data/samples/student_answer.pdf"
        memo_pdf = "../data/samples/memo.pdf"
        
        student_answers = processor.extract_student_answers(student_pdf)
        memo_answers = processor.extract_memo_answers(memo_pdf)
        
        print(f"Extracted {len(student_answers)} answers from student PDF")
        print(f"Extracted {len(memo_answers)} answers from memo PDF")
        
    except Exception as e:
        print(f"Error: {str(e)}")