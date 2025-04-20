# app/core/marker.py
"""
Core marking functionality for the Setswana Marking System.
This handles the comparison between student answers and memo answers.
"""

import os
import torch
import logging
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetswanaMarker:
    """Class for marking Setswana papers using TswanaBERT."""
    
    def __init__(self, model_name="MoseliMotsoehli/TswanaBert"):
        """
        Initialize the marker with the specified model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        
        try:
            logger.info(f"Loading model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # For similarity calculation, we use the base model without classification head
            self.model = AutoModel.from_pretrained(model_name)
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def calculate_similarity(self, student_answer, correct_answer):
        """
        Calculate semantic similarity between student answer and correct answer.
        
        Args:
            student_answer (str): Student's answer text
            correct_answer (str): Correct answer text from memo
            
        Returns:
            float: Similarity score (0-100)
        """
        # Clean and prepare inputs
        student_answer = student_answer.strip()
        correct_answer = correct_answer.strip()
        
        if not student_answer or not correct_answer:
            logger.warning("Empty input for similarity calculation")
            return 0.0
        
        try:
            # Encode student answer
            student_inputs = self.tokenizer(student_answer, 
                                           return_tensors="pt", 
                                           padding=True, 
                                           truncation=True, 
                                           max_length=512)
            
            with torch.no_grad():
                student_outputs = self.model(**student_inputs)
                # Get the [CLS] token embedding (first token)
                student_embedding = student_outputs.last_hidden_state[:, 0, :]
            
            # Encode correct answer
            correct_inputs = self.tokenizer(correct_answer, 
                                           return_tensors="pt", 
                                           padding=True, 
                                           truncation=True, 
                                           max_length=512)
            
            with torch.no_grad():
                correct_outputs = self.model(**correct_inputs)
                # Get the [CLS] token embedding
                correct_embedding = correct_outputs.last_hidden_state[:, 0, :]
            
            # Calculate cosine similarity
            cosine_sim = F.cosine_similarity(student_embedding, correct_embedding)
            # Convert to percentage (0-100)
            similarity = float(cosine_sim * 100)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_grade(self, similarity, grading_scale=None):
        """
        Convert similarity percentage to letter grade.
        
        Args:
            similarity (float): Similarity score (0-100)
            grading_scale (dict, optional): Custom grading scale
            
        Returns:
            str: Letter grade
        """
        if grading_scale is None:
            grading_scale = {
                90: "A",  # 90% and above
                80: "B",  # 80-89%
                70: "C",  # 70-79%
                60: "D",  # 60-69%
                0: "F"    # Below 60%
            }
        
        for threshold, grade in sorted(grading_scale.items(), reverse=True):
            if similarity >= threshold:
                return grade
                
        return "F"  # Default if no threshold matches
    
    def mark_paper(self, student_answers, memo_answers):
        """
        Mark a complete paper by comparing student answers to memo answers.
        
        Args:
            student_answers (dict): Dictionary of question numbers to student answers
            memo_answers (dict): Dictionary of question numbers to correct answers
            
        Returns:
            dict: Dictionary with marking results
        """
        results = {
            "overall_score": 0,
            "overall_grade": "",
            "questions": {}
        }
        
        # Check which questions we can mark (present in both student answers and memo)
        questions_to_mark = set(student_answers.keys()).intersection(set(memo_answers.keys()))
        
        if not questions_to_mark:
            logger.warning("No matching questions found between student answers and memo")
            return results
        
        total_similarity = 0
        
        for question_num in questions_to_mark:
            student_answer = student_answers[question_num]
            correct_answer = memo_answers[question_num]
            
            similarity = self.calculate_similarity(student_answer, correct_answer)
            grade = self.get_grade(similarity)
            
            results["questions"][question_num] = {
                "student_answer": student_answer,
                "correct_answer": correct_answer,
                "similarity": similarity,
                "grade": grade
            }
            
            total_similarity += similarity
        
        # Calculate overall score and grade
        if questions_to_mark:
            results["overall_score"] = total_similarity / len(questions_to_mark)
            results["overall_grade"] = self.get_grade(results["overall_score"])
        
        return results
    
    def fine_tune_model(self, training_data, output_dir, epochs=3):
        """
        Fine-tune the model on specific Setswana marking data.
        
        Args:
            training_data (list): List of tuples (student_answer, correct_answer, similarity_score)
            output_dir (str): Directory to save the fine-tuned model
            epochs (int): Number of training epochs
            
        Returns:
            bool: True if fine-tuning was successful
        """
        # This is a placeholder - implementing actual fine-tuning would require 
        # a more complex setup with proper dataset preparation and training loop
        logger.info("Model fine-tuning is not yet implemented")
        return False

# Example usage
if __name__ == "__main__":
    marker = SetswanaMarker()
    
    # Example: Test similarity calculation
    student_answer = "Diane di na le tiro ya go ruta melao le maitsholo a a siameng mo sechabeng."
    correct_answer = "Diane di na le tiro ya go ruta melao le maitsholo a a siameng mo sechabeng. Di dirisiwa go gakolola batho."
    
    similarity = marker.calculate_similarity(student_answer, correct_answer)
    grade = marker.get_grade(similarity)
    
    print(f"Similarity: {similarity:.2f}%")
    print(f"Grade: {grade}")