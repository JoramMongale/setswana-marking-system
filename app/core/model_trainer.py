"""
Model training module for the Setswana Marking System.
This module handles fine-tuning of the TswanaBERT model.
"""

import os
import torch
import logging
import json
import numpy as np
import pandas as pd
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SetswanaModelTrainer:
    """Class for fine-tuning the TswanaBERT model for Setswana marking."""
    
    def __init__(self, model_name="MoseliMotsoehli/TswanaBert", output_dir="./data/models"):
        """
        Initialize the model trainer.
        
        Args:
            model_name (str): Name of the pre-trained model to use
            output_dir (str): Directory to save the fine-tuned model
        """
        self.model_name = model_name
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
                logger.info(f"Created output directory: {output_dir}")
            except Exception as e:
                logger.error(f"Error creating output directory: {str(e)}")
                raise
    
    def prepare_training_data(self, data_path, test_size=0.2):
        """
        Prepare training data from a CSV or JSON file.
        
        Args:
            data_path (str): Path to data file (CSV or JSON)
            test_size (float): Proportion of data to use for testing
            
        Returns:
            tuple: Train dataset and test dataset
        """
        try:
            # Load data based on file extension
            file_ext = os.path.splitext(data_path)[1].lower()
            
            if file_ext == '.csv':
                df = pd.read_csv(data_path)
            elif file_ext == '.json':
                with open(data_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Check required columns
            required_cols = ["student_answer", "correct_answer", "similarity"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")
            
            # Split into train and test sets
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
            
            # Convert to Hugging Face datasets
            train_dataset = Dataset.from_pandas(train_df)
            test_dataset = Dataset.from_pandas(test_df)
            
            logger.info(f"Prepared training data: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
            
            return train_dataset, test_dataset
            
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}")
            raise
    
    def preprocess_function(self, examples):
        """
        Preprocess examples for training.
        
        Args:
            examples: Examples to preprocess
            
        Returns:
            dict: Preprocessed examples
        """
        student_answers = examples["student_answer"]
        correct_answers = examples["correct_answer"]
        
        # Tokenize inputs
        tokenized_inputs = self.tokenizer(
            student_answers,
            correct_answers,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Convert similarity scores to labels for regression
        # Scale to range expected by the model
        tokenized_inputs["labels"] = torch.tensor(examples["similarity"], dtype=torch.float32) / 100.0
        
        return tokenized_inputs
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for model evaluation.
        
        Args:
            eval_pred: Evaluation predictions
            
        Returns:
            dict: Metrics
        """
        predictions, labels = eval_pred
        predictions = predictions.flatten()
        
        # Scale predictions and labels back to 0-100 range
        predictions = predictions * 100.0
        labels = labels * 100.0
        
        # Calculate metrics
        mse = np.mean((predictions - labels) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - labels))
        
        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae
        }
    
    def fine_tune_model(self, train_dataset, test_dataset, output_model_name=None, epochs=3, batch_size=16):
        """
        Fine-tune the model on Setswana marking data.
        
        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            output_model_name (str, optional): Name for the fine-tuned model
            epochs (int): Number of training epochs
            batch_size (int): Training batch size
            
        Returns:
            str: Path to the saved model
        """
        try:
            # Create model name if not provided
            if output_model_name is None:
                model_base = os.path.basename(self.model_name)
                output_model_name = f"{model_base}-setswana-marker"
            
            model_output_path = os.path.join(self.output_dir, output_model_name)
            
            # Load tokenizer and model
            logger.info(f"Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Load model for regression (1 output)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=1
            )
            
            # Preprocess datasets
            train_dataset = train_dataset.map(
                self.preprocess_function,
                batched=True,
                desc="Preprocessing training data"
            )
            
            test_dataset = test_dataset.map(
                self.preprocess_function,
                batched=True,
                desc="Preprocessing test data"
            )
            
            # Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=model_output_path,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir=os.path.join(model_output_path, "logs"),
                logging_steps=100,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="rmse",
                greater_is_better=False,
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=test_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics
            )
            
            # Train the model
            logger.info("Starting model fine-tuning")
            trainer.train()
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            logger.info(f"Evaluation results: {eval_result}")
            
            # Save the model
            trainer.save_model(model_output_path)
            self.tokenizer.save_pretrained(model_output_path)
            
            logger.info(f"Model fine-tuning complete. Model saved to {model_output_path}")
            
            return model_output_path
            
        except Exception as e:
            logger.error(f"Error fine-tuning model: {str(e)}")
            raise
    
    def prepare_data_from_marking_results(self, marking_results_dir, output_path=None):
        """
        Prepare training data from marking results.
        
        Args:
            marking_results_dir (str): Directory containing marking result JSON files
            output_path (str, optional): Path to save the prepared dataset
            
        Returns:
            str: Path to the prepared dataset
        """
        try:
            if output_path is None:
                output_path = os.path.join(self.output_dir, "training_data.json")
            
            # Collect data from marking result files
            training_data = []
            
            for filename in os.listdir(marking_results_dir):
                if filename.endswith('.json'):
                    file_path = os.path.join(marking_results_dir, filename)
                    
                    with open(file_path, 'r', encoding='utf-8') as f:
                        result = json.load(f)
                    
                    # Extract question-level data
                    if "questions" in result:
                        for question_num, question_data in result["questions"].items():
                            if all(k in question_data for k in ["student_answer", "correct_answer", "similarity"]):
                                training_data.append({
                                    "student_answer": question_data["student_answer"],
                                    "correct_answer": question_data["correct_answer"],
                                    "similarity": question_data["similarity"]
                                })
            
            # Save the training data
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(training_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Prepared {len(training_data)} training examples, saved to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error preparing data from marking results: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    trainer = SetswanaModelTrainer()
    
    # Example: Create synthetic training data
    import random
    
    synthetic_data = []
    
    # Generate 100 synthetic examples with varying degrees of similarity
    for i in range(100):
        # For demonstration only - in practice, use real Setswana examples
        correct = "Diane di na le tiro ya go ruta melao le maitsholo a a siameng mo sechabeng."
        
        # Create a student answer with varying similarity
        similarity_level = random.choice(["high", "medium", "low"])
        
        if similarity_level == "high":
            # High similarity (90-100%)
            student = correct
            similarity = random.uniform(90, 100)
        elif similarity_level == "medium":
            # Medium similarity (60-89%)
            words = correct.split()
            # Remove or change some words
            for j in range(1, random.randint(1, 3)):
                idx = random.randint(0, len(words)-1)
                action = random.choice(["remove", "change"])
                if action == "remove":
                    words.pop(idx)
                else:
                    words[idx] = "batho"  # replacement word
            student = " ".join(words)
            similarity = random.uniform(60, 89)
        else:
            # Low similarity (20-59%)
            student = "Batho ba tshwanetse go tlotla bagolo."
            similarity = random.uniform(20, 59)
        
        synthetic_data.append({
            "student_answer": student,
            "correct_answer": correct,
            "similarity": similarity
        })
    
    # Save synthetic data
    output_file = os.path.join(trainer.output_dir, "synthetic_training_data.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(synthetic_data, f, ensure_ascii=False, indent=2)
    
    print(f"Created synthetic training data at: {output_file}")
    
    # Note: Running the actual fine-tuning would require sufficient data and compute resources
    # train_dataset, test_dataset = trainer.prepare_training_data(output_file)
    # model_path = trainer.fine_tune_model(train_dataset, test_dataset, epochs=1)
    # print(f"Fine-tuned model saved at: {model_path}")