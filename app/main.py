# app/main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QTextEdit
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class SetswanaMarkingSystem(QMainWindow):
    def __init__(self):
        # ... (rest of the code remains the same)
        self.model_name = "MoseliMotsoehli/TswanaBert"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        # Load TswanaBERT Model
        self.model_name = "MoseliMotsoehli/TswanaBert"
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Add title
        title = QLabel("Setswana Paper Marking System")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # Add buttons
        self.import_btn = QPushButton("Import Answer File")
        self.import_btn.clicked.connect(self.import_file)
        layout.addWidget(self.import_btn)
        
        self.memo_btn = QPushButton("Create/Edit Memo")
        self.memo_btn.clicked.connect(self.edit_memo)
        layout.addWidget(self.memo_btn)
        
        self.process_btn = QPushButton("Process Papers")
        self.process_btn.clicked.connect(self.process_papers)
        layout.addWidget(self.process_btn)
        
        self.results_btn = QPushButton("View Results")
        self.results_btn.clicked.connect(self.view_results)
        layout.addWidget(self.results_btn)
        
        # Text area for memo input
        self.memo_input = QTextEdit()
        self.memo_input.setPlaceholderText("Enter or edit the correct answers here...")
        layout.addWidget(self.memo_input)
        
        # Add status label
        self.status = QLabel("Ready")
        layout.addWidget(self.status)
        
        # Store answer file path
        self.answer_file = ""
    
    def import_file(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Answer File", "", "Text Files (*.txt);;All Files (*)", options=options)
        if file_name:
            self.answer_file = file_name
            self.status.setText(f"Loaded answer file: {file_name}")
    
    def edit_memo(self):
        self.status.setText("Edit Memo functionality coming soon")
    
    def process_papers(self):
        if not self.answer_file:
            self.status.setText("No answer file loaded!")
            return
        
        with open(self.answer_file, "r", encoding="utf-8") as f:
            student_answer = f.read().strip()
        
        correct_answer = self.memo_input.toPlainText().strip()
        
        if not correct_answer:
            self.status.setText("No correct answer provided in memo!")
            return
        
        result = self.mark_answer(student_answer, correct_answer)
        self.status.setText(f"Marking Result: {result}")
    
    def mark_answer(self, student_answer, correct_answer):
        """Compare student answer with correct answer using TswanaBERT."""
        inputs = self.tokenizer(student_answer, correct_answer, padding=True, truncation=True, return_tensors="pt")
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
        return "Correct" if prediction == 1 else "Incorrect"
    
    def view_results(self):
        self.status.setText("Results View functionality coming soon")

def main():
    app = QApplication(sys.argv)
    window = SetswanaMarkingSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
