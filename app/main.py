import sys
import os
import logging
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, 
    QFileDialog, QTextEdit, QTabWidget, QHBoxLayout, QProgressBar, QMessageBox,
    QListWidget, QGroupBox, QFormLayout, QLineEdit, QCheckBox, QComboBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Import the modules we've created
from app.core.marker import SetswanaMarker
from app.utils.pdf_processor import PDFProcessor
from app.utils.report_generator import ReportGenerator
from app.core.model_trainer import SetswanaModelTrainer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WorkerThread(QThread):
    """Worker thread for background processing to keep UI responsive."""
    update_progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, task_type, **kwargs):
        super().__init__()
        self.task_type = task_type
        self.kwargs = kwargs
    
    def run(self):
        try:
            self.update_progress.emit(10)
            
            if self.task_type == "process_pdf":
                pdf_processor = PDFProcessor(language=self.kwargs.get("language", "eng"))
                self.update_progress.emit(20)
                
                pdf_path = self.kwargs.get("pdf_path")
                use_ocr = self.kwargs.get("use_ocr", True)
                
                self.update_progress.emit(30)
                result = pdf_processor.extract_text_from_pdf(pdf_path, use_ocr)
                self.update_progress.emit(90)
                
                self.finished.emit({"text_by_page": result})
                
            elif self.task_type == "mark_paper":
                marker = SetswanaMarker()
                self.update_progress.emit(20)
                
                student_answers = self.kwargs.get("student_answers", {})
                memo_answers = self.kwargs.get("memo_answers", {})
                
                self.update_progress.emit(40)
                result = marker.mark_paper(student_answers, memo_answers)
                self.update_progress.emit(90)
                
                self.finished.emit({"marking_results": result})
                
            elif self.task_type == "generate_report":
                report_gen = ReportGenerator()
                self.update_progress.emit(30)
                
                student_info = self.kwargs.get("student_info", {})
                marking_results = self.kwargs.get("marking_results", {})
                
                self.update_progress.emit(50)
                report_path = report_gen.generate_individual_report(student_info, marking_results)
                self.update_progress.emit(90)
                
                self.finished.emit({"report_path": report_path})
                
            elif self.task_type == "annotate_pdf":
                report_gen = ReportGenerator()
                self.update_progress.emit(30)
                
                original_pdf = self.kwargs.get("original_pdf", "")
                marking_results = self.kwargs.get("marking_results", {})
                
                self.update_progress.emit(50)
                pdf_path = report_gen.create_annotated_pdf(original_pdf, marking_results)
                self.update_progress.emit(90)
                
                self.finished.emit({"annotated_pdf": pdf_path})
                
            else:
                self.error.emit(f"Unknown task type: {self.task_type}")
                
            self.update_progress.emit(100)
            
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")
            self.error.emit(str(e))

class SetswanaMarkingSystem(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Setswana Marking System")
        self.resize(900, 700)
        
        # Initialize components
        self.init_ui()
        
        # Store paths and data
        self.student_pdf = ""
        self.memo_pdf = ""
        self.student_answers = {}
        self.memo_answers = {}
        self.marking_results = {}
        self.current_student_info = {
            "id": "",
            "name": "",
            "date": ""
        }
        
        # Initialize system components
        self.pdf_processor = PDFProcessor(language="eng")  # Default to English, can be changed in settings
        self.marker = SetswanaMarker()
        self.report_generator = ReportGenerator()
        
        logger.info("Setswana Marking System initialized")
    
    def init_ui(self):
        """Initialize the user interface."""
        # Create central widget with tab layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        main_layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.add_marking_tab()
        self.add_batch_tab()
        self.add_reports_tab()
        self.add_training_tab()
        self.add_settings_tab()
        
        # Add status bar with progress
        status_layout = QHBoxLayout()
        
        self.status_label = QLabel("Ready")
        status_layout.addWidget(self.status_label, 7)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        status_layout.addWidget(self.progress_bar, 3)
        
        main_layout.addLayout(status_layout)
    
    def add_marking_tab(self):
        """Add the main marking tab."""
        marking_tab = QWidget()
        layout = QVBoxLayout(marking_tab)
        
        # Title
        title = QLabel("Setswana Paper Marking")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # File input section
        file_group = QGroupBox("Input Files")
        file_layout = QFormLayout()
        
        self.student_file_btn = QPushButton("Browse...")
        self.student_file_btn.clicked.connect(self.select_student_file)
        self.student_file_label = QLabel("No file selected")
        file_layout.addRow("Student Answer Sheet:", self.student_file_btn)
        file_layout.addRow("", self.student_file_label)
        
        self.memo_file_btn = QPushButton("Browse...")
        self.memo_file_btn.clicked.connect(self.select_memo_file)
        self.memo_file_label = QLabel("No file selected")
        file_layout.addRow("Marking Memo:", self.memo_file_btn)
        file_layout.addRow("", self.memo_file_label)
        
        self.ocr_checkbox = QCheckBox("Use OCR for scanned documents")
        self.ocr_checkbox.setChecked(True)
        file_layout.addRow("", self.ocr_checkbox)
        
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)
        
        # Student info section
        student_group = QGroupBox("Student Information")
        student_layout = QFormLayout()
        
        self.student_id_input = QLineEdit()
        student_layout.addRow("Student ID:", self.student_id_input)
        
        self.student_name_input = QLineEdit()
        student_layout.addRow("Student Name:", self.student_name_input)
        
        student_group.setLayout(student_layout)
        layout.addWidget(student_group)
        
        # Processing buttons
        button_layout = QHBoxLayout()
        
        self.process_btn = QPushButton("Process Answer Sheet")
        self.process_btn.clicked.connect(self.process_paper)
        button_layout.addWidget(self.process_btn)
        
        self.view_results_btn = QPushButton("View Results")
        self.view_results_btn.clicked.connect(self.view_marking_results)
        self.view_results_btn.setEnabled(False)
        button_layout.addWidget(self.view_results_btn)
        
        self.generate_report_btn = QPushButton("Generate Report")
        self.generate_report_btn.clicked.connect(self.generate_student_report)
        self.generate_report_btn.setEnabled(False)
        button_layout.addWidget(self.generate_report_btn)
        
        layout.addLayout(button_layout)
        
        # Add to tabs
        self.tabs.addTab(marking_tab, "Marking")
    
    def add_batch_tab(self):
        """Add the batch processing tab."""
        batch_tab = QWidget()
        layout = QVBoxLayout(batch_tab)
        
        # Title
        title = QLabel("Batch Processing")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # File selection
        batch_group = QGroupBox("Batch Files")
        batch_layout = QFormLayout()
        
        self.batch_dir_btn = QPushButton("Select Folder...")
        self.batch_dir_btn.clicked.connect(self.select_batch_directory)
        self.batch_dir_label = QLabel("No directory selected")
        batch_layout.addRow("Student Papers Directory:", self.batch_dir_btn)
        batch_layout.addRow("", self.batch_dir_label)
        
        self.memo_batch_btn = QPushButton("Select Memo...")
        self.memo_batch_btn.clicked.connect(self.select_batch_memo)
        self.memo_batch_label = QLabel("No memo selected")
        batch_layout.addRow("Common Marking Memo:", self.memo_batch_btn)
        batch_layout.addRow("", self.memo_batch_label)
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        # File list
        list_group = QGroupBox("Files to Process")
        list_layout = QVBoxLayout()
        
        self.file_list = QListWidget()
        list_layout.addWidget(self.file_list)
        
        list_group.setLayout(list_layout)
        layout.addWidget(list_group)
        
        # Processing buttons
        button_layout = QHBoxLayout()
        
        self.process_batch_btn = QPushButton("Process Batch")
        self.process_batch_btn.clicked.connect(self.process_batch)
        button_layout.addWidget(self.process_batch_btn)
        
        self.generate_class_report_btn = QPushButton("Generate Class Report")
        self.generate_class_report_btn.clicked.connect(self.generate_class_report)
        self.generate_class_report_btn.setEnabled(False)
        button_layout.addWidget(self.generate_class_report_btn)
        
        layout.addLayout(button_layout)
        
        # Add to tabs
        self.tabs.addTab(batch_tab, "Batch Processing")
    
    def add_reports_tab(self):
        """Add the reports tab."""
        reports_tab = QWidget()
        layout = QVBoxLayout(reports_tab)
        
        # Title
        title = QLabel("Reports")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
      # app/main.py (continued)

        # Reports list
        reports_group = QGroupBox("Generated Reports")
        reports_layout = QVBoxLayout()
        
        self.reports_list = QListWidget()
        self.reports_list.itemDoubleClicked.connect(self.open_report)
        reports_layout.addWidget(self.reports_list)
        
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_reports)
        reports_layout.addWidget(refresh_btn)
        
        reports_group.setLayout(reports_layout)
        layout.addWidget(reports_group)
        
        # Add to tabs
        self.tabs.addTab(reports_tab, "Reports")
    
    def add_training_tab(self):
        """Add the model training tab."""
        training_tab = QWidget()
        layout = QVBoxLayout(training_tab)
        
        # Title
        title = QLabel("Model Training")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # Training data section
        data_group = QGroupBox("Training Data")
        data_layout = QVBoxLayout()
        
        # Options for data source
        data_source_layout = QFormLayout()
        self.data_source_combo = QComboBox()
        self.data_source_combo.addItems(["Previous Marking Results", "Custom Dataset"])
        data_source_layout.addRow("Data Source:", self.data_source_combo)
        data_layout.addLayout(data_source_layout)
        
        # Data selection
        self.training_data_btn = QPushButton("Select Training Data...")
        self.training_data_btn.clicked.connect(self.select_training_data)
        self.training_data_label = QLabel("No data selected")
        data_layout.addWidget(self.training_data_btn)
        data_layout.addWidget(self.training_data_label)
        
        data_group.setLayout(data_layout)
        layout.addWidget(data_group)
        
        # Training parameters
        params_group = QGroupBox("Training Parameters")
        params_layout = QFormLayout()
        
        self.epochs_input = QLineEdit("3")
        params_layout.addRow("Epochs:", self.epochs_input)
        
        self.batch_size_input = QLineEdit("16")
        params_layout.addRow("Batch Size:", self.batch_size_input)
        
        self.output_model_input = QLineEdit("TswanaBert-setswana-marker")
        params_layout.addRow("Output Model Name:", self.output_model_input)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)
        
        # Training buttons
        button_layout = QHBoxLayout()
        
        self.prepare_data_btn = QPushButton("Prepare Training Data")
        self.prepare_data_btn.clicked.connect(self.prepare_training_data)
        button_layout.addWidget(self.prepare_data_btn)
        
        self.train_model_btn = QPushButton("Train Model")
        self.train_model_btn.clicked.connect(self.train_model)
        button_layout.addWidget(self.train_model_btn)
        
        layout.addLayout(button_layout)
        
        # Add to tabs
        self.tabs.addTab(training_tab, "Model Training")
    
    def add_settings_tab(self):
        """Add the settings tab."""
        settings_tab = QWidget()
        layout = QVBoxLayout(settings_tab)
        
        # Title
        title = QLabel("Settings")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title)
        
        # OCR settings
        ocr_group = QGroupBox("OCR Settings")
        ocr_layout = QFormLayout()
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English (eng)", "Setswana (tsn)"])
        self.language_combo.currentIndexChanged.connect(self.update_ocr_language)
        ocr_layout.addRow("OCR Language:", self.language_combo)
        
        ocr_group.setLayout(ocr_layout)
        layout.addWidget(ocr_group)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QFormLayout()
        
        self.model_path_input = QLineEdit("MoseliMotsoehli/TswanaBert")
        model_layout.addRow("Model Path:", self.model_path_input)
        
        self.use_custom_model = QCheckBox("Use Custom Fine-tuned Model")
        self.use_custom_model.stateChanged.connect(self.toggle_custom_model)
        model_layout.addRow("", self.use_custom_model)
        
        self.custom_model_btn = QPushButton("Browse...")
        self.custom_model_btn.clicked.connect(self.select_custom_model)
        self.custom_model_btn.setEnabled(False)
        model_layout.addRow("Custom Model:", self.custom_model_btn)
        
        self.custom_model_label = QLabel("No custom model selected")
        self.custom_model_label.setEnabled(False)
        model_layout.addRow("", self.custom_model_label)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Path settings
        path_group = QGroupBox("File Paths")
        path_layout = QFormLayout()
        
        self.results_dir_input = QLineEdit("./data/results")
        path_layout.addRow("Results Directory:", self.results_dir_input)
        
        self.models_dir_input = QLineEdit("./data/models")
        path_layout.addRow("Models Directory:", self.models_dir_input)
        
        path_group.setLayout(path_layout)
        layout.addWidget(path_group)
        
        # Save button
        save_btn = QPushButton("Save Settings")
        save_btn.clicked.connect(self.save_settings)
        layout.addWidget(save_btn)
        
        # Add to tabs
        self.tabs.addTab(settings_tab, "Settings")
    
    def select_student_file(self):
        """Open file dialog to select student answer PDF."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Student Answer File", "", 
            "PDF Files (*.pdf);;Text Files (*.txt);;All Files (*)", 
            options=options
        )
        
        if file_name:
            self.student_pdf = file_name
            self.student_file_label.setText(os.path.basename(file_name))
            self.status_label.setText(f"Loaded student file: {os.path.basename(file_name)}")
    
    def select_memo_file(self):
        """Open file dialog to select memo PDF."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Memo File", "", 
            "PDF Files (*.pdf);;Text Files (*.txt);;All Files (*)", 
            options=options
        )
        
        if file_name:
            self.memo_pdf = file_name
            self.memo_file_label.setText(os.path.basename(file_name))
            self.status_label.setText(f"Loaded memo file: {os.path.basename(file_name)}")
    
    def process_paper(self):
        """Process the student paper and memo."""
        if not self.student_pdf:
            self.show_error("No student file selected!")
            return
            
        if not self.memo_pdf:
            self.show_error("No memo file selected!")
            return
        
        # Get student info
        self.current_student_info["id"] = self.student_id_input.text()
        self.current_student_info["name"] = self.student_name_input.text()
        
        # Start processing
        self.status_label.setText("Processing student paper...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Process student answers
        self.worker = WorkerThread(
            "process_pdf",
            pdf_path=self.student_pdf,
            use_ocr=self.ocr_checkbox.isChecked(),
            language=self.get_ocr_language()
        )
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_student_pdf_processed)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def on_student_pdf_processed(self, result):
        """Handle the processed student PDF."""
        text_by_page = result.get("text_by_page", [])
        combined_text = "\n\n".join(text_by_page)
        
        # Extract student answers
        self.student_answers = self.extract_answers(combined_text)
        
        # Now process memo
        self.status_label.setText("Processing memo...")
        self.progress_bar.setValue(0)
        
        self.worker = WorkerThread(
            "process_pdf",
            pdf_path=self.memo_pdf,
            use_ocr=self.ocr_checkbox.isChecked(),
            language=self.get_ocr_language()
        )
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_memo_pdf_processed)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def on_memo_pdf_processed(self, result):
        """Handle the processed memo PDF."""
        text_by_page = result.get("text_by_page", [])
        combined_text = "\n\n".join(text_by_page)
        
        # Extract memo answers
        self.memo_answers = self.extract_answers(combined_text)
        
        # Now mark the paper
        self.status_label.setText("Marking paper...")
        self.progress_bar.setValue(0)
        
        self.worker = WorkerThread(
            "mark_paper",
            student_answers=self.student_answers,
            memo_answers=self.memo_answers
        )
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_paper_marked)
        self.worker.error.connect(self.show_error)
        self.worker.start()
    
    def on_paper_marked(self, result):
        """Handle the marked paper results."""
        self.marking_results = result.get("marking_results", {})
        
        # Display overall score
        overall_score = self.marking_results.get("overall_score", 0)
        overall_grade = self.marking_results.get("overall_grade", "")
        
        self.status_label.setText(f"Marking complete. Overall Score: {overall_score:.1f}% - Grade: {overall_grade}")
        self.progress_bar.setVisible(False)
        
        # Enable buttons
        self.view_results_btn.setEnabled(True)
        self.generate_report_btn.setEnabled(True)
        
        # Show results
        self.view_marking_results()
    
    def view_marking_results(self):
        """Display marking results."""
        if not self.marking_results:
            self.show_error("No marking results available!")
            return
        
        # Create a message box with results
        results_dialog = QMessageBox()
        results_dialog.setWindowTitle("Marking Results")
        
        # Add overall score and grade
        overall_score = self.marking_results.get("overall_score", 0)
        overall_grade = self.marking_results.get("overall_grade", "")
        
        results_text = f"Overall Score: {overall_score:.1f}% - Grade: {overall_grade}\n\n"
        results_text += "Question Details:\n\n"
        
        # Add question details
        questions = self.marking_results.get("questions", {})
        
        for question_num, question_data in sorted(questions.items()):
            similarity = question_data.get("similarity", 0)
            grade = question_data.get("grade", "")
            
            results_text += f"Question {question_num}:\n"
            results_text += f"Score: {similarity:.1f}% - Grade: {grade}\n"
            results_text += f"Student Answer: {question_data.get('student_answer', '')[:100]}...\n"
            results_text += f"Correct Answer: {question_data.get('correct_answer', '')[:100]}...\n\n"
        
        results_dialog.setText(results_text)
        results_dialog.exec_()
    
    def generate_student_report(self):
        """Generate a detailed report for the student."""
        if not self.marking_results:
            self.show_error("No marking results available!")
            return
            
        # Start report generation
        self.status_label.setText("Generating student report...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        
        self.worker = WorkerThread(
            "generate_report",
            student_info=self.current_student_info,
            marking_results=self.marking_results
        )
        self.worker.update_progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_report_generated)
        self.worker.error.connect(self.show_error)
        self.worker.start()
        
        # Also generate annotated PDF if available
        if self.student_pdf:
            self.worker = WorkerThread(
                "annotate_pdf",
                original_pdf=self.student_pdf,
                marking_results=self.marking_results
            )
            self.worker.update_progress.connect(self.update_progress)
            self.worker.finished.connect(self.on_pdf_annotated)
            self.worker.error.connect(self.show_error)
            self.worker.start()
    
    def on_report_generated(self, result):
        """Handle the generated report."""
        report_path = result.get("report_path", "")
        
        if report_path:
            self.status_label.setText(f"Report generated: {os.path.basename(report_path)}")
            self.progress_bar.setVisible(False)
            
            # Update reports list
            self.refresh_reports()
            
            # Ask if user wants to open the report
            reply = QMessageBox.question(
                self, "Report Generated", 
                f"Report generated successfully at {report_path}. Open now?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                self.open_file(report_path)
        else:
            self.status_label.setText("Error generating report")
            self.progress_bar.setVisible(False)
    
    def on_pdf_annotated(self, result):
        """Handle the annotated PDF."""
        pdf_path = result.get("annotated_pdf", "")
        
        if pdf_path:
            # Just update the status, we already have a dialog open for the main report
            self.status_label.setText(f"Annotated PDF generated: {os.path.basename(pdf_path)}")
    
    def select_batch_directory(self):
        """Open directory dialog to select batch processing directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Directory with Student Papers", ""
        )
        
        if directory:
            self.batch_dir_label.setText(directory)
            
            # Populate file list
            self.file_list.clear()
            pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]
            
            for pdf_file in pdf_files:
                self.file_list.addItem(pdf_file)
            
            self.status_label.setText(f"Found {len(pdf_files)} PDF files in {directory}")
    
    def select_batch_memo(self):
        """Open file dialog to select batch memo PDF."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Memo File", "", 
            "PDF Files (*.pdf);;Text Files (*.txt);;All Files (*)", 
            options=options
        )
        
        if file_name:
            self.memo_batch_label.setText(os.path.basename(file_name))
            self.status_label.setText(f"Loaded batch memo file: {os.path.basename(file_name)}")
    
    def process_batch(self):
        """Process a batch of papers."""
        # This would need more complex implementation to handle multiple files
        self.status_label.setText("Batch processing not yet implemented")
        
        # Future implementation would:
        # 1. Get the memo file and extract answers
        # 2. Loop through each PDF in the selected directory
        # 3. Process each one and store results
        # 4. Generate individual reports for each
        # 5. Finally enable the class report button
        
        # Enable class report for demo purposes
        self.generate_class_report_btn.setEnabled(True)
    
    def generate_class_report(self):
        """Generate a class-level report."""
        # This would need data from batch processing
        self.status_label.setText("Class report generation not yet implemented")
        
        # Future implementation would:
        # 1. Collect all the marking results from batch processing
        # 2. Generate summary statistics
        # 3. Create Excel and PDF reports
    
    def refresh_reports(self):
        """Refresh the list of available reports."""
        self.reports_list.clear()
        
        results_dir = self.results_dir_input.text()
        if not os.path.exists(results_dir):
            return
            
        # Find PDF and Excel files in the results directory
        report_files = []
        for f in os.listdir(results_dir):
            if f.endswith('.pdf') or f.endswith('.xlsx'):
                report_files.append(f)
        
        # Add to list
        for report_file in sorted(report_files, reverse=True):
            self.reports_list.addItem(report_file)
    
    def open_report(self, item):
        """Open a selected report."""
        report_name = item.text()
        report_path = os.path.join(self.results_dir_input.text(), report_name)
        
        if os.path.exists(report_path):
            self.open_file(report_path)
        else:
            self.show_error(f"Report file not found: {report_path}")
    
    def select_training_data(self):
        """Select training data for model fine-tuning."""
        source = self.data_source_combo.currentText()
        
        if source == "Previous Marking Results":
            # Select directory with results
            directory = QFileDialog.getExistingDirectory(
                self, "Select Directory with Marking Results", ""
            )
            
            if directory:
                self.training_data_label.setText(directory)
        else:
            # Select custom dataset file
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(
                self, "Select Training Data File", "", 
                "JSON Files (*.json);;CSV Files (*.csv);;All Files (*)", 
                options=options
            )
            
            if file_name:
                self.training_data_label.setText(file_name)
    
    def prepare_training_data(self):
        """Prepare training data from selected source."""
        data_path = self.training_data_label.text()
        
        if not data_path or not os.path.exists(data_path):
            self.show_error("No valid training data selected!")
            return
        
        # This would need implementation to extract training samples
        self.status_label.setText("Training data preparation not yet implemented")
        
        # Future implementation would:
        # 1. If directory, find all json results and extract question/answer pairs
        # 2. If file, verify format
        # 3. Save processed data to training file
    
    def train_model(self):
        """Train the model with selected data."""
        # This would need implementation for actual model training
        self.status_label.setText("Model training not yet implemented")
        
        # Future implementation would:
        # 1. Load prepared training data
        # 2. Set up training parameters
        # 3. Launch training in background thread
        # 4. Save resulting model
    
    def update_ocr_language(self):
        """Update OCR language setting."""
        language = self.get_ocr_language()
        self.pdf_processor = PDFProcessor(language=language)
        self.status_label.setText(f"OCR language set to: {language}")
    
    def get_ocr_language(self):
        """Get selected OCR language code."""
        selected = self.language_combo.currentText()
        
        if "Setswana" in selected:
            return "tsn"
        else:
            return "eng"
    
    def toggle_custom_model(self, state):
        """Toggle custom model selection."""
        enabled = state == Qt.Checked
        self.custom_model_btn.setEnabled(enabled)
        self.custom_model_label.setEnabled(enabled)
    
    def select_custom_model(self):
        """Select custom model directory."""
        directory = QFileDialog.getExistingDirectory(
            self, "Select Custom Model Directory", ""
        )
        
        if directory:
            self.custom_model_label.setText(directory)
    
    def save_settings(self):
        """Save current settings."""
        # This would typically save to a config file
        self.status_label.setText("Settings saved")
        
        # Update components with new settings
        results_dir = self.results_dir_input.text()
        self.report_generator = ReportGenerator(output_directory=results_dir)
        
        # If using custom model, update marker
        if self.use_custom_model.isChecked() and os.path.exists(self.custom_model_label.text()):
            # This would reload the marker with custom model
            pass
        else:
            model_path = self.model_path_input.text()
            self.marker = SetswanaMarker(model_name=model_path)
    
    def extract_answers(self, text):
        """
        Extract answers from text.
        This is a simple implementation and would need refinement.
        
        Args:
            text (str): Text to extract answers from
            
        Returns:
            dict: Dictionary of question numbers to answers
        """
        answers = {}
        current_question = None
        current_answer = []
        
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if this line starts a new question (simple heuristic)
            # Adjust the regex pattern based on your question numbering format
            if line.startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.')):
                # Save the previous question's answer if there was one
                if current_question is not None and current_answer:
                    answers[current_question] = '\n'.join(current_answer)
                
                # Parse question number
                parts = line.split('.', 1)
                try:
                    current_question = int(parts[0])
                    
                    # Start collecting the new answer
                    if len(parts) > 1 and parts[1].strip():
                        current_answer = [parts[1].strip()]
                    else:
                        current_answer = []
                except ValueError:
                    # Not a question number after all
                    if current_question is not None:
                        current_answer.append(line)
            elif current_question is not None:
                current_answer.append(line)
        
        # Save the last question's answer
        if current_question is not None and current_answer:
            answers[current_question] = '\n'.join(current_answer)
            
        return answers
    
    def update_progress(self, value):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def show_error(self, message):
        """Show error message."""
        QMessageBox.critical(self, "Error", message)
        self.status_label.setText(f"Error: {message}")
        self.progress_bar.setVisible(False)
    
    def open_file(self, path):
        """Open a file with the default system application."""
        import subprocess
        import platform
        
        try:
            if platform.system() == 'Windows':
                os.startfile(path)
            elif platform.system() == 'Darwin':  # macOS
                subprocess.call(('open', path))
            else:  # Linux and other Unix-like
                subprocess.call(('xdg-open', path))
        except Exception as e:
            self.show_error(f"Error opening file: {str(e)}")

def main():
    app = QApplication(sys.argv)
    window = SetswanaMarkingSystem()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()