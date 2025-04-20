"""
Report generation module for the Setswana Marking System.
This module creates reports for student assessments.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import fitz  # PyMuPDF
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    """Class for generating assessment reports."""
    
    def __init__(self, output_directory="./data/results"):
        """
        Initialize the report generator.
        
        Args:
            output_directory (str): Directory to save generated reports
        """
        self.output_directory = output_directory
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_directory):
            try:
                os.makedirs(output_directory)
                logger.info(f"Created output directory: {output_directory}")
            except Exception as e:
                logger.error(f"Error creating output directory: {str(e)}")
                raise
    
    def create_annotated_pdf(self, original_pdf_path, marking_results, output_path=None):
        """
        Create an annotated PDF with marking results.
        
        Args:
            original_pdf_path (str): Path to the original student answer PDF
            marking_results (dict): Dictionary with marking results
            output_path (str, optional): Path to save the annotated PDF
            
        Returns:
            str: Path to the generated PDF
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"marked_{os.path.basename(original_pdf_path).replace('.pdf', '')}_{timestamp}.pdf"
            output_path = os.path.join(self.output_directory, filename)
        
        try:
            # Open the original PDF
            doc = fitz.open(original_pdf_path)
            
            # Add overall score to first page
            first_page = doc[0]
            overall_score = marking_results.get("overall_score", 0)
            overall_grade = marking_results.get("overall_grade", "")
            
            # Add a red rectangle at the top of the first page
            rect = fitz.Rect(50, 50, 550, 100)
            first_page.draw_rect(rect, color=(1, 0, 0), width=2)
            
            # Add text annotation
            text_point = fitz.Point(60, 80)
            first_page.insert_text(
                text_point,
                f"Overall Score: {overall_score:.1f}% - Grade: {overall_grade}",
                fontsize=16,
                color=(1, 0, 0)
            )
            
            # Add score for each question
            # We'll need to determine which page each question is on
            # For simplicity, we'll add annotations to sequential pages
            questions = marking_results.get("questions", {})
            
            # Simple approach: distribute questions across pages
            pages_count = len(doc)
            questions_per_page = max(1, len(questions) // pages_count)
            
            for i, (question_num, question_data) in enumerate(questions.items()):
                page_index = min(i // questions_per_page, pages_count - 1)
                page = doc[page_index]
                
                # Calculate vertical position based on question number
                vertical_pos = 120 + (i % questions_per_page) * 30
                
                # Add annotation text
                similarity = question_data.get("similarity", 0)
                grade = question_data.get("grade", "")
                
                annotation_text = f"Question {question_num}: {similarity:.1f}% - Grade: {grade}"
                
                text_point = fitz.Point(60, vertical_pos)
                page.insert_text(
                    text_point,
                    annotation_text,
                    fontsize=12,
                    color=(0, 0, 1)  # Blue color
                )
            
            # Save the annotated PDF
            doc.save(output_path)
            logger.info(f"Saved annotated PDF to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error creating annotated PDF: {str(e)}")
            raise
    
    def generate_individual_report(self, student_info, marking_results, output_path=None):
        """
        Generate a detailed PDF report for an individual student.
        
        Args:
            student_info (dict): Dictionary with student information
            marking_results (dict): Dictionary with marking results
            output_path (str, optional): Path to save the report
            
        Returns:
            str: Path to the generated report
        """
        if output_path is None:
            student_id = student_info.get("id", "unknown")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{student_id}_{timestamp}.pdf"
            output_path = os.path.join(self.output_directory, filename)
        
        try:
            # Create document
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()
            
            # Add custom style for Setswana text
            styles.add(ParagraphStyle(
                name='SetswanaText',
                parent=styles['Normal'],
                fontName='Helvetica',
                fontSize=10,
                spaceAfter=6
            ))
            
            # Create content elements
            elements = []
            
            # Title
            title = Paragraph("Setswana Assessment Report", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Student Information
            student_name = student_info.get("name", "Unknown")
            student_id = student_info.get("id", "Unknown")
            assessment_date = student_info.get("date", datetime.now().strftime("%Y-%m-%d"))
            
            student_info_data = [
                ["Student Name:", student_name],
                ["Student ID:", student_id],
                ["Assessment Date:", assessment_date],
                ["Overall Score:", f"{marking_results.get('overall_score', 0):.1f}%"],
                ["Overall Grade:", marking_results.get('overall_grade', '')]
            ]
            
            student_table = Table(student_info_data, colWidths=[2*inch, 4*inch])
            student_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(student_table)
            elements.append(Spacer(1, 24))
            
            # Question Details
            elements.append(Paragraph("Question Details", styles['Heading2']))
            elements.append(Spacer(1, 12))
            
            questions = marking_results.get("questions", {})
            
            for question_num, question_data in sorted(questions.items()):
                # Question header
                question_header = Paragraph(f"Question {question_num}", styles['Heading3'])
                elements.append(question_header)
                
                # Score and grade
                score_text = f"Score: {question_data.get('similarity', 0):.1f}% - Grade: {question_data.get('grade', '')}"
                score_para = Paragraph(score_text, styles['Normal'])
                elements.append(score_para)
                elements.append(Spacer(1, 6))
                
                # Student answer
                student_answer = question_data.get("student_answer", "")
                elements.append(Paragraph("Student Answer:", styles['Heading4']))
                elements.append(Paragraph(student_answer, styles['SetswanaText']))
                elements.append(Spacer(1, 6))
                
                # Correct answer
                correct_answer = question_data.get("correct_answer", "")
                elements.append(Paragraph("Correct Answer:", styles['Heading4']))
                elements.append(Paragraph(correct_answer, styles['SetswanaText']))
                
                elements.append(Spacer(1, 12))
            
            # Build the PDF
            doc.build(elements)
            logger.info(f"Generated individual report for {student_name} at {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating individual report: {str(e)}")
            raise
    
    def generate_class_report(self, class_results, output_path=None):
        """
        Generate a class-level report with statistics and visualizations.
        
        Args:
            class_results (list): List of dictionaries with student results
            output_path (str, optional): Path to save the report
            
        Returns:
            tuple: Paths to the generated report files (Excel, PDF)
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"class_report_{timestamp}.xlsx"
            pdf_filename = f"class_report_{timestamp}.pdf"
            chart_filename = f"score_distribution_{timestamp}.png"
            
            excel_path = os.path.join(self.output_directory, excel_filename)
            pdf_path = os.path.join(self.output_directory, pdf_filename)
            chart_path = os.path.join(self.output_directory, chart_filename)
        else:
            base_path = os.path.splitext(output_path)[0]
            excel_path = f"{base_path}.xlsx"
            pdf_path = f"{base_path}.pdf"
            chart_path = f"{base_path}_chart.png"
        
        try:
            # Prepare data for reports
            report_data = []
            
            for student_result in class_results:
                student_info = student_result.get("student_info", {})
                marking_results = student_result.get("marking_results", {})
                
                student_row = {
                    "Student ID": student_info.get("id", "Unknown"),
                    "Student Name": student_info.get("name", "Unknown"),
                    "Overall Score": marking_results.get("overall_score", 0),
                    "Overall Grade": marking_results.get("overall_grade", "")
                }
                
                # Add individual question scores
                questions = marking_results.get("questions", {})
                for question_num, question_data in questions.items():
                    student_row[f"Q{question_num} Score"] = question_data.get("similarity", 0)
                    student_row[f"Q{question_num} Grade"] = question_data.get("grade", "")
                
                report_data.append(student_row)
            
            # Create DataFrame
            df = pd.DataFrame(report_data)
            
            # Generate Excel report
            df.to_excel(excel_path, index=False)
            logger.info(f"Generated class Excel report at {excel_path}")
            
            # Generate score distribution chart
            plt.figure(figsize=(10, 6))
            
            if not df.empty and "Overall Score" in df.columns:
                plt.hist(df["Overall Score"], bins=10, alpha=0.7, color='blue')
                plt.title("Distribution of Scores")
                plt.xlabel("Score (%)")
                plt.ylabel("Number of Students")
                plt.grid(True, alpha=0.3)
                
                # Add stats to the chart
                avg_score = df["Overall Score"].mean()
                max_score = df["Overall Score"].max()
                min_score = df["Overall Score"].min()
                
                stats_text = (f"Average: {avg_score:.1f}%\n"
                             f"Maximum: {max_score:.1f}%\n"
                             f"Minimum: {min_score:.1f}%")
                
                plt.annotate(stats_text, xy=(0.7, 0.8), xycoords='axes fraction',
                            bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(chart_path)
                plt.close()
            
            # Generate PDF summary report
            doc = SimpleDocTemplate(pdf_path, pagesize=letter)
            styles = getSampleStyleSheet()
            elements = []
            
            # Title
            title = Paragraph("Setswana Class Assessment Report", styles['Title'])
            elements.append(title)
            elements.append(Spacer(1, 12))
            
            # Date and class info
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            date_para = Paragraph(f"Report Date: {timestamp}", styles['Normal'])
            elements.append(date_para)
            elements.append(Spacer(1, 12))
            
            # Summary statistics
            if not df.empty and "Overall Score" in df.columns:
                avg_score = df["Overall Score"].mean()
                max_score = df["Overall Score"].max()
                min_score = df["Overall Score"].min()
                
                stats_data = [
                    ["Statistic", "Value"],
                    ["Number of Students", len(df)],
                    ["Average Score", f"{avg_score:.1f}%"],
                    ["Maximum Score", f"{max_score:.1f}%"],
                    ["Minimum Score", f"{min_score:.1f}%"]
                ]
                
                stats_table = Table(stats_data, colWidths=[3*inch, 3*inch])
                stats_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(stats_table)
                elements.append(Spacer(1, 24))
            
            # Score distribution chart
            if os.path.exists(chart_path):
                chart = Image(chart_path, width=6*inch, height=4*inch)
                elements.append(chart)
                elements.append(Spacer(1, 12))
            
         # app/utils/report_generator.py (continued)

            # Grade distribution
            if not df.empty and "Overall Grade" in df.columns:
                grade_counts = df["Overall Grade"].value_counts().sort_index()
                
                grade_data = [["Grade", "Count", "Percentage"]]
                for grade, count in grade_counts.items():
                    percentage = (count / len(df)) * 100
                    grade_data.append([grade, count, f"{percentage:.1f}%"])
                
                grade_table = Table(grade_data, colWidths=[2*inch, 2*inch, 2*inch])
                grade_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(Paragraph("Grade Distribution", styles['Heading2']))
                elements.append(Spacer(1, 6))
                elements.append(grade_table)
                elements.append(Spacer(1, 24))
            
            # Student scores table (top 10)
            if not df.empty:
                elements.append(Paragraph("Top 10 Student Scores", styles['Heading2']))
                elements.append(Spacer(1, 6))
                
                top_students = df.sort_values("Overall Score", ascending=False).head(10)
                
                # Create table data
                top_data = [["Student ID", "Student Name", "Score", "Grade"]]
                
                for _, row in top_students.iterrows():
                    top_data.append([
                        row["Student ID"],
                        row["Student Name"],
                        f"{row['Overall Score']:.1f}%",
                        row["Overall Grade"]
                    ])
                
                top_table = Table(top_data, colWidths=[1.5*inch, 2.5*inch, 1*inch, 1*inch])
                top_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(top_table)
            
            # Build the PDF
            doc.build(elements)
            logger.info(f"Generated class PDF report at {pdf_path}")
            
            # Clean up chart file if needed
            # os.remove(chart_path)
            
            return excel_path, pdf_path
            
        except Exception as e:
            logger.error(f"Error generating class report: {str(e)}")
            raise

# Example usage
if __name__ == "__main__":
    report_gen = ReportGenerator()
    
    # Example: Generate a sample individual report
    student_info = {
        "id": "STD123",
        "name": "Mpho Molefe",
        "date": "2025-04-12"
    }
    
    marking_results = {
        "overall_score": 82.5,
        "overall_grade": "B",
        "questions": {
            1: {
                "student_answer": "Monyaise DPS",
                "correct_answer": "Monyaise DPS",
                "similarity": 100.0,
                "grade": "A"
            },
            2: {
                "student_answer": "Tlhaolele e ne e le tsamaiso ya kgethololo e e neng e kgetholla batho go ya ka mmala wa letlalo.",
                "correct_answer": "Tlhaolele e ne e le tsamaiso ya kgethololo e e neng e kgetholla batho go ya ka mmala wa letlalo.",
                "similarity": 95.0,
                "grade": "A"
            },
            3: {
                "student_answer": "Maboko a setso le maboko a segompieno",
                "correct_answer": "Maboko a setso (ditoko, dipoko) le maboko a segompieno",
                "similarity": 85.0,
                "grade": "B"
            }
        }
    }
    
    try:
        report_path = report_gen.generate_individual_report(student_info, marking_results)
        print(f"Generated report at: {report_path}")
    except Exception as e:
        print(f"Error: {str(e)}")