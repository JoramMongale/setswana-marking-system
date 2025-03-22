# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies including Tesseract OCR and Setswana language pack
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    libleptonica-dev \
    pkg-config \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libfontconfig1 \
    libice6 \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Setswana language data for Tesseract
RUN mkdir -p /usr/share/tesseract-ocr/4.00/tessdata
RUN wget https://github.com/tesseract-ocr/tessdata/raw/main/tsn.traineddata -P /usr/share/tesseract-ocr/4.00/tessdata/

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install PyQt5 dependencies
RUN apt-get update && apt-get install -y \
    qt5-default \
    libqt5gui5 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Set environment variables
ENV PYTHONPATH=/app
ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

# Command to run the application
CMD ["python", "-m", "app.main"]