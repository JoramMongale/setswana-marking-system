
# setswana-marking-system
An automated marking system for Setswana language papers using Python and Claude API.(MVP)

echo "# Setswana Marking System

An automated system for marking Setswana language papers using Python and the Claude API.

## Features

- Import student answer files in text format
- Edit or create answer memos
- Process student responses using TswanaBERT
- Display results for correctness

## Setup Options

### Using Docker (Recommended)

1. **Prerequisites**
   - Docker and Docker Compose installed

2. **Setup Steps**
   ```bash
   # Clone the repository
   git clone https://github.com/yourusername/setswana-marking-system.git
   cd setswana-marking-system
   
   # Build and run the container
   docker build -t setswana-marking .
   docker run -it --rm setswana-marking
   ```

### Manual Setup

1. **System Dependencies**
   ```bash
   # On Ubuntu/Debian
   sudo apt-get update
   sudo apt-get install -y python3-pip python3-venv
   ```

2. **Python Environment**
   ```bash
   # Create and activate virtual environment
   python3 -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # venv\Scripts\activate  # On Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   python -m app.main
   ```

## Project Structure

```
setswana_marking_system/
│
├── app/                      # Main application directory
│   ├── __init__.py
│   ├── main.py               # Entry point for GUI application
│   ├── core/                 # Core processing modules
│   └── utils/                # Utility functions
│
├── data/                     # Data storage
│   ├── students/             # Student answer files
│   ├── memos/                # Answer keys/memos
│   └── results/              # Marking results
│
├── requirements.txt          # Project dependencies
├── Dockerfile                # Docker configuration
├── README.md                 # Project documentation
└── .env                      # Environment variables
```

## Development Workflow

1. Set up the development environment using either Docker or manual setup
2. Implement and test components incrementally:
   - File import for student answers
   - Memo editor for teachers
   - TswanaBERT integration for answer checking
   - UI for teacher interaction

## Requirements

- Python 3.8+
- TswanaBERT model
- PyQt5 for GUI

## License

This is a project for my Lovely Beatiful Wife Glory.


## Contributing

Contributions are welcome! Please submit a Pull Request.

