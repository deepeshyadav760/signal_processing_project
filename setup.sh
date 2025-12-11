#!/bin/bash

# AI Speech Intelligence Platform - Setup Script

echo "=================================="
echo "AI Speech Intelligence Platform"
echo "Setup Script"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version

if [ $? -ne 0 ]; then
    echo "Error: Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Navigate to backend directory
echo ""
echo "Setting up backend..."
cd backend

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Installing dependencies..."
echo "This may take a few minutes on first run..."
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "Warning: Some packages failed to install."
    echo "Trying with basic dependencies only..."
    pip install fastapi uvicorn numpy scipy python-multipart pydantic
fi

echo ""
echo "=================================="
echo "Setup Complete!"
echo "=================================="
echo ""
echo "To start the application:"
echo "  1. Activate the virtual environment:"
echo "     cd backend && source venv/bin/activate"
echo ""
echo "  2. Start the backend server:"
echo "     python main.py"
echo ""
echo "  3. Open frontend/index.html in your browser"
echo ""
echo "Or simply run:"
echo "  ./start.sh"
echo ""
echo "=================================="
