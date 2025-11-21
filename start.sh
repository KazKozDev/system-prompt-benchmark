#!/bin/bash

# System Prompt Security Benchmark - Startup Script
# This script starts the Universal Benchmark application

set -e  # Exit on error

echo "========================================="
echo "System Prompt Security Benchmark"
echo "Universal Edition"
echo "========================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.9+ from https://www.python.org/"
    exit 1
fi

# Check if Streamlit is installed
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "Streamlit is not installed. Installing dependencies..."
    pip install -r requirements.txt
    echo ""
fi

# Check if Ollama is running (optional but recommended)
if ! curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo "Warning: Ollama is not running"
    echo "Automatic judging will not work without Ollama"
    echo ""
    echo "To start Ollama:"
    echo "  1. Install from https://ollama.ai/"
    echo "  2. Run: ollama serve"
    echo "  3. Run: ollama pull qwen3:14b"
    echo ""
    read -p "Continue without Ollama? (y/n) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "Starting Universal Benchmark application..."
echo ""
echo "The application will open in your browser at:"
echo "  http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Streamlit with browser auto-open
python3 -m streamlit run app.py --server.port=8501 --server.address=localhost --browser.gatherUsageStats=false
