#!/bin/bash

# Sentiment Classification Environment Setup Script
# This script sets up the Python virtual environment and installs dependencies

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Sentiment Classification Environment Setup ===${NC}"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Python version to use
PYTHON_VERSION="python3.13"

# Check if Python 3.13 is available
echo -e "${YELLOW}Checking Python version...${NC}"
if command -v $PYTHON_VERSION &> /dev/null; then
    echo -e "${GREEN}✓ Found $PYTHON_VERSION${NC}"
    PYTHON_CMD=$PYTHON_VERSION
elif command -v python3 &> /dev/null; then
    PY_VER=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    echo -e "${YELLOW}Python 3.13 not found. Using python3 (version $PY_VER)${NC}"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Python 3 not found. Please install Python 3.13 or later.${NC}"
    exit 1
fi

# Create virtual environment
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}Removing existing virtual environment...${NC}"
        rm -rf "$VENV_DIR"
        echo -e "${GREEN}Creating new virtual environment...${NC}"
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi
else
    echo -e "${GREEN}Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# Install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
pip install torch numpy matplotlib

# Verify installation
echo ""
echo -e "${GREEN}=== Installation Complete ===${NC}"
echo -e "${YELLOW}Installed packages:${NC}"
pip list | grep -E "^(torch|numpy|matplotlib)"

# Print Python version
echo ""
echo -e "${YELLOW}Python version:${NC}"
python --version

# Check if GloVe embeddings exist
echo ""
if [ -f "data/glove.6B.50d-relativized.txt" ]; then
    echo -e "${GREEN}✓ GloVe embeddings found${NC}"
else
    echo -e "${YELLOW}⚠ GloVe embeddings not found in data/ directory${NC}"
    echo "  You may need to download them for the DAN model to work."
fi

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo -e "To activate the environment, run:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "To train a model, run:"
echo -e "  ${YELLOW}python sentiment_classifier.py --model LR --feats UNIGRAM${NC}"
echo -e "  ${YELLOW}python sentiment_classifier.py --model DAN${NC}"
