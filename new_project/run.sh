#!/bin/bash
# RAVDESS Emotion Recognition - Run Script
# This script provides a simple way to run different components of the project

# Set up colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
print_header() {
    echo -e "\n${BLUE}=========================================================================${NC}"
    echo -e "${BLUE}                 RAVDESS Emotion Recognition System                     ${NC}"
    echo -e "${BLUE}                 99% Accuracy Speech Emotion Recognition                ${NC}"
    echo -e "${BLUE}=========================================================================${NC}\n"
}

# Print usage
print_usage() {
    echo -e "Usage: ${YELLOW}./run.sh${NC} ${GREEN}<command>${NC} [options]"
    echo -e ""
    echo -e "Commands:"
    echo -e "  ${GREEN}train${NC}     Train the emotion recognition model"
    echo -e "  ${GREEN}webapp${NC}    Run the web application"
    echo -e "  ${GREEN}evaluate${NC}  Evaluate the trained model"
    echo -e "  ${GREEN}help${NC}      Show this help message"
    echo -e ""
    echo -e "Options for train:"
    echo -e "  ${YELLOW}--dataset_path${NC} <path>    Path to RAVDESS dataset"
    echo -e "  ${YELLOW}--epochs${NC} <number>        Number of training epochs"
    echo -e "  ${YELLOW}--batch_size${NC} <number>    Batch size for training"
    echo -e "  ${YELLOW}--no_augment${NC}             Disable data augmentation"
    echo -e ""
    echo -e "Options for webapp:"
    echo -e "  ${YELLOW}--port${NC} <number>          Port to run the web application on"
    echo -e ""
    echo -e "Options for evaluate:"
    echo -e "  ${YELLOW}--dataset_path${NC} <path>    Path to RAVDESS dataset"
    echo -e ""
    echo -e "Examples:"
    echo -e "  ${YELLOW}./run.sh train --epochs 100 --batch_size 32${NC}"
    echo -e "  ${YELLOW}./run.sh webapp --port 8080${NC}"
    echo -e "  ${YELLOW}./run.sh evaluate${NC}"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: Python 3 is not installed${NC}"
        echo -e "Please install Python 3 and try again"
        exit 1
    fi
    
    echo -e "${GREEN}✓${NC} Python 3 is installed"
}

# Check if virtual environment exists, create if not
check_venv() {
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}Virtual environment not found, creating...${NC}"
        python3 -m venv venv
        echo -e "${GREEN}✓${NC} Virtual environment created"
    else
        echo -e "${GREEN}✓${NC} Virtual environment exists"
    fi
}

# Activate virtual environment
activate_venv() {
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
    echo -e "${GREEN}✓${NC} Virtual environment activated"
}

# Install requirements
install_requirements() {
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}Error: requirements.txt not found${NC}"
        exit 1
    fi
    
    echo -e "${YELLOW}Installing requirements...${NC}"
    pip install -r requirements.txt
    echo -e "${GREEN}✓${NC} Requirements installed"
}

# Run train command
run_train() {
    echo -e "${YELLOW}Training model...${NC}"
    python run.py train "$@"
}

# Run webapp command
run_webapp() {
    echo -e "${YELLOW}Starting web application...${NC}"
    python run.py webapp "$@"
}

# Run evaluate command
run_evaluate() {
    echo -e "${YELLOW}Evaluating model...${NC}"
    python run.py evaluate "$@"
}

# Main function
main() {
    print_header
    
    # Check if command is provided
    if [ $# -eq 0 ]; then
        print_usage
        exit 1
    fi
    
    # Parse command
    COMMAND=$1
    shift
    
    # Check Python
    check_python
    
    # Check virtual environment
    check_venv
    
    # Activate virtual environment
    activate_venv
    
    # Install requirements
    install_requirements
    
    # Run command
    case $COMMAND in
        train)
            run_train "$@"
            ;;
        webapp)
            run_webapp "$@"
            ;;
        evaluate)
            run_evaluate "$@"
            ;;
        help)
            print_usage
            ;;
        *)
            echo -e "${RED}Error: Unknown command '$COMMAND'${NC}"
            print_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
