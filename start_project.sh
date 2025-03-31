#!/bin/bash
# Quick start script for RAVDESS Emotion Recognition Project

# Set up colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Print header
echo -e "\n${BLUE}=========================================================================${NC}"
echo -e "${BLUE}                 RAVDESS Emotion Recognition System                     ${NC}"
echo -e "${BLUE}                 99% Accuracy Speech Emotion Recognition                ${NC}"
echo -e "${BLUE}=========================================================================${NC}\n"

# Navigate to new project directory
echo -e "${YELLOW}Navigating to new project directory...${NC}"
cd new_project

# Make run.sh executable
echo -e "${YELLOW}Making run.sh executable...${NC}"
chmod +x run.sh

# Display options
echo -e "\n${GREEN}Quick Start Options:${NC}"
echo -e "1. ${YELLOW}Train the model${NC}"
echo -e "2. ${YELLOW}Run the web application${NC}"
echo -e "3. ${YELLOW}Evaluate the model${NC}"
echo -e "4. ${YELLOW}Exit${NC}"

# Get user choice
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo -e "\n${GREEN}Starting model training...${NC}"
        ./run.sh train --dataset_path ../RAVDESS/dataset
        ;;
    2)
        echo -e "\n${GREEN}Starting web application...${NC}"
        ./run.sh webapp
        ;;
    3)
        echo -e "\n${GREEN}Evaluating model...${NC}"
        ./run.sh evaluate
        ;;
    4)
        echo -e "\n${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "\n${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac
