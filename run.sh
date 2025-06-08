#!/bin/bash

# Black Box Challenge - KNN-based Reimbursement Predictor
# This script uses a KNN model with extensive feature engineering to predict reimbursements
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate virtual environment and run Python predictor
cd "$SCRIPT_DIR"
source venv/bin/activate
python3 learned_predictor.py "$1" "$2" "$3"