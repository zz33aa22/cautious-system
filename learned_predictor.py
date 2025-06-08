#!/usr/bin/env python3

import json
import sys
import math
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
import os
import pickle

def is_prime(n):
    """Check if a number is prime"""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True

def is_perfect_square(n):
    """Check if a number is a perfect square"""
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n

def extract_comprehensive_features(trip_duration_days, miles_traveled, total_receipts_amount):
    """Extract comprehensive features based on reverse engineering insights"""
    features = []
    
    # === BASIC FEATURES ===
    features.extend([trip_duration_days, miles_traveled, total_receipts_amount])
    
    # === DERIVED RATIOS ===
    miles_per_day = miles_traveled / trip_duration_days if trip_duration_days > 0 else 0
    spending_per_day = total_receipts_amount / trip_duration_days if trip_duration_days > 0 else 0
    features.extend([miles_per_day, spending_per_day])
    
    # === TRIP CATEGORIZATION ===
    features.append(1 if trip_duration_days == 1 else 0)
    features.append(1 if trip_duration_days == 2 else 0)
    features.append(1 if trip_duration_days == 3 else 0)
    features.append(1 if trip_duration_days == 4 else 0)
    features.append(1 if trip_duration_days == 5 else 0)  # 5-day bonus
    features.append(1 if trip_duration_days == 6 else 0)
    features.append(1 if trip_duration_days >= 7 else 0)
    features.append(1 if trip_duration_days >= 8 else 0)  # vacation penalty
    
    # === EFFICIENCY CATEGORIES ===
    features.append(1 if miles_per_day < 50 else 0)
    features.append(1 if 50 <= miles_per_day < 100 else 0)
    features.append(1 if 100 <= miles_per_day < 150 else 0)
    features.append(1 if 150 <= miles_per_day < 180 else 0)
    features.append(1 if 180 <= miles_per_day <= 220 else 0)  # optimal
    features.append(1 if 220 < miles_per_day < 300 else 0)
    features.append(1 if miles_per_day >= 300 else 0)
    
    # === MILEAGE TIERS ===
    tier1_miles = min(miles_traveled, 100)
    tier2_miles = min(max(0, miles_traveled - 100), 400)
    tier3_miles = max(0, miles_traveled - 500)
    features.extend([tier1_miles, tier2_miles, tier3_miles])
    
    # Tier percentages
    features.append(tier1_miles / max(miles_traveled, 1))
    features.append(tier2_miles / max(miles_traveled, 1))
    features.append(tier3_miles / max(miles_traveled, 1))
    
    # === RECEIPT PATTERNS ===
    cents = int((total_receipts_amount * 100) % 100)
    features.append(cents)
    
    # Prime and perfect square patterns for all receipt variations
    receipt_times_100 = int(total_receipts_amount * 100)
    receipt_int = int(total_receipts_amount)
    
    # Cents patterns
    features.append(1 if is_prime(cents) else 0)
    features.append(1 if is_perfect_square(cents) else 0)
    
    # Receipt amount patterns  
    features.append(1 if is_prime(receipt_int) else 0)
    features.append(1 if is_perfect_square(receipt_int) else 0)
    
    # Receipt × 100 patterns
    features.append(1 if is_prime(receipt_times_100) else 0)
    features.append(1 if is_perfect_square(receipt_times_100) else 0)
    
    # Receipt × 100, drop trailing zeros, check if prime
    receipt_no_zeros = int(str(receipt_times_100).rstrip('0'))
    features.append(1 if is_prime(receipt_no_zeros) else 0)
    
    # Receipt with consecutive repeating digits (ignore decimal, but .00 doesn't count as repeat)
    receipt_str = str(receipt_times_100)  # Convert to cents to ignore decimal
    has_consecutive_repeats = False
    for i in range(len(receipt_str) - 1):
        if receipt_str[i] == receipt_str[i + 1]:
            has_consecutive_repeats = True
            break
    features.append(1 if has_consecutive_repeats else 0)
    
    # Receipt reversal (multiply by 100, remove trailing zeros) mod 99
    receipt_times_100_str = str(int(total_receipts_amount * 100)).rstrip('0')
    receipt_reversed = int(receipt_times_100_str[::-1])
    receipt_original = int(receipt_times_100_str)
    receipt_diff = abs(receipt_original - receipt_reversed)
    features.append(1 if receipt_diff % 99 == 0 else 0)
    
    # Receipt * 100, remove trailing zeros, mod 6
    receipt_no_zeros = int(str(int(total_receipts_amount * 100)).rstrip('0'))
    features.append(receipt_no_zeros % 6)
    
    # Special cent endings
    features.append(1 if cents in [49, 99] else 0)  # Important: cents_49_99 was #9 feature!
    features.append(1 if cents in [47, 97] else 0)
    features.append(1 if cents in [29, 79] else 0)
    features.append(1 if cents == 0 else 0)
    
    # === MILE PATTERNS ===
    features.append(1 if is_prime(miles_traveled) else 0)
    features.append(1 if is_perfect_square(miles_traveled) else 0)
    features.append(miles_traveled % 10)
    features.append(1 if miles_traveled % 25 == 0 else 0)
    features.append(1 if miles_traveled % 50 == 0 else 0)
    
    # Mile reverse subtraction pattern
    miles_str = str(int(miles_traveled))
    miles_reversed = int(miles_str[::-1])
    mile_diff = abs(miles_traveled - miles_reversed)
    features.append(1 if mile_diff % 99 == 0 else 0)
    
    # Miles divisible by 9
    features.append(1 if miles_traveled % 9 == 0 else 0)
    
    # Miles × 100, drop trailing zeros, check if prime
    miles_times_100 = int(miles_traveled * 100)
    miles_no_zeros = int(str(miles_times_100).rstrip('0'))
    features.append(1 if is_prime(miles_no_zeros) else 0)
    
    # Miles with no repeating digits
    miles_str = str(int(miles_traveled))
    has_unique_digits = len(miles_str) == len(set(miles_str))
    features.append(1 if has_unique_digits else 0)
    
    # Miles with consecutive repeating digits (like 112, 122)
    has_consecutive_repeats = False
    for i in range(len(miles_str) - 1):
        if miles_str[i] == miles_str[i + 1]:
            has_consecutive_repeats = True
            break
    features.append(1 if has_consecutive_repeats else 0)
    
    # === SPENDING OPTIMIZATION ===
    if trip_duration_days <= 3:
        optimal_spending = 1 if spending_per_day <= 75 else 0
    elif 4 <= trip_duration_days <= 6:
        optimal_spending = 1 if spending_per_day <= 120 else 0
    else:
        optimal_spending = 1 if spending_per_day <= 90 else 0
    features.append(optimal_spending)
    
    # Spending categories
    features.append(1 if spending_per_day < 50 else 0)
    features.append(1 if 50 <= spending_per_day < 75 else 0)
    features.append(1 if 75 <= spending_per_day < 100 else 0)
    features.append(1 if 100 <= spending_per_day < 150 else 0)
    features.append(1 if spending_per_day >= 150 else 0)
    
    # === SPECIAL COMBINATIONS ===
    sweet_spot_combo = (trip_duration_days == 5 and 
                       miles_per_day >= 180 and 
                       spending_per_day < 100)
    features.append(1 if sweet_spot_combo else 0)
    
    vacation_penalty = (trip_duration_days >= 8 and spending_per_day > 100)
    features.append(1 if vacation_penalty else 0)
    
    high_efficiency_low_spend = (miles_per_day > 200 and spending_per_day < 80)
    features.append(1 if high_efficiency_low_spend else 0)
    
    low_efficiency_high_spend = (miles_per_day < 100 and spending_per_day > 120)
    features.append(1 if low_efficiency_high_spend else 0)
    
    # === SMALL RECEIPT PENALTIES ===
    features.append(1 if total_receipts_amount < 30 and trip_duration_days > 1 else 0)
    features.append(1 if total_receipts_amount < 50 and trip_duration_days > 2 else 0)
    features.append(1 if total_receipts_amount < 100 and trip_duration_days > 4 else 0)
    
    # === INTERACTION FEATURES ===
    efficiency_score = (miles_per_day * trip_duration_days) / (spending_per_day + 1)
    features.append(efficiency_score)        # Feature #6 in importance!
    features.append(efficiency_score ** 0.5) # Feature #5 in importance!
    
    total_spending_efficiency = total_receipts_amount / (miles_traveled + 1)
    features.append(total_spending_efficiency)
    
    trip_density = (miles_traveled + total_receipts_amount) / trip_duration_days
    features.append(trip_density)
    
    # === NON-LINEAR TRANSFORMATIONS ===
    # These are the top features from importance analysis!
    features.append(trip_duration_days ** 2)     # duration_sq
    features.append(trip_duration_days ** 0.5)   # duration_sqrt - Feature #8!
    features.append(math.log(trip_duration_days + 1))  # log_duration
    
    features.append(miles_traveled ** 0.5)
    features.append(math.log(miles_traveled + 1))
    features.append(miles_traveled ** 0.25)
    
    features.append(total_receipts_amount ** 0.5)    # receipts_sqrt - Feature #3!
    features.append(math.log(total_receipts_amount + 1))  # log_receipts - Feature #1!
    features.append(total_receipts_amount ** 2)      # receipts_sq - Feature #4!
    
    # === ESTIMATED COMPONENTS ===
    base_per_diem = trip_duration_days * 100        # Feature #11
    features.append(base_per_diem)
    
    est_mileage = tier1_miles * 0.58 + tier2_miles * 0.45 + tier3_miles * 0.30
    features.append(est_mileage)
    
    receipt_reimbursement = min(total_receipts_amount * 0.7, trip_duration_days * 80)  # Feature #7!
    features.append(receipt_reimbursement)
    
    # === MAGIC PATTERNS ===
    features.append(1 if total_receipts_amount == 847 else 0)
    features.append(1 if abs(total_receipts_amount - 847) < 50 else 0)
    
    return features

class LearnedPredictor:
    def __init__(self):
        self.model = None
        self.trained = False
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(script_dir, 'learned_model.pkl')
        
        # Try to load pre-trained model
        if os.path.exists(self.model_path):
            self.load_model()
        else:
            self.train_model()
    
    def train_model(self):
        """Train the RandomForest model on public data"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(script_dir, 'public_cases.json'), 'r') as f:
                data = json.load(f)
        except FileNotFoundError:
            with open('public_cases.json', 'r') as f:
                data = json.load(f)
        
        X = []
        y = []
        
        for case in data:
            inp = case['input']
            features = extract_comprehensive_features(
                inp['trip_duration_days'],
                inp['miles_traveled'],
                inp['total_receipts_amount']
            )
            X.append(features)
            y.append(case['expected_output'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train XGBoost (gradient boosting model)
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror'
        )
        
        self.model.fit(X, y)
        self.trained = True
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load_model(self):
        """Load pre-trained model"""
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True
    
    def predict(self, trip_duration_days, miles_traveled, total_receipts_amount):
        """Predict reimbursement amount"""
        if not self.trained:
            raise ValueError("Model not trained!")
        
        features = extract_comprehensive_features(trip_duration_days, miles_traveled, total_receipts_amount)
        features = np.array([features])
        
        prediction = self.model.predict(features)[0]
        return round(prediction, 2)

# Global predictor instance
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = LearnedPredictor()
    return _predictor

def main():
    if len(sys.argv) != 4:
        print("Usage: python learned_predictor.py <trip_duration_days> <miles_traveled> <total_receipts_amount>")
        sys.exit(1)
    
    try:
        trip_duration_days = int(sys.argv[1])
        miles_traveled = float(sys.argv[2])
        total_receipts_amount = float(sys.argv[3])
        
        predictor = get_predictor()
        result = predictor.predict(trip_duration_days, miles_traveled, total_receipts_amount)
        
        print(result)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()