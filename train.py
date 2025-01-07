import csv
import numpy as np

def estimate_price(mileage, theta0, theta1):
    return theta0 + (theta1 * mileage)

def train_model(data_file, learning_rate=0.0001, num_iterations=1000):
    # Read data from CSV file
    mileages = []
    prices = []
    
    with open(data_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header
        for row in csv_reader:
            mileages.append(float(row[0]))
            prices.append(float(row[1]))
    
    # Convert to numpy arrays
    mileages = np.array(mileages)
    prices = np.array(prices)
    
    # Normalize the data to prevent numerical issues
    mileage_mean = np.mean(mileages)
    mileage_std = np.std(mileages)
    normalized_mileages = (mileages - mileage_mean) / mileage_std
    
    price_mean = np.mean(prices)
    price_std = np.std(prices)
    normalized_prices = (prices - price_mean) / price_std
    
    # Initialize parameters
    theta0 = 0.0
    theta1 = 0.0
    m = len(mileages)  # Number of training examples
    
    print("Starting training...")
    
    # Training loop
    for iteration in range(num_iterations):
        # Vectorized computation
        predictions = estimate_price(normalized_mileages, theta0, theta1)
        errors = predictions - normalized_prices
        
        # Compute gradients
        gradient_theta0 = (1/m) * np.sum(errors)
        gradient_theta1 = (1/m) * np.sum(errors * normalized_mileages)
        
        # Update parameters
        theta0 = theta0 - learning_rate * gradient_theta0
        theta1 = theta1 - learning_rate * gradient_theta1
        
        # Print progress occasionally
        if iteration % 100 == 0:
            print(f"Iteration {iteration}: theta0 = {theta0:.6f}, theta1 = {theta1:.6f}")
    
    # Denormalize parameters for actual predictions
    final_theta1 = (theta1 * price_std) / mileage_std
    final_theta0 = (theta0 * price_std) + price_mean - (final_theta1 * mileage_mean)
    
    # Save final theta values to file
    with open('theta_values.txt', 'w') as file:
        file.write(f"{final_theta0}\n")
        file.write(f"{final_theta1}\n")
    
    print("\nTraining complete!")
    print(f"Final theta0: {final_theta0}")
    print(f"Final theta1: {final_theta1}")

