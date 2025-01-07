from train import *
from esti import *

def main():
    print("well lets evaluate your car")
    choice = input("Do you want to (1) Train the model or (2) Estimate a price? Enter 1 or 2: ").strip()
    
    if choice == "1":
        # Train the model
        data_file = input("Enter the path to the training data file (e.g., 'data.csv'): ").strip()
        learning_rate = float(input("Enter the learning rate (default 0.0001): ") or 0.0001)
        num_iterations = int(input("Enter the number of iterations (default 1000): ") or 1000)
        train_model(data_file, learning_rate, num_iterations)
    elif choice == "2":
        # Estimate a price
        mileage = float(input("Enter the mileage of the car: "))
        estimated_price = estimate_price_from_file(mileage)
        if estimated_price is not None:
            print(f"The estimated price for mileage {mileage} is: {estimated_price:.2f}")
    else:
        print("Invalid choice. Please run the program again and select 1 or 2.")

if __name__ == "__main__":
    main()