from train import estimate_price

def load_parameters(file_path='theta_values.txt'):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            theta0 = float(lines[0].strip())
            theta1 = float(lines[1].strip())
        return theta0, theta1
    except FileNotFoundError:
        print(f"Error: {file_path} not found. Train the model first.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def estimate_price_from_file(mileage, file_path='theta_values.txt'):
    theta0, theta1 = load_parameters(file_path)
    if theta0 is not None and theta1 is not None:
        return estimate_price(mileage, theta0, theta1)
    else:
        return None
