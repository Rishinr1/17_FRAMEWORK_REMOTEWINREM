import pandas as pd
import socket
import os
import configparser
import time
import shutil
from datetime import datetime

# Load configuration
config = configparser.ConfigParser()
config.read(r"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\config.ini")

# File paths
shared_csv_path = r"\\CA1MDLIDATA2408\D$\RISHIN\framework_for_uni_py\demo_csv.csv"
temp_shared_csv_path = r"D:\RISHIN\framework_for_uni_py\temp_demo_csv.csv"
split_csv_path = rf"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\split_data_{socket.gethostname()}.csv"

# Get machine name
machine_name = socket.gethostname()

# Ensure output directory exists
output_dir = config["DATA"]["output_directory"]
os.makedirs(output_dir, exist_ok=True)

# Read CSV (split_data file) without locking
df = pd.read_csv(split_csv_path)

if not df.empty:
    first_row = df.iloc[:1]

    # Generate unique output filename
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_filename = f"output_{machine_name}_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)

    # Save to output directory
    first_row.to_csv(output_path, index=False)
    print(f"Processed {split_csv_path} and saved first row to {output_path}")
else:
    print(f"Warning: {split_csv_path} is empty. No data processed.")

# Read shared CSV with retry mechanism to handle file locks
max_retries = 5
retry_delay = 10  # Seconds

for attempt in range(max_retries):
    try:
        # Create a temporary copy of the shared CSV
        shutil.copy(shared_csv_path, temp_shared_csv_path)
        df2 = pd.read_csv(temp_shared_csv_path)
        df2 = df2.head(5)

        # Write changes safely
        df2.to_csv(r"D:\RISHIN\framework_for_uni_py\demo_success.csv", index=False)
        print(f"Successfully wrote to demo_success.csv")
        break  # Success!
    except PermissionError:
        print(f"Attempt {attempt + 1}: File is locked, retrying in {retry_delay} sec...")
        time.sleep(retry_delay)
else:
    print("Failed to access shared CSV after multiple attempts.")
