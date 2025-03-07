import pandas as pd
import socket
import os
import configparser
from datetime import datetime

# Load configuration
config = configparser.ConfigParser()
config.read(r"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\config.ini")

# Get machine name
machine_name = socket.gethostname()

# Get paths from config
split_csv_path = f"D:\\RISHIN\\framework_for_uni_py\\SETUP_WORK\\temp\\split_data_{machine_name}.csv"
output_dir = config["DATA"]["output_directory"]

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read CSV and extract first row
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
