import pandas as pd
import socket
import os
import configparser
from datetime import datetime

# Load configuration
config = configparser.ConfigParser()
config_path = r"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\config.ini"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")

config.read(config_path)

csv_path = r"D:\RISHIN\framework_for_uni_py\temp\copied_path\entirefolderB\NZEQ_modifiers_2_combinations.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df2 = pd.read_csv(csv_path)

output_dir = config["DATA"]["output_directory"]
#output_dir=r"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\out"
# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)
# Example usage of the loaded data
df2=df2.head(5)
df2.to_csv(os.path.join(output_dir, "output.csv"), index=False)