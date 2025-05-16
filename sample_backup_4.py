import polars as pl
import glob
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os
import gc
import pandas as pd,glob, os, numpy as np, matplotlib.pyplot as plt 
from collections import defaultdict
import multiprocessing as mp
import configparser
import socket
import platform
machine_name = socket.gethostname()

gc.collect()
# Get the path to the config.ini file in the current working directory
# Set the path to the config.ini file
config_file_path = r"D:\RISHIN\framework_for_uni_py\SETUP_WORK\temp\copied_path\entirefolderB\config.ini"

# Initialize the ConfigParser
config = configparser.ConfigParser()

# Read the config.ini file
config.read(config_file_path)
import os
import pyodbc

# === CONFIGURATION ===
mdf_path = config['python']['mdf_path'] # <-- update this path
database_name = config['python']['database_name']          # <-- update this name
server_name = r'localhost'       # <-- update your SQL Server instance if needed

# Construct the expected LDF file path
ldf_path = mdf_path.replace('.mdf', '_log.ldf')

# Connect to SQL Server using Windows Authentication
conn_str = f'DRIVER={{SQL Server}};SERVER={server_name};Trusted_Connection=yes;'
conn = pyodbc.connect(conn_str, autocommit=True)
cursor = conn.cursor()

# First check if the database is already attached
print(f"Checking if '{database_name}' is already attached...")
cursor.execute("SELECT name FROM sys.databases WHERE name = ?", database_name)
db_exists = cursor.fetchone() is not None

if db_exists:
    # If database exists, detach it first
    print(f"Database '{database_name}' is already attached. Detaching first...")
    try:
        # Force close existing connections and detach
        cursor.execute(f"ALTER DATABASE [{database_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
        cursor.execute(f"EXEC sp_detach_db '{database_name}', 'true'")
        print(f"Database '{database_name}' detached successfully.")
        
        # Wait briefly to ensure file handles are released
        import time
        time.sleep(2)
    except Exception as e:
        print(f"Warning: Could not detach database - {str(e)}")

# Now try to delete the log file if it exists
if os.path.exists(ldf_path):
    print(f"Attempting to delete existing log file: {ldf_path}")
    try:
        os.remove(ldf_path)
        print(f"Log file deleted successfully.")
    except PermissionError:
        print(f"Warning: Could not delete log file - it may still be in use.")
        print(f"Will attempt to rebuild log file during attach operation.")

# Attach database from MDF only (SQL Server will rebuild the LDF)
print(f"Attaching database '{database_name}' from MDF only...")
attach_sql = f"""
CREATE DATABASE [{database_name}]
ON (FILENAME = N'{mdf_path}')
FOR ATTACH_REBUILD_LOG
"""

try:
    cursor.execute(attach_sql)
    print(f"Database '{database_name}' attached successfully with a new log file.")
except Exception as e:
    print(f"Error attaching database: {e}")
    raise


import struct
import polars as pl
import numpy as np
import os
from collections import defaultdict
from sqlalchemy import create_engine, Engine

def engine_initializer(server: str, db: str) -> Engine:
    """Create and return a SQL Server database engine.
    
    Args:
        server: SQL Server instance name
        db: Database name
    
    Returns:
        SQLAlchemy Engine instance configured for SQL Server connection
    """
    connection_string = (
        "DRIVER={SQL Server};"
        f"SERVER={server};"
        f"DATABASE={db};"
        "Trusted_Connection=yes;"
    )
    engine = create_engine(f"mssql+pyodbc:///?odbc_connect={connection_string}")
    return engine

def parse_vulnerability_binary_files(index_binary_path: str, data_binary_path: str) -> tuple:
    """
    Parse vulnerability data from binary files containing index and hazard data.
    
    Args:
        index_binary_path: Path to the index binary file
        data_binary_path: Path to the data binary file
        
    Returns:
        tuple containing:
            - count: Number of records
            - index_df: DataFrame containing PDCNUM, DCKEY, CGHSID, and INDEX
            - haz_sev_array: Array of hazard severity values
            - damage_perc_array_all: Array of damage percentage values
            
    Raises:
        ValueError: If input paths are not strings
        FileNotFoundError: If either binary file is not found
        struct.error: If binary data cannot be unpacked
    """
    if not isinstance(index_binary_path, str):
        raise ValueError('index_binary_path must be a string')
    if not isinstance(data_binary_path, str):
        raise ValueError('data_binary_path must be a string')

    if not os.path.exists(index_binary_path):
        raise FileNotFoundError(f'Index binary file not found: {index_binary_path}')
    if not os.path.exists(data_binary_path):
        raise FileNotFoundError(f'Data binary file not found: {data_binary_path}')
    
    with open(index_binary_path, 'rb') as f1:
        count = struct.unpack('h', f1.read(2))[0]
        records = []

        while True:
            try:
                pdcnum = struct.unpack('h', f1.read(2))[0]
                dckey = struct.unpack('B', f1.read(1))[0]
                cghsid = struct.unpack('B', f1.read(1))[0]
                index = struct.unpack('i', f1.read(4))[0]
                records.append({'PDCNUM': pdcnum, 'DCKEY': dckey, 'CGHSID': cghsid, 'INDEX': index})
            except:
                break

    index_df = pl.DataFrame(records)
    with open(data_binary_path, 'rb') as f2:
        haz_sev_array = np.array([struct.unpack('f', f2.read(4))[0] for _ in range(count)])
        floats = []
        while True:
            chunk = f2.read(4)
            
            if len(chunk) < 4:
                break
            try:
                num = struct.unpack('f', chunk)[0]
                floats.append(num)
            except struct.error:
                print("Error unpacking data")
                break
    damage_perc_array_all = np.array(floats)
    return count, index_df, haz_sev_array, damage_perc_array_all

def get_damage_percentage_single_row(dckey: int, pdcnum: int, cghsid: int, index_df: pl.DataFrame, damage_perc_array_all: np.ndarray, count: int) -> np.ndarray:
    try:
        index_value = index_df.filter(
            (pl.col('DCKEY') == dckey) & 
            (pl.col('PDCNUM') == pdcnum) & 
            (pl.col('CGHSID') == cghsid)
        ).select('INDEX').to_series().item()
        result_array = damage_perc_array_all[(index_value - 1) * count:(index_value - 1) * count + count]
    except IndexError:
        raise Exception(f'No record found for PdcNum: {pdcnum} DCKey: {dckey} CghsId: {cghsid} in index_df.')
    except:
        raise Exception(f'Error while fetching damage percentage for PdcNum: {pdcnum} DCKey: {dckey} CghsId: {cghsid}')
    if len(result_array) != count:
        raise Exception(f'Invalid length of damage percentage array for PdcNum: {pdcnum} DCKey: {dckey} CghId: {cghsid} - Expected length: {count}, Actual length: {len(result_array)}.')
    if not np.all(result_array[:-1] <= result_array[1:]):
        raise Exception(f'Damage percentage array for PdcNum: {pdcnum} DCKey: {dckey} CghId: {cghsid} is not monotonically increasing.')
    return result_array

def get_damage_percentage_multiple_rows(dckey: int, vinv_output: pl.DataFrame, cghsid: int, index_df: pl.DataFrame, damage_perc_array_all: np.ndarray, count: int) -> np.ndarray:
    final_damage_perc = np.zeros(count)
    for row in vinv_output.iter_rows(named=True):
        damage_perc = get_damage_percentage_single_row(
            dckey=dckey,
            pdcnum=row['PdcNum'],
            cghsid=cghsid,
            index_df=index_df,
            damage_perc_array_all=damage_perc_array_all,
            count=count
        )
        final_damage_perc += damage_perc * (row['InvPercentage'] / 100)
    return final_damage_perc

def get_damage_percentage(dckey: int, vinv_output: pl.DataFrame, cghsid: int, index_df: pl.DataFrame, damage_perc_array_all: np.ndarray, count: int) -> np.ndarray:
    if type(dckey) != int:
        dckey = int(dckey)
    if type(cghsid) != int:
        cghsid = int(cghsid)
    if type(count) != int:
        count = int(count)
    if not isinstance(index_df, pl.DataFrame):
        raise ValueError('index_df should be a Polars DataFrame')
    if not isinstance(damage_perc_array_all, np.ndarray):
        raise ValueError('damage_perc_array_all should be a numpy array')
    if not isinstance(vinv_output, pl.DataFrame):
        raise ValueError('vinv_output should be a Polars DataFrame')
    
    if len(vinv_output) > 1:
        return get_damage_percentage_multiple_rows(
            dckey=dckey,
            vinv_output=vinv_output,
            cghsid=cghsid,
            index_df=index_df,
            damage_perc_array_all=damage_perc_array_all,
            count=count
        )
    elif len(vinv_output) == 1:
        return get_damage_percentage_single_row(
            dckey=dckey,
            pdcnum=vinv_output['PdcNum'][0],
            cghsid=cghsid,
            index_df=index_df,
            damage_perc_array_all=damage_perc_array_all,
            count=count
        )
    else:
        raise ValueError('Unable to get damage percentage. Check the input data.')

def organize_vulnerability_binary_files(file_paths: list[str]) -> dict[str, tuple[str, ...]]:
    file_dict = defaultdict(list)
    for path in file_paths:
        file_dict[os.path.basename(path).split('_')[0]].append(path)
    return {k: tuple(sorted(v)) for k, v in file_dict.items()}

def find_matching_vgeo_row(loc_row: dict, vgeo_df: pl.DataFrame) -> dict:
    """
    Find matching row in vgeo_df for a single row from loc_df using hierarchical fallback approach.
    If no match is found, returns the row where all geo IDs are 0.
    
    Parameters:
    -----------
    loc_row : dict
        A single row from loc_df containing POSTCODEEXTGEOID, ADMIN2EXTGEOID, and ADMIN1EXTGEOID
    vgeo_df : pl.DataFrame
        DataFrame containing PostalCodeGeoId, Admin2GeoId, and Admin1GeoId
    
    Returns:
    --------
    dict
        Matched row from vgeo_df
    """
    hierarchy = [
        ('LocationCodeGeoID', 'LocationCodeGeoId'),
        ('POSTCODEEXTGEOID', 'PostalCodeGeoId'),
        ('ADMIN2EXTGEOID', 'Admin2GeoId'),
        ('ADMIN1EXTGEOID', 'Admin1GeoId')
    ]
    
    for loc_col, vgeo_col in hierarchy:
        if loc_row[loc_col]is None or loc_row[loc_col] == 0:
            continue
            
        matches = vgeo_df.filter(pl.col(vgeo_col) == loc_row[loc_col])
        
        if matches.shape[0] > 0:
            return matches.row(0)
    
    default_row = vgeo_df.filter(
        (vgeo_df['LocationCodeGeoId'] == 0) &
        (pl.col('PostalCodeGeoId') == 0) & 
        (pl.col('Admin2GeoId') == 0) & 
        (pl.col('Admin1GeoId') == 0)
    )
    
    if default_row.shape[0] > 0:
        return default_row.row(0)
    else:
        raise ValueError("No default row found with all geo IDs set to 0")
    


def fetch_vinv_cghs_data(row, ConstructionMap_NZ, OccupancyMap_NZ, vcc, vocc, imap, vinv, cghs, inv_key,bikey, coverage, cvg_grade, haz_type, comb_row):
    height = comb_row['NUMSTORIES']
    year = comb_row['YEARBUILT'] if comb_row['YEARBUILT'] is not None else 0
    occupancy_code = comb_row['OCCTYPE']
    occupancy_scheme = comb_row['OCCSCHEME']
    construction_code = comb_row['BLDGCLASS']
    construction_scheme = comb_row['BLDGSCHEME']
    
    #print(f"Height: {height}, Year: {year}, Occupancy Code: {occupancy_code}, Occupancy Scheme: {occupancy_scheme}")
    #print(f"Construction Code: {construction_code}, Construction Scheme: {construction_scheme}")
    
    construction_id = ConstructionMap_NZ.filter(
        (pl.col('ConstructionScheme') == construction_scheme) & 
        (pl.col('ConstructionCode') == str(construction_code))
    ).select('ConstructionId').to_series().item()
    
    #print(f"Construction ID: {construction_id}")
    
    occupancy_id = OccupancyMap_NZ.filter(
        (pl.col('OccupancyScheme') == occupancy_scheme) & 
        (pl.col('OccupancyCode') == str(occupancy_code))
    ).select('OccupancyId').to_series().item()
    
    #print(f"Occupancy ID: {occupancy_id}")
    
    vcc_output = vcc.filter(pl.col('ConstructionId') == construction_id).select(['MapCCTier1', 'MapCCTier2', 'MapCCTier3']).to_dict(as_series=False)
    #print(f"VCC Output: {vcc_output}")
    
    vocc_output = vocc.filter(pl.col('OccupancyId') == occupancy_id).select(['InvOcc', 'BIOcc','OccupancyId']).to_dict(as_series=False)
    #print(f"VOCC Output: {vocc_output}")
    
    inv_occ = vocc_output['InvOcc'][0]
    bi_occ = vocc_output['BIOcc'][0]
    occ_id = vocc_output['OccupancyId'][0]
    # print(f"Inv Occ: {inv_occ}")
    # print(f"Inv_Key: {inv_key}")
    # print(f"BI_Key: {bikey}")
    # print(f"BIOcc: {bi_occ}")
    # print(f"Occ_ID: {occ_id}")
    imap_result = imap.filter(
        (pl.col('InvKey') == inv_key) & 
        (pl.col('MapCCTier1') == vcc_output['MapCCTier1'][0]) & 
        (pl.col('MapCCTier2') == vcc_output['MapCCTier2'][0]) & 
        (pl.col('MapCCTier3') == vcc_output['MapCCTier3'][0]) & 
        (pl.col('HeightFrom') <= height) & 
        (pl.col('HeightTo') >= height) & 
        (pl.col('YearFrom') <= year) & 
        (pl.col('YearTo') >= year) & 
        (pl.col('InvOcc') == inv_occ)
    ).select('InvRecNum').to_series().first()
    
    #print(f"IMAP Result: {imap_result}")
    
    vinv_output = vinv.filter(pl.col('InvRecNum') == imap_result).select(['PdcNum', 'InvPercentage']).to_dict(as_series=False)
    #print(f"VINV Output: {vinv_output}")
    
    cghs_output = cghs.filter(
        (pl.col('Coverage') == coverage) & 
        (pl.col('CvgGrade') == cvg_grade) & 
        (pl.col('HazType') == haz_type)
    ).select('CghsId').to_series().item()
    
    #print(f"CGHS Output: {cghs_output}")
    
    return vinv_output, cghs_output,bi_occ,occ_id



def calculate_pla_values(filtered_df_hazard, row, comb_row, occupancy_pla, pla):
    # Check if OccupancyCode and Scheme from comb_row are present in occupancy_pla
    if not occupancy_pla.filter((pl.col('OccupancyCode') == comb_row['OCCTYPE']) & (pl.col('OccupancyScheme') == comb_row['OCCSCHEME'])).is_empty():
        # Check if Admin1GeoID is present in pla table
        if row['Admin1GeoID'] in pla['Admin1GeoID'].to_list():
            pla_rows = pla.filter(pl.col('Admin1GeoID') == row['Admin1GeoID'])
            matching_pla_rows = pla_rows.filter(pl.col('LOBID') == 1)
            if not matching_pla_rows.is_empty():
                filtered_df_hazard = filtered_df_hazard.join(matching_pla_rows[['EVENTID', 'PLA_BDG', 'PLA_CTS','PLA_BI']], left_on='EventId', right_on='EVENTID', how='left')
                filtered_df_hazard = filtered_df_hazard.with_columns([
                    pl.col('PLA_BDG').fill_null(1),
                    pl.col('PLA_CTS').fill_null(1),
                    pl.col('PLA_BI').fill_null(1)
                ])
            else:
                filtered_df_hazard = filtered_df_hazard.with_columns([
                    pl.lit(1).alias('PLA_BDG'),
                    pl.lit(1).alias('PLA_CTS'),
                    pl.lit(1).alias('PLA_BI')
                ])
        else:
            filtered_df_hazard = filtered_df_hazard.with_columns([
                pl.lit(1).alias('PLA_BDG'),
                pl.lit(1).alias('PLA_CTS'),
                pl.lit(1).alias('PLA_BI')
            ])
    else:
        # If OccupancyCode and Scheme are not found in occupancy_pla, consider LOBID != 1
        if row['Admin1GeoID'] in pla['Admin1GeoID'].to_list():
            pla_rows = pla.filter(pl.col('Admin1GeoID') == row['Admin1GeoID'])
            non_lob1_rows = pla_rows.filter(pl.col('LOBID') != 1)
            if not non_lob1_rows.is_empty():
                filtered_df_hazard = filtered_df_hazard.join(non_lob1_rows[['EVENTID', 'PLA_BDG', 'PLA_CTS','PLA_BI']], left_on='EventId', right_on='EVENTID', how='left')
                filtered_df_hazard = filtered_df_hazard.with_columns([
                    pl.col('PLA_BDG').fill_null(1),
                    pl.col('PLA_CTS').fill_null(1), 
                    pl.col('PLA_BI').fill_null(1)
                ])
            else:
                filtered_df_hazard = filtered_df_hazard.with_columns([
                    pl.lit(1).alias('PLA_BDG'),
                    pl.lit(1).alias('PLA_CTS'),
                    pl.lit(1).alias('PLA_BI')
                ])
        else:
            filtered_df_hazard = filtered_df_hazard.with_columns([
                pl.lit(1).alias('PLA_BDG'),
                pl.lit(1).alias('PLA_CTS'),
                pl.lit(1).alias('PLA_BI')
            ])
    return filtered_df_hazard



#POlars convertion working
def interpolate_series2(x_query: pl.Series, x_known: pl.Series, y_known: pl.Series) -> pl.Series:
    # Ensure x_known and y_known are sorted by x_known
    sorted_df = pl.DataFrame({"x_known": x_known, "y_known": y_known}).sort("x_known")
    x_known = sorted_df["x_known"]
    y_known = sorted_df["y_known"]

    # Find the indices of the bins where x_query falls
    indices = x_known.search_sorted(x_query)

    # Clip indices to ensure they are within bounds
    indices = indices.clip(1, len(x_known) - 1)

    # Get the x and y values for interpolation
    x0 = x_known[indices - 1]
    x1 = x_known[indices]
    y0 = y_known[indices - 1]
    y1 = y_known[indices]

    # Perform linear interpolation
    interpolated = y0 + (x_query - x0) * (y1 - y0) / (x1 - x0)

    # Handle extrapolation for values outside the range of x_known
    interpolated = pl.when(x_query < x_known[0]).then(y_known[0]) \
                   .when(x_query > x_known[-1]).then(y_known[-1]) \
                   .otherwise(interpolated)

    return interpolated 

# Precompute bounds for return periods working
def precompute_bounds(rp_result, return_periods):
    sorted_rp = rp_result.sort("Return_Period")  # Ensure sorted by Return_Period
    lower_bounds = []
    higher_bounds = []

    for rp in return_periods:
        lower_bound = sorted_rp.filter(pl.col("Return_Period") < rp).tail(1)
        higher_bound = sorted_rp.filter(pl.col("Return_Period") > rp).head(1)

        lower_bounds.append(lower_bound)
        higher_bounds.append(higher_bound)

    return lower_bounds, higher_bounds


# Optimized function for return period interpolation
def get_total_aal_for_return_period_optimized(return_period, lower_bound, higher_bound):
    if not lower_bound.is_empty() and lower_bound["Return_Period"][0] == return_period:
        return lower_bound["TOTAL_AAL"][0]

    if not lower_bound.is_empty() and not higher_bound.is_empty():
        lower_rp = lower_bound["Return_Period"][0]
        higher_rp = higher_bound["Return_Period"][0]

        lower_aal = lower_bound["TOTAL_AAL"][0]
        higher_aal = higher_bound["TOTAL_AAL"][0]

        return lower_aal + (higher_aal - lower_aal) * (return_period - lower_rp) / (higher_rp - lower_rp)

    return None      

input_folder_path = config["python"]["input_folder_path"]
# Get all parquet files
files = glob.glob(f"{input_folder_path}/*.parquet")

# Create a lazy query that includes the unique operation
lazy_query = (
    pl.concat([
        pl.scan_parquet(f).select(['LocId', 'EventId', 'SD']) 
        for f in files
    ])
    .unique()
)

# Execute the query with controlled memory usage
df_hazard = lazy_query.collect(streaming=True)

##PET FILES - POLARS DATAFRAME

# Define the folder containing the Parquet files
input_folder_path2 = config["python"]["input_folder_path2"]


# Get the list of all Parquet files in the input folder
files = glob.glob(f"{input_folder_path2}/*.parquet")



# Extract the LocId values from the DataFrame
#locid_list = locid_df['LOCID'].tolist()

# Read and concatenate the Parquet files using lazy evaluation
lazy_frames = [pl.scan_parquet(f) for f in files]
group = pl.concat(lazy_frames)

# # Filter the DataFrame for the unique LocIds
# #filtered_df = group.filter(pl.col('LocId').is_in(locid_list))

# # Collect the result into a DataFrame
df_weight = group.collect()

# # Print the resulting DataFrame
df_weight

liq = config["python"]["liq"]
liq=pl.read_parquet(liq)
vuln_dir = config["python"]["vuln_dir"]
parquet_files = [os.path.join(vuln_dir, path) for path in os.listdir(vuln_dir) if path.endswith('.parquet')]
for file_path in parquet_files:
    basename = os.path.basename(file_path)
    df = pd.read_parquet(file_path)
    #print(f"{basename}: {df.columns.tolist()}")

vgeo = pl.read_parquet(config["python"]["vgeo"])
vcc = pl.read_parquet(config["python"]["vcc"])
vocc = pl.read_parquet(config["python"]["vocc"])
imap = pl.read_parquet(config["python"]["imap"])
vinv = pl.read_parquet(config["python"]["vinv"])
vdc0 = pd.read_parquet(config["python"]["vdc0"])
cghs = pl.read_parquet(config["python"]["cghs"])
vhaz = pd.read_parquet(config["python"]["vhaz"])
AlternateVulnBi4Param_NZ = pd.read_parquet(config["python"]["AlternateVulnBi4Param_NZ"])
Vf0f1_NZ = pd.read_parquet(config["python"]["Vf0f1_NZ"])
VulnBi4Param_NZ = pd.read_parquet(config["python"]["VulnBi4Param_NZ"])
BiModifier_NZ = pd.read_parquet(config["python"]["BiModifier_NZ"])
ConstructionMap_NZ = pl.read_parquet(config["python"]["ConstructionMap_NZ"])
OccupancyMap_NZ = pl.read_parquet(config["python"]["OccupancyMap_NZ"])
alt = pl.read_parquet(config["python"]["AlternateVulnBi4Param_NZ"])
bimod = pl.read_parquet(config["python"]["VulnBi4Param_NZ"])
file_paths = glob.glob(os.path.join(vuln_dir, '*_0_NZ_SD_INDEX.DAT')) + glob.glob(os.path.join(vuln_dir, '*_0_NZ_SD_DATA.DAT'))
parent_vuln_dir = os.path.dirname(vuln_dir)  # Remove 'Ifm' by getting the parent directory
file_paths = glob.glob(os.path.join(parent_vuln_dir, '*_0_NZ_SD_INDEX.DAT')) + glob.glob(os.path.join(parent_vuln_dir, '*_0_NZ_SD_DATA.DAT'))
file_dict = organize_vulnerability_binary_files(file_paths)
subperil_data = {}
haz_types = {'SH' : 0}
coverages = {'structure' : {'coverage': 1, 'cvg_grade': 0}, 'content': {'coverage': 2, 'cvg_grade': 3}}
for subperil, file_paths in file_dict.items():
    data_binary_path, index_binary_path = file_paths
    count, index_df, haz_sev_array, damage_perc_array_all = parse_vulnerability_binary_files(index_binary_path=index_binary_path, data_binary_path=data_binary_path)
    subperil_data[subperil] = {'count': count, 'index_df': index_df, 'haz_sev_array': haz_sev_array, 'damage_perc_array_all': damage_perc_array_all, 'haz_type': haz_types.get(subperil)}

engine = engine_initializer(machine_name, 'NZ_C1T00H1Y0_EDM_part1')

loccvg = pd.read_sql_query('select LOCID,VALUEAMT,* from loccvg', engine)
loc = pd.read_sql_query('select * from loc', engine)
address = pd.read_sql_query('select AddressID,Admin1GeoID from address', engine)
loc['YEARBUILT'] = pd.to_datetime(loc['YEARBUILT'], errors='coerce')
#pd.read_sql_query('select * from loc', engine)['YEARBUILT']
loc_address = loc.merge(address, left_on='AddressID', right_on='AddressID', how='left')
loc_address = pl.from_pandas(loc_address)


# Read the index file and create a Polars DataFrame
path_index = config["python"]["path_index"]
records_index = []
with open(path_index, 'rb') as f1:
    while True:
        try:
            val1 = struct.unpack('I', f1.read(4))[0]
            val2 = struct.unpack('I', f1.read(4))[0]
            val3 = struct.unpack('I', f1.read(4))[0]
            val4 = struct.unpack('I', f1.read(4))[0]
            val5 = struct.unpack('I', f1.read(4))[0]
            records_index.append({'EVENTID': val1, 'QuantileId': val2, 'LOBID': val3, 'Starting_records': val4, 'Num_records': val5})
        except:
            break
pla_index = pl.DataFrame(records_index)

# Read the data file and create a Polars DataFrame
path_data = config["python"]["path_data"]
records_data = []
with open(path_data, 'rb') as f2:
    while True:
        try:
            val1 = struct.unpack('Q', f2.read(8))[0]
            val2 = struct.unpack('f', f2.read(4))[0]
            val3 = struct.unpack('f', f2.read(4))[0]
            val4 = struct.unpack('f', f2.read(4))[0]
            val5 = struct.unpack('f', f2.read(4))[0]
            val6 = struct.unpack('f', f2.read(4))[0]
            val7 = struct.unpack('f', f2.read(4))[0]
            val8 = struct.unpack('f', f2.read(4))[0]
            val9 = struct.unpack('f', f2.read(4))[0]
            val10 = struct.unpack('f', f2.read(4))[0]
            records_data.append({'Admin1GeoID': val1, 'EDS_BDG': val2, 'EDS_CTS': val3, 'EDS_BI': val4, 'CI_BDG': val5, 'CI_CTS': val6, 'CI_BI': val7, 'SC_BDG': val8, 'SC_CTS': val9, 'SC_BI': val10})
        except:
            break
pla_data = pl.DataFrame(records_data)

# Repeat rows in pla_index according to Num_records
repeated_rows = []
for row in pla_index.iter_rows(named=True):
    repeated_rows.extend([row] * row['Num_records'])
repeated_df = pl.DataFrame(repeated_rows)

# Ensure pla_data has the same number of rows as repeated_df
if len(pla_data) != len(repeated_df):
    raise ValueError("The number of rows in pla_data does not match the total number of repeated rows in pla_index.")

# Add the columns EVENTID, QuantileId, and LOBID from repeated_df to pla_data
pla_data = pla_data.with_columns([
    repeated_df['EVENTID'],
    repeated_df['QuantileId'],
    repeated_df['LOBID']
])

# Filter and calculate PLA values
dist_geoid = pla_data.filter(pl.col('QuantileId') == 2)
dist_geoid = dist_geoid.with_columns([
    (pl.col('EDS_BDG') * pl.col('CI_BDG') * pl.col('SC_BDG')).alias('PLA_BDG'),
    (pl.col('EDS_CTS') * pl.col('CI_CTS') * pl.col('SC_CTS')).alias('PLA_CTS'),
    (pl.col('EDS_BI') * pl.col('CI_BI') * pl.col('SC_BI')).alias('PLA_BI')
])
pla = dist_geoid.select(['Admin1GeoID', 'EVENTID', 'QuantileId', 'LOBID', 'PLA_BDG', 'PLA_CTS', 'PLA_BI'])

# Read the occupancy mapping file and create a Polars DataFrame
path_occ = config["python"]["path_occ"]
records_occ = []
with open(path_occ, 'rb') as f1:
    while True:
        try:
            val1 = struct.unpack('I', f1.read(4))[0]
            val2 = struct.unpack('I', f1.read(4))[0]
            records_occ.append({'OCCID': val1, 'LOB': val2})
        except:
            break
pla_occ = pl.DataFrame(records_occ)

# Filter occupancy_pla
occupancy_pla = OccupancyMap_NZ.filter(pl.col('OccupancyId').is_in([12, 20, 28, 36, 15, 23]))
occupancy_pla = occupancy_pla.with_columns([
    pl.col('OccupancyCode').cast(pl.Int32)
])

# # Print the resulting DataFrames
# print(pla_index)
# print(pla_data)
# print(pla)
# print(pla_occ)
# print(occupancy_pla)
#LIST OF COMBINATIONS
comb = pd.read_csv(config["python"]["comb"])
comb = pl.from_pandas(comb)
# Extract the last directory name
last_directory_name = os.path.basename(os.path.normpath(input_folder_path))

# Split the directory name by underscore
split_name = last_directory_name.split('_')

# Get the first word
first_word = split_name[0]
# Filter the DataFrame
comb = comb.filter(pl.col('Combination') == first_word)

df_hazard = df_hazard.lazy()
df_weight = df_weight.lazy()
liq = liq.lazy()
# Initialize an empty list to store the final results
final_results = []

# Get the list of LOCID values directly
locid_list = loc_address['LOCID'].to_list()

# Initialize a list to keep track of processed LOCID values
processed_locid_list = []

# Filter loc_address once and reuse it
loc_filtered = loc_address.lazy().filter(pl.col('LOCID').is_in([88329]))###82196
#loc_filtered = loc_address.lazy().filter(pl.col('LOCID').is_in(locid_list))###82196


# List of return periods to find
return_periods = [100, 250, 500]

# Initialize lists to store intermediate results
locname_list = []
locid_list = []
combination_list = []
edm_list = []
aal_bdg_list = []
aal_cnt_list = []
aal_bi_list = []
score_dict = {rp: [] for rp in return_periods}

# Perform the join of locid_weight2 outside the loop
df_hazard_weight = df_hazard.join(df_weight, left_on='EventId', right_on='EVENTID', how='left')

for idx, row in enumerate(loc_filtered.collect().iter_rows(named=True)):
    processed_locid_list.append(row['LOCID'])
    vgeo_output = find_matching_vgeo_row(row, vgeo)
    filtered_df_hazard = df_hazard_weight.filter(pl.col('LocId') == row['LOCID'])
    filtered_liq = liq.filter(pl.col('LocId') == row['LOCID'])

    # Join with the new data based on EVENTID
    locid_weight_lazy = filtered_df_hazard.join(filtered_liq, left_on=['EventId'], right_on=['EventId'], how='left').unique()

    # Collect the final locid_weight LazyFrame to a DataFrame
    locid_weight = locid_weight_lazy.collect(streaming = False)

    for comb_idx, comb_row in enumerate(comb.iter_rows(named=True)):
        edm_bdg_list = []
        edm_cnt_list = []

        key, values = next(iter(subperil_data.items()))
        count = values.get('count')
        index_df = values.get('index_df')
        haz_sev_array = values.get('haz_sev_array')
        damage_perc_array_all = values.get('damage_perc_array_all')
        haz_type = values.get('haz_type')
        
        for coverage_key, coverage_values in coverages.items():
            coverage = coverage_values['coverage']
            cvg_grade = coverage_values['cvg_grade']

            inv_key = vgeo_output[-2]
            dckey = vgeo_output[-3]
            bikey = vgeo_output[-1]

            vinv_output, cghs_output, bi_occ, occ_id = fetch_vinv_cghs_data(
                row, ConstructionMap_NZ, OccupancyMap_NZ, vcc, vocc, imap, vinv, cghs, 
                inv_key, bikey, coverage, cvg_grade, haz_type, comb_row
            )

            damage_perc_array = get_damage_percentage(
                dckey=dckey,
                vinv_output=pl.DataFrame(vinv_output),
                cghsid=cghs_output,
                index_df=index_df,
                damage_perc_array_all=damage_perc_array_all,
                count=count
            )
            for haz_sev, damage_perc in zip(haz_sev_array, damage_perc_array):
                data = {
                    'Hazard Severity': haz_sev,
                    'Damage Percentage': damage_perc,
                    'Subperil': key,
                    'Coverage': coverage,
                    'Cvg_Grade': cvg_grade,
                    'Hazard Type': haz_type,
                    'DCKey': dckey,
                    'InvKey': inv_key,
                    'BIKey': bikey,
                    'CghsId': cghs_output,
                    'BIOcc': bi_occ,
                    #'OccId': occ_id
                }
                
                if coverage == 1:
                    edm_bdg_list.append(data)
                elif coverage == 2:
                    edm_cnt_list.append(data)

        df_edm_bdg = pl.DataFrame(edm_bdg_list)
        df_edm_cnt = pl.DataFrame(edm_cnt_list)

        locid_weight = locid_weight.with_columns([
            interpolate_series2(
                locid_weight['SD'], 
                df_edm_bdg['Hazard Severity'], 
                df_edm_bdg['Damage Percentage']
            ).alias('DAMAGE_PCT'), ###dfamage Curve for Building
            interpolate_series2(
                locid_weight['SD'], 
                df_edm_cnt['Hazard Severity'], 
                df_edm_cnt['Damage Percentage']
            ).alias('DAMAGE_PCT_CNT'),   ###Damage Curve for Content
            pl.lit(1000000).alias('BDG_EXP')
        ])
        #EVENTLOSS BDG AND CTS
        locid_weight = locid_weight.with_columns([
            (pl.col('DAMAGE_PCT') * pl.col('BDG_EXP') / 100).alias('EVENTLOSS_BDG'),
            (pl.col('DAMAGE_PCT_CNT') * pl.col('BDG_EXP') / 100).alias('EVENTLOSS_CNT')
        ])
        ##LIQ Calculation for BDG and CTS
        locid_weight = locid_weight.with_columns([
            (pl.col('EVENTLOSS_BDG') + pl.col('LIQ_BDG')).alias('EVENT_LIQ_BDG'),
            (pl.col('EVENTLOSS_CNT') + pl.col('LIQ_CNT')).alias('EVENT_LIQ_CTS')
        ])
        ##BI Calculation
        locid_weight = locid_weight.with_columns([
            pl.lit(df_edm_bdg['BIKey'][0]).alias('BIKey'),
            pl.lit(df_edm_bdg['BIOcc'][0]).alias('BIOcc')
        ])

        filtered_bimod = bimod.filter(
            (pl.col('BiKey') == locid_weight['BIKey'][0]) & 
            (pl.col('BiOccupancy') == locid_weight['BIOcc'][0])
        )
        
        filtered_alt = alt.filter((pl.col('BiOccupancy') == locid_weight['BIOcc'][0]) & (pl.col('BiKey') == locid_weight['BIKey'][0]))

        if filtered_alt.height > 0:
            locid_weight = locid_weight.with_columns([
                interpolate_series2(
                    locid_weight['SD'], 
                    filtered_alt['HazSeverity'], 
                    filtered_alt['DamagePct']
                ).alias('DAMAGE_PCT_ALT') ###Damage Curve for BI
            ])

            locid_weight = locid_weight.with_columns([
                (pl.col('DAMAGE_PCT_ALT') * pl.col('BDG_EXP') / 100).alias('EVENTLOSS_BI_ALT')
            ])
        else:
            locid_weight = locid_weight.with_columns([
                pl.lit(0).alias('EVENTLOSS_BI_ALT')
            ])

        # locid_weight = locid_weight.with_columns([
        #     (pl.col('DAMAGE_PCT') * pl.col('BDG_EXP') / 100).alias('EVENTLOSS_BDG'),
        #     (pl.col('DAMAGE_PCT_CNT') * pl.col('BDG_EXP') / 100).alias('EVENTLOSS_CNT')
        # ])

        locid_weight = locid_weight.with_columns([
            interpolate_series2(locid_weight['DAMAGE_PCT'], filtered_bimod['CentralDamageFactor'], filtered_bimod['TimeFor30']).alias('TimeFor30'),
            interpolate_series2(locid_weight['DAMAGE_PCT'], filtered_bimod['CentralDamageFactor'], filtered_bimod['TimeFor60']).alias('TimeFor60'),
            interpolate_series2(locid_weight['DAMAGE_PCT'], filtered_bimod['CentralDamageFactor'], filtered_bimod['TimeFor100']).alias('TimeFor100'),
        ])

        locid_weight = locid_weight.with_columns([
            ((pl.col('TimeFor30') + 0.55 * (pl.col('TimeFor60') - pl.col('TimeFor30')) + 0.2 * (pl.col('TimeFor100') - pl.col('TimeFor60'))) / 365 * pl.col('BDG_EXP')).alias('EVENTLOSS_BI')
        ])

        locid_weight = locid_weight.with_columns([
            pl.when(pl.col('EVENTLOSS_BI') > pl.col('EVENTLOSS_BI_ALT'))
              .then(pl.col('EVENTLOSS_BI'))
              .otherwise(pl.col('EVENTLOSS_BI_ALT'))
              .alias('EVENTLOSS_BI_MAX')
        ])

        

        # locid_weight = locid_weight.with_columns([
        #     (pl.col('EVENTLOSS_BDG') + pl.col('LIQ_BDG')).alias('EVENT_LIQ_BDG'),
        #     (pl.col('EVENTLOSS_CNT') + pl.col('LIQ_CNT')).alias('EVENT_LIQ_CTS'),
        #     (pl.col('EVENTLOSS_BI_MAX') + pl.col('LIQ_BI')).alias('EVENT_LIQ_BI') # no BI for LIQ
        # ])

        locid_weight = calculate_pla_values(locid_weight, row, comb_row, occupancy_pla, pla)

        locid_weight = locid_weight.with_columns([
            (pl.col('EVENT_LIQ_BDG') * pl.col('PLA_BDG')).alias('EVENT_PLA_BDG'),
            (pl.col('EVENT_LIQ_CTS') * pl.col('PLA_CTS')).alias('EVENT_PLA_CTS'),
            (pl.col('EVENTLOSS_BI_MAX') * pl.col('PLA_BI')).alias('EVENT_PLA_BI')
        ])

        
        # RP LOSS CALCULATION
        locid_weight = locid_weight.with_columns([
            (pl.col('EVENT_PLA_BDG') + pl.col('EVENT_PLA_CTS') + pl.col('EVENT_PLA_BI')).alias('TOTAL_AAL')
        ])
        
        rp_result = (
            locid_weight.select(['PERIODID', 'WEIGHT', 'TOTAL_AAL'])
            .sort(['PERIODID', 'TOTAL_AAL'], descending=[False, True])
            .unique(subset=['PERIODID'], keep='first')
            .sort('TOTAL_AAL', descending=True)
            .with_columns([
                pl.col('WEIGHT').cum_sum().alias('EP')
            ])
            .with_columns([
                (1 / pl.col('EP')).alias('Return_Period')
            ])
        )

        lower_bounds, higher_bounds = precompute_bounds(rp_result, return_periods)

        # Compute total_rp_results using precomputed bounds
        total_rp_results = {
            rp: get_total_aal_for_return_period_optimized(rp, lower_bounds[i], higher_bounds[i])
            for i, rp in enumerate(return_periods)
        }

        # Dictionary to store the results
        #total_rp_results = {rp: get_total_aal_for_return_period2(rp, rp_result) for rp in return_periods}

        # Append results to lists
        locname_list.append(row['LOCNAME'])
        locid_list.append(row['LOCID'])
        combination_list.append(comb_row['Combination'])
        #edm_list.append(comb_row['EDM'])
        # aal_bdg_list.append(sum_product_bdg)
        # aal_cnt_list.append(sum_product_cnt)
        # aal_bi_list.append(sum_product_bi_alt)
        aal_bdg_list.append(locid_weight.select((pl.col('EVENT_PLA_BDG') * pl.col('WEIGHT')).sum()).to_series()[0])
        aal_cnt_list.append(locid_weight.select((pl.col('EVENT_PLA_CTS') * pl.col('WEIGHT')).sum()).to_series()[0])
        aal_bi_list.append(locid_weight.select((pl.col('EVENT_PLA_BI') * pl.col('WEIGHT')).sum()).to_series()[0])

        
        for rp in return_periods:
            score_dict[rp].append(total_rp_results[rp])
        # Delete intermediate variables to free up memory
        # del edm_bdg_list, edm_cnt_list, df_edm_bdg, df_edm_cnt, filtered_bimod, filtered_alt, sum_products, sum_product_bdg, sum_product_cnt, sum_product_bi_alt, total_rp_results, rp_result
    print(f"Total LOCID values processed: {len(processed_locid_list)}", end='\r')

# Create the final DataFrame from the lists
final_result_aal_locid = pl.DataFrame({
    "LOCNAME": locname_list,
    "LocId": locid_list,
    "COMBINATION": combination_list,
    #"EDM": edm_list,
    "AAL_BDG": aal_bdg_list,
    "AAL_CNT": aal_cnt_list,
    "AAL_BI": aal_bi_list,
    **{f"Score_{rp}": score_dict[rp] for rp in return_periods}
})

# Collect the LazyFrame to execute the query and get the final DataFrame
#final_result_aal_locid_df = final_result_aal_locid.collect()

# Print the final DataFrame- FINAL FRAMEWORK-RP WORKING 126MIN
#final_result_aal_locid

# Define the output directory and file path
output_dir = config["python"]["output_dir"]
output_file = os.path.join(output_dir, "Final_Results.csv")

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Assuming final_result_aal_locid is a polars DataFrame
final_result_aal_locid.write_csv(output_file)


# # Detach the database using the SQL script
# try:
#     print(f"Detaching database '{database_name}'...")
    
#     # Use the SQL script to detach the database
#     detach_sql = f"""
#     USE [master];
#     EXEC master.dbo.sp_detach_db @dbname = N'{database_name}';
#     """
#     cursor.execute(detach_sql)
#     print(f"Database '{database_name}' detached successfully.")
# except Exception as e:
#     print(f"Error detaching database '{database_name}': {e}")
# finally:
#     # Close the cursor and connection
#     cursor.close()
#     conn.close()
#     print("Database connection closed.")
# Robust database detachment with connection handling and retries
def safe_detach_database(conn, cursor, database_name, max_retries=3):
    """Safely detach a SQL Server database with proper error handling and retries"""
    
    # First check if database exists
    cursor.execute("SELECT name FROM sys.databases WHERE name = ?", database_name)
    if cursor.fetchone() is None:
        print(f"Database '{database_name}' is not attached - nothing to detach")
        return True
    
    print(f"Detaching database '{database_name}'...")
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Set database to SINGLE_USER mode to force disconnect any users
            cursor.execute(f"ALTER DATABASE [{database_name}] SET SINGLE_USER WITH ROLLBACK IMMEDIATE")
            print(f"Database set to SINGLE_USER mode, disconnecting all active users")
            
            # Execute the detach command
            cursor.execute(f"EXEC master.dbo.sp_detach_db @dbname = N'{database_name}', @skipchecks = 'true'")
            
            # Verify the database was actually detached
            cursor.execute("SELECT name FROM sys.databases WHERE name = ?", database_name)
            if cursor.fetchone() is None:
                print(f"Database '{database_name}' detached successfully")
                return True
            else:
                print(f"Warning: Detach command executed but database still appears to be attached")
                retry_count += 1
                
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            
            # Handle specific known errors
            if "database is in use" in error_msg.lower():
                print(f"Retry {retry_count}/{max_retries}: Database still in use, waiting 2 seconds...")
                import time
                time.sleep(2)
            elif "does not exist" in error_msg.lower():
                print(f"Database '{database_name}' does not exist or is already detached")
                return True
            else:
                print(f"Retry {retry_count}/{max_retries}: Error detaching database: {e}")
                if retry_count >= max_retries:
                    print(f"Failed to detach database after {max_retries} attempts")
                    return False
                import time
                time.sleep(1)
    
    return False

# Use the safe detach function
try:
    # Make sure we're in the master database context for detaching
    cursor.execute("USE [master]")
    
    # Call the robust detach function
    detach_result = safe_detach_database(conn, cursor, database_name)
    
    if not detach_result:
        print(f"Warning: Could not properly detach database '{database_name}'")
        
except Exception as e:
    print(f"Error during database detachment process: {e}")
    
finally:
    # Always close connections
    if 'cursor' in locals() and cursor:
        cursor.close()
    if 'conn' in locals() and conn:
        conn.close()
    print("Database connections closed")
