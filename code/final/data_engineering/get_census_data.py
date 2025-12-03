import censusdis.data as ced
import pandas as pd
import os 

# --- Configuration ---
# The year range for the American Community Survey 5-Year Estimates (ACS5).
START_YEAR = 2009
END_YEAR = 2018
# File name for the food insecurity data provided by the user
FOOD_INSECURITY_FILE = "data/MMG_county_df_clean.csv"

# ACS Detailed Table Variables (Estimates - suffix 'E')
ECONOMIC_VARIABLES = [
    # 1. Median Household Income (Key indicator of financial resources)
    "B19013_001E",

    # 2. Poverty Status (Total Population for whom poverty status is determined)
    "B17001_001E",
    "B17001_002E", # Estimate of Population Below Poverty Level

    # 3. Employment Status (Indicator of labor market vulnerability)
    "B23025_001E", # Total Population 16 years and over
    "B23025_007E", # Unemployed Population 16 years and over

    # 4. Supplemental Nutrition Program (SNAP) Receipts (Indicator of aid dependency)
    "B22003_001E", # Total Households
    "B22003_002E", # Households receiving SNAP benefits
]

def download_economic_data():
    """
    Downloads key economic variables from the ACS 5-Year Estimates for all US counties
    for the period 2009 through 2018. The data is combined into a single DataFrame.
    """
    print(f"Starting download of ACS 5-Year economic data for vintages {START_YEAR} to {END_YEAR}...")

    all_data_frames = []

    for vintage in range(START_YEAR, END_YEAR + 1):
        print(f"-> Downloading data for vintage: {vintage} (ACS 5-Year estimates ending in {vintage})...")
        try:
            # The core censusdis download function.
            df = ced.download(
                dataset="acs/acs5",
                vintage=vintage,
                download_variables=ECONOMIC_VARIABLES,
                state="*",
                county="*",
                with_geometry=False 
            )

            # Add the vintage year to the DataFrame before appending
            df['YEAR'] = vintage
            all_data_frames.append(df)
            print(f"   Successfully downloaded {len(df)} county records for {vintage}.")

        except Exception as e:
            print(f"   ERROR: Could not download data for vintage {vintage}: {e}")
            continue

    if not all_data_frames:
        print("\nFATAL: No data frames were successfully downloaded.")
        return None

    # Concatenate all year-specific DataFrames into one
    master_df = pd.concat(all_data_frames, ignore_index=True)
    
    print("\n--- Data Cleaning and Feature Engineering ---")

    # Clean up column names for easier ML use
    master_df.rename(columns={
        'B19013_001E': 'MEDIAN_HOUSEHOLD_INCOME',
        'B17001_001E': 'POP_POVERTY_DETERMINED',
        'B17001_002E': 'POP_BELOW_POVERTY',
        'B23025_001E': 'POP_16_PLUS',
        'B23025_007E': 'POP_UNEMPLOYED',
        'B22003_001E': 'HOUSEHOLDS_TOTAL',
        'B22003_002E': 'HOUSEHOLDS_SNAP',
        'NAME': 'GEOGRAPHY_NAME',
        'STATE': 'STATE_FIPS',
        'COUNTY': 'COUNTY_FIPS'
    }, inplace=True)

    # --- FIX FOR FIPS FORMATTING ---
    # Convert FIPS codes to strings with leading zeros for proper joining
    # STATE FIPS is 2 digits, COUNTY FIPS is 3 digits.
    # We must ensure they are treated as strings before concatenation.
    
    # Check if FIPS columns exist and are not already strings (they are often int/float from the API)
    if 'STATE_FIPS' in master_df.columns and 'COUNTY_FIPS' in master_df.columns:
        # 1. Standardize the format: convert to string, then pad with zeros
        master_df['STATE_FIPS'] = master_df['STATE_FIPS'].astype(str).str.zfill(2)
        master_df['COUNTY_FIPS'] = master_df['COUNTY_FIPS'].astype(str).str.zfill(3)

        # 2. Create the full 5-digit FIPS code
        master_df['FIPS'] = master_df['STATE_FIPS'] + master_df['COUNTY_FIPS']
    else:
        print("WARNING: STATE_FIPS or COUNTY_FIPS columns missing. FIPS creation skipped.")
        return None # Return None if critical FIPS columns are missing

    # Feature Engineering: Calculate rates
    # The .loc[:, ...] is used for safe assignment on a DataFrame
    master_df.loc[:, 'POVERTY_RATE'] = (
        master_df['POP_BELOW_POVERTY'] / master_df['POP_POVERTY_DETERMINED']
    ) * 100
    
    master_df.loc[:, 'UNEMPLOYMENT_RATE'] = (
        master_df['POP_UNEMPLOYED'] / master_df['POP_16_PLUS']
    ) * 100
    
    master_df.loc[:, 'SNAP_RECEIPT_RATE'] = (
        master_df['HOUSEHOLDS_SNAP'] / master_df['HOUSEHOLDS_TOTAL']
    ) * 100
    
    # Set the MultiIndex using the standardized FIPS column
    master_df.set_index(['YEAR', 'FIPS'], inplace=True)

    print(f"\nSuccessfully created Census Economic DataFrame with {len(master_df)} total records.")
    
    return master_df

def combine_dataframes(economic_df: pd.DataFrame, fi_file_path: str) -> pd.DataFrame:
    """
    Loads the food insecurity data and merges it with the economic data.

    Args:
        economic_df: The DataFrame of Census economic variables.
        fi_file_path: The file path to the food insecurity CSV.

    Returns:
        A combined DataFrame for ML modeling.
    """
    if economic_df is None:
        print("ERROR: Cannot combine dataframes. Economic data is None.")
        return None

    if not os.path.exists(fi_file_path):
        print(f"ERROR: Food insecurity file not found at path: {fi_file_path}")
        return None

    print(f"\n--- Combining DataFrames with {fi_file_path} ---")

    # 1. Load the food insecurity data
    fi_df = pd.read_csv(fi_file_path)

    # 2. Prepare the food insecurity DataFrame keys
    fi_df.rename(columns={'Year': 'YEAR'}, inplace=True)
    
    # --- FIX FOR CSV FIPS FORMATTING ---
    # Ensure the CSV FIPS is also a 5-digit zero-padded string for reliable merge
    if 'FIPS' in fi_df.columns:
        # Convert to integer first (to handle float representation like 1001.0), then to 5-digit string
        fi_df['FIPS'] = fi_df['FIPS'].astype(int).astype(str).str.zfill(5)
    else:
        print("WARNING: FIPS column missing in the food insecurity CSV.")
        return None

    # Set index for the food insecurity data for cleaner merging
    fi_df.set_index(['YEAR', 'FIPS'], inplace=True)

    print(f"Food Insecurity DataFrame loaded with {len(fi_df)} records.")
    print("Food Insecurity DataFrame index and columns ready for merge.")

    # 3. Perform the merge
    # 'inner' merge ensures we only have matched (Year, FIPS) pairs, which is usually best for ML training
    combined_df = economic_df.merge(
        fi_df,
        left_index=True,
        right_index=True,
        how='inner' 
    )

    print("\n--- Combined Master DataFrame Summary ---")
    print(f"Successfully combined data. Total records after merge: {len(combined_df)}")
    print("Combined DataFrame Info:")
    combined_df.info()

    print("\nCombined DataFrame Head:")
    print(combined_df.head())

    return combined_df

if __name__ == "__main__":
    # 1. Download and process the Census economic data
    economic_data_df = download_economic_data()

    # 2. Combine the Census data with the user's Food Insecurity CSV
    final_combined_df = combine_dataframes(economic_data_df, FOOD_INSECURITY_FILE)

    if final_combined_df is not None:
        final_combined_df.to_csv('data.csv')
        print("\nFinal combined DataFrame is ready for your ML model training.")