import os
import pandas as pd
import xarray as xr
from pathlib import Path

def process_netcdf_files():
    # Define paths
    raw_data_dir = Path("raw")
    processed_dir = Path("processed")
    output_file = processed_dir / "indian_ocean_argo.csv"
    
    # Create processed directory if it doesn't exist
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    
    # Process each NetCDF file
    for nc_file in raw_data_dir.glob("*.nc"):
        try:
            # Open the NetCDF file
            ds = xr.open_dataset(nc_file)
            
            # Convert to pandas DataFrame
            df = ds.to_dataframe().reset_index()
            
            # Add file source for reference
            df['source_file'] = nc_file.name
            
            all_data.append(df)
            print(f"Processed {nc_file.name}")
            
        except Exception as e:
            print(f"Error processing {nc_file.name}: {str(e)}")
    
    if not all_data:
        print("No data was processed. Check if there are .nc files in the raw data directory.")
        return
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    print(f"\nSuccessfully processed {len(all_data)} files.")
    print(f"Combined data saved to: {output_file.absolute()}")
    
    # Print some basic info about the data
    print("\nData Summary:")
    print(f"Total rows: {len(combined_df)}")
    print("\nFirst few rows:")
    print(combined_df.head())
    print("\nColumns:", combined_df.columns.tolist())

if __name__ == "__main__":
    process_netcdf_files()
