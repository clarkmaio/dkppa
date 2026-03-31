import os
import arraylake
import xarray as xr
import polars as pl
from dotenv import load_dotenv
from datetime import datetime
from datasets import Dataset
from huggingface_hub import login, HfApi
from typing import Optional, Dict
from loguru import logger

def load_environment() -> Dict[str, str]:
    """
    Loads environment variables and ensures required tokens are present.
    
    Returns:
        Dict[str, str]: A dictionary containing 'HUGGINGFACE_TOKEN' and 'EARTHMOVER_TOKEN'.
        
    Raises:
        ValueError: If either HUGGINGFACE_TOKEN or EARTHMOVER_TOKEN is missing from the environment.
    """
    load_dotenv()
    
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN is not set in the environment.")
        
    em_token = os.getenv('EARTHMOVER_TOKEN')
    if not em_token:
        raise ValueError("EARTHMOVER_TOKEN is not set in the environment.")
        
    return {
        "HUGGINGFACE_TOKEN": hf_token,
        "EARTHMOVER_TOKEN": em_token
    }

def get_huggingface_client(hf_token: str) -> HfApi:
    """
    Authenticates with Hugging Face and returns the API client.
    
    Args:
        hf_token (str): The Hugging Face authentication token.
        
    Returns:
        HfApi: An authenticated Hugging Face Hub API client.
    """
    login(token=hf_token)
    return HfApi()

def get_earthmover_dataset(em_token: str) -> xr.Dataset:
    """
    Connects to Earthmover via Arraylake and returns the ECMWF ERA5 temporal Zarr dataset.
    
    Args:
        em_token (str): The Earthmover authentication token.
        
    Returns:
        xr.Dataset: The Xarray dataset connected to the Zarr store for ERA5 weather data.
    """
    logger.info("Connecting to Earthmover...")
    client = arraylake.Client(token=em_token)
    
    repo_id = "myexperiment/ecmwf-era5"
    repo = client.get_repo(repo_id)
    session = repo.readonly_session(branch="main")
    
    logger.info("Opening Zarr dataset (temporal group)...")
    return xr.open_zarr(session.store, zarr_format=3, group='temporal')

def get_perimeter() -> Dict[str, float]:
    """
    Extracts bounding box geographic coordinates from environment variables.
    
    Returns:
        Dict[str, float]: A dictionary containing numerical limits 'lat_min', 'lat_max',
                          'lon_min', and 'lon_max'.
                          
    Raises:
        ValueError: If any expected coordinate is missing from the environment.
    """
    try:
        return {
            "lat_min": float(os.environ['MIN_LAT']),
            "lat_max": float(os.environ['MAX_LAT']),
            "lon_min": float(os.environ['MIN_LON']),
            "lon_max": float(os.environ['MAX_LON'])
        }
    except KeyError as e:
        raise ValueError(f"Missing perimeter coordinate in dot env: {e}")

def process_year(year: int, ds: xr.Dataset, perimeter: Dict[str, float]) -> Optional[pl.DataFrame]:
    """
    Slices the global dataset for a specific year and perimeter, returning a Polars DataFrame.
    
    Args:
        year (int): The target year to process data for.
        ds (xr.Dataset): The original Xarray weather dataset representing global/overall temporal data.
        perimeter (Dict[str, float]): The geographic bounding limits specifying latitude and longitude to slice.
        
    Returns:
        Optional[pl.DataFrame]: A flattened Polars DataFrame of 'u100' and 'v100' variables for the 
                                specified year and boundary. Returns None if no data is found for that period.
    """
    sliced_ds = ds.sel(
        latitude=slice(perimeter["lat_max"], perimeter["lat_min"]),
        longitude=slice(perimeter["lon_min"], perimeter["lon_max"]),
        time=slice(datetime(year, 1, 1), datetime(year, 12, 31, 23))
    )[['u100', 'v100']]
    
    sliced_df = sliced_ds.to_dataframe()
    
    if sliced_df.empty:
        return None
        
    return pl.from_pandas(sliced_df.reset_index())

def upload_dataframe_to_hf(df_pl: pl.DataFrame, year: int, api: HfApi, hf_repo_id: str) -> None:
    """
    Prepares and uploads a given Polars DataFrame to Hugging Face Hub as a Parquet file.
    
    Args:
        df_pl (pl.DataFrame): The Polars DataFrame containing the sliced weather data.
        year (int): The corresponding year, used for dynamically naming the Parquet file.
        api (HfApi): An authenticated `HfApi` reference used for pushing files to the Hub.
        hf_repo_id (str): The destination Hugging Face repository ID (e.g., 'username/datasetName').
    """
    file_name = f"dk_era5_{year}.parquet"
    path_in_repo = f"era5/{file_name}"
    
    try:
        logger.info(f"Creating HuggingFace Dataset for year {year}...")
        hf_dataset = Dataset.from_polars(df_pl)
        
        # Save locally as Parquet
        hf_dataset.to_parquet(file_name)
        
        logger.info(f"Uploading {file_name} to HuggingFace Hub ({hf_repo_id} at {path_in_repo})...")
        api.upload_file(
            path_or_fileobj=file_name,
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        logger.success(f"Year {year} uploaded successfully.")
        
    finally:
        # Cleanup local file handling
        if os.path.exists(file_name):
            os.remove(file_name)

def upload_weather_data(start_year: int = 1980, end_year: int = 2024) -> None:
    """
    Main orchestration function controlling the ETL pipeline of weather data.
    Logs into required APIs, pulls coordinate limits, filters datasets iteratively 
    year-by-year, and exports files into a Hugging Face dataset repository.
    
    Args:
        start_year (int, optional): Processing span starting year. Defaults to 1980.
        end_year (int, optional): Processing span end year. Defaults to current system year.
    """
    logger.info("Initializing configuration and connections...")
    env = load_environment()
    api = get_huggingface_client(env["HUGGINGFACE_TOKEN"])
    ds = get_earthmover_dataset(env["EARTHMOVER_TOKEN"])
    perimeter = get_perimeter()
    
    hf_repo_id = "clarkmaio/dkppa"
    
    logger.info(f"Starting weather data extraction from {start_year} to {end_year}")
    
    for year in range(start_year, end_year + 1):
        logger.info(f"--- Processing year {year} ---")
        try:
            df_pl = process_year(year, ds, perimeter)
            
            if df_pl is None:
                logger.warning(f"No data found for year {year}. Skipping.")
                continue
                
            upload_dataframe_to_hf(df_pl, year, api, hf_repo_id)
            
        except Exception as e:
            logger.error(f"Error processing year {year}: {e}")

if __name__ == "__main__":
    upload_weather_data(start_year=1980, end_year=2024)
