import os
import pandas as pd
import polars as pl
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from huggingface_hub import login, HfApi
from loguru import logger
from datasets import Dataset
from typing import Tuple, List

def load_environment() -> Tuple[str, str]:
    """
    Loads environment variables from the .env file and validates the presence of required tokens.
    
    Returns:
        Tuple[str, str]: A tuple containing (entsoe_token, hf_token).
        
    Raises:
        ValueError: If ENSOE_TOKEN or HUGGINGFACE_TOKEN are missing.
    """
    load_dotenv()
    entsoe_token = os.getenv('ENSOE_TOKEN')
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    
    if not entsoe_token:
        raise ValueError("ENSOE_TOKEN is missing from .env")
    if not hf_token:
        raise ValueError("HUGGINGFACE_TOKEN is missing from .env")
        
    return entsoe_token, hf_token

def get_huggingface_client(hf_token: str) -> HfApi:
    """
    Authenticates with Hugging Face using the provided token and returns the API client.
    
    Args:
        hf_token (str): The access token for Hugging Face Hub.
        
    Returns:
        HfApi: Initialized API client.
    """
    login(token=hf_token)
    return HfApi()

def fetch_entsoe_data(client: EntsoePandasClient, country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Downloads Day Ahead prices for a single country code.
    
    Args:
        client (EntsoePandasClient): The ENTSOE client in use.
        country_code (str): The bidding zone code, such as 'DK_1'.
        start (pd.Timestamp): Extraction start date.
        end (pd.Timestamp): Extraction end date.
        
    Returns:
        pd.DataFrame: DataFrame containing 'time' (flattened index), 'price', and 'zone'.
    """
    logger.info(f"Fetching Day Ahead prices for {country_code} from {start} to {end}...")
    try:
        series = client.query_day_ahead_prices(country_code=country_code, start=start, end=end)
        if series is None or series.empty:
            logger.warning(f"No data returned for {country_code}.")
            return pd.DataFrame()
            
        df = series.to_frame(name="price")
        df.index.name = "time"
        df = df.reset_index()
        df["zone"] = country_code
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {country_code}: {e}")
        return pd.DataFrame()

def fetch_all_zones(client: EntsoePandasClient, zones: List[str], start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """
    Iterates through a list of bidding zones and downloads prices, combining the results into a single DataFrame.
    
    Args:
        client (EntsoePandasClient): The ENTSOE client in use.
        zones (List[str]): List of country codes (e.g., ['DK_1', 'DK_2']).
        start (pd.Timestamp): Start timestamp.
        end (pd.Timestamp): End timestamp.
        
    Returns:
        pd.DataFrame: Concatenated DataFrame. If nothing is found, returns an empty DataFrame.
    """
    dfs = []
    for zone in zones:
        df_zone = fetch_entsoe_data(client=client, country_code=zone, start=start, end=end)
        if not df_zone.empty:
            dfs.append(df_zone)
            
    if not dfs:
        return pd.DataFrame()
        
    return pd.concat(objs=dfs, ignore_index=True)

def convert_to_polars_dataframe(df: pd.DataFrame) -> pl.DataFrame:
    """
    Converts a Pandas DataFrame to a Polars DataFrame, applying the required 
    formatting to time (time_zone='UTC' and time_unit='ms').
    
    Args:
        df (pd.DataFrame): The Pandas DataFrame with a 'time' column.
        
    Returns:
        pl.DataFrame: DataFrame converted to formatted Polars dtypes.
    """
    logger.info("Converting to Polars DataFrame and processing time column...")
    pl_df = pl.from_pandas(data=df)
    
    # Converte il tempo in UTC e specifica la risoluzione in ms
    pl_df = pl_df.with_columns(
        pl.col("time")
        .dt.convert_time_zone("UTC")
        .cast(pl.Datetime(time_unit="ms", time_zone="UTC"))
    )
    return pl_df

def upload_to_huggingface(pl_df: pl.DataFrame, api: HfApi, file_name: str, hf_repo_id: str):
    """
    Prepares a parquet file using the `datasets` library and uploads it to Hugging Face.
    
    Args:
        pl_df (pl.DataFrame): The prepared Polars dataframe with time, price, and zone columns.
        api (HfApi): huggingface_hub API.
        file_name (str): The local and destination file name.
        hf_repo_id (str): The destination repository on the Hub.
    """
    path_in_repo = f"price/{file_name}"
    
    try:
        logger.info(f"Creating HuggingFace Dataset and saving locally as {file_name}...")
        hf_dataset = Dataset.from_polars(pl_df)
        hf_dataset.to_parquet(file_name)
        
        logger.info(f"Uploading to HuggingFace Hub ({hf_repo_id} at {path_in_repo})...")
        api.upload_file(
            path_or_fileobj=file_name,
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            repo_type="dataset"
        )
        logger.success("Upload completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during upload: {e}")
        
    finally:
        if os.path.exists(file_name):
            os.remove(file_name)

def upload_price_data():
    """
    Orchestrator function to extract data from Entsoe (2015-01-01 / 2025-01-01), 
    transform it into a Polars DataFrame, and upload it to Hugging Face Hub as a Parquet file.
    """
    entsoe_token, hf_token = load_environment()
    api = get_huggingface_client(hf_token=hf_token)
    client = EntsoePandasClient(api_key=entsoe_token)
    
    start = pd.Timestamp(ts_input='20150101', tz='Europe/Brussels')
    end = pd.Timestamp(ts_input='20250101', tz='Europe/Brussels')
    
    df_combined = fetch_all_zones(client=client, zones=['DK_1', 'DK_2'], start=start, end=end)
    
    if df_combined.empty:
        logger.error("No data fetched from ENTSOE. Exiting.")
        return
        
    pl_df = convert_to_polars_dataframe(df=df_combined)
    
    hf_repo_id = "clarkmaio/dkppa"
    file_name = "daprice.parquet"
    
    upload_to_huggingface(
        pl_df=pl_df, 
        api=api, 
        file_name=file_name, 
        hf_repo_id=hf_repo_id
    )

if __name__ == "__main__":
    upload_price_data()
