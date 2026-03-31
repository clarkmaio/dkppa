
import polars as pl
from lpspline import f, cs, bs, l
from lpspline.viz import plot_diagnostic
import matplotlib.pyplot as plt
from loguru import logger
import os
from huggingface_hub import HfApi, login
from lpspline import LpRegressor


def load_daregression_data(years: list[int] = range(2018, 2025)):
    logger.info("Loading daprice data from HuggingFace...")
    price_url = "https://huggingface.co/datasets/clarkmaio/dkppa/resolve/main/price/daprice.parquet"
    price = pl.read_parquet(price_url)

    logger.info(f"Loading and processing ERA5 weather data {years}...")
    weather_urls = [f"https://huggingface.co/datasets/clarkmaio/dkppa/resolve/main/era5/dk_era5_{year}.parquet" for year in years]
    weather = pl.scan_parquet(weather_urls).with_columns(
        s100 = (pl.col('u100')**2 + pl.col('v100')**2).sqrt(),
        time = pl.col('time').dt.cast_time_unit('ms').dt.replace_time_zone('UTC')
    ).select('time', 's100').group_by('time').agg(pl.col('s100').mean()).collect().sort('time')

    logger.info("Merging weather and price data...")
    df = price.join(weather, on='time')

    logger.info("Computing daily average...")
    df_daily = df.with_columns(
        time = pl.col('time').dt.date()
    ).group_by('time', 'zone').agg(
        pl.col('price').mean(),
        pl.col('s100').mean()
    ).sort('time')

    logger.info("Computing calendar features...")
    df_daily = df_daily.with_columns(
        ordinal_day = pl.col('time').dt.ordinal_day(),
        weekday = pl.col('time').dt.weekday(),
        year = pl.col('time').dt.year(),
    )
    
    logger.info("Filtering for zone DK_1...")
    return df_daily.filter(pl.col('zone') == 'DK_1')


def fit_daregression(years: list[int] = range(2018, 2025)):
    logger.info("Starting model fitting process for DA regression...")
    df = load_daregression_data(years)

    logger.info("Initializing lpspline model...")
    estimatordaily = (
        +cs('ordinal_day', order=5)
        +f('weekday')
        +f('year')
        +l('s100')
    )

    logger.info("Fitting the model on daily data...")
    estimatordaily.fit(df, df['price'])
    
    logger.info("Saving fitted model to 'estimatordaily.pkl'...")
    estimatordaily.save("estimatordaily.pkl")

    logger.info("Generating and saving diagnostic plots...")
    plot_diagnostic(model=estimatordaily, X=df, y=df['price'], ncols=2)
    plt.savefig("daregression_diagnostic.png")
    logger.info("Model fitting and diagnostics completed.")



def generate_daprice_scenario(baseline_year: int = 2024, years: list[int] = range(2018, 2025)):
    logger.info("Initializing HuggingFace API for scenario generation...")
    hf_token = os.getenv('HUGGINGFACE_TOKEN')
    if not hf_token:
        from dotenv import load_dotenv
        load_dotenv()
        hf_token = os.getenv('HUGGINGFACE_TOKEN')
        if not hf_token:
            logger.error("HUGGINGFACE_TOKEN not found in environment.")
            return

    login(token=hf_token, add_to_git_credential=False)
    api = HfApi()

    logger.info("Loading trained LpRegressor model (estimatordaily.pkl)...")
    try:
        model = LpRegressor.load("estimatordaily.pkl")
    except Exception as e:
        logger.error(f"Failed to load estimatordaily.pkl: {e}")
        return

    hf_repo_id = "clarkmaio/dkppa"

    for year in years:
        logger.info(f"Processing scenario for year {year}...")
        url = f"https://huggingface.co/datasets/clarkmaio/dkppa/resolve/main/era5/dk_era5_{year}.parquet"
        file_name = f"scenario_{year}.parquet"
        
        try:
            weather = pl.read_parquet(url).with_columns(
                s100 = (pl.col('u100')**2 + pl.col('v100')**2).sqrt(),
                time = pl.col('time').dt.cast_time_unit('ms').dt.replace_time_zone('UTC')
            )
            
            # Spatial average (grouping by time to average across all lat/lons for the snapshot)
            weather = weather.group_by('time').agg(pl.col('s100').mean()).sort('time')
            
            # Daily average
            weather_daily = weather.with_columns(
                time = pl.col('time').dt.date()
            ).group_by('time').agg(pl.col('s100').mean()).sort('time')
            
            # Calendar features with 'year' strictly overridden by baseline_year
            weather_daily = weather_daily.with_columns(
                ordinal_day = pl.col('time').dt.ordinal_day(),
                weekday = pl.col('time').dt.weekday(),
                year = pl.lit(baseline_year)
            )
            
            # Predict the scenario using the model
            predicted = model.predict(weather_daily)
            if hasattr(predicted, "to_numpy"):
                predicted = predicted.to_numpy()
                
            # Format output dataframe
            out_df = weather_daily.with_columns(
                price = pl.Series(predicted),
                scenario = pl.lit(year)
            ).select(['time', 'price', 'scenario'])
            
            # Save local parquet
            out_df.write_parquet(file_name)
            
            # Upload to HuggingFace
            path_in_repo = f"price/scenario/{file_name}"
            logger.info(f"Uploading {file_name} to HuggingFace Hub ({hf_repo_id} at {path_in_repo})...")
            api.upload_file(
                path_or_fileobj=file_name,
                path_in_repo=path_in_repo,
                repo_id=hf_repo_id,
                repo_type="dataset"
            )
            logger.success(f"Scenario {year} uploaded successfully.")
            
        except Exception as e:
            logger.error(f"Error processing scenario for year {year}: {e}")
            
        finally:
            if os.path.exists(file_name):
                os.remove(file_name)


if __name__ == '__main__':
    #fit_daregression(years = [2018, 2019, 2023, 2024])
    generate_daprice_scenario(baseline_year=2024, years = [1980, 1981, 1982])
