

import numpy as np




def normalized_logistic(x, saturation=30, steep=0.5):
    """
    Implements a normalized logistic curve where f(0)=0 and f(C)=1.

    Formula
    \frac{\frac{1}{1+e^{-k(\frac{x}{C}-M)}}-\frac{1}{1+e^{Mk}}}{\frac{1}{1+e^{-Mk}}-\frac{1}{1+e^{Mk}}}\cdot C
    
    Args:
        x (float or np.array): Input value(s).
        saturation (C): When x = C, the output is 1.
        steep (k): The steepness of the S-curve.
    """
    C = saturation
    k = steep
    M = 0.5
    
    num = (1 / (1 + np.exp(-k * (x / C - M)))) - (1 / (1 + np.exp(M * k)))
    den = (1 / (1 + np.exp(-M * k))) - (1 / (1 + np.exp(M * k)))
    return num / den


def download_era5_dataset():
    """
    Downloads all parquet files from the Hugging Face dataset "clarkmaio/dkppa/era5"
    and saves them locally in the "dataset/era5" folder.
    """
    from huggingface_hub import snapshot_download
    from loguru import logger
    import os
    
    os.makedirs('dataset/era5', exist_ok=True)
    logger.info("Downloading era5 dataset from Hugging Face...")
    snapshot_download(
        repo_id="clarkmaio/dkppa",
        repo_type="dataset",
        local_dir="dataset",
        allow_patterns="era5/*.parquet",
        local_dir_use_symlinks=False
    )
    logger.success("Download complete.")


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')

    x = np.linspace(0, 35, 1000)
    y = normalized_logistic(x, saturation=30, steep=-10)
    plt.plot(x, y)
    plt.savefig('test.png')
    plt.show(block=True)
