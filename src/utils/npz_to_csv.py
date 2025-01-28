import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("DATA_DIR")
NPZ_FILE = os.getenv("NPZ_FILE")
MAIN_CSV_FILE = os.getenv("MAIN_CSV_FILE")

# Conversion
arrays = dict(np.load(DATA_DIR + NPZ_FILE))
data = {
    k: (
        [s.decode("utf-8") for s in v.tobytes().split(b"\x00")]
        if v.dtype == np.uint8
        else v
    )
    for k, v in arrays.items()
}
transactions = pd.DataFrame.from_dict(data)

# Save to CSV
transactions.to_csv(DATA_DIR + MAIN_CSV_FILE, header=True)
