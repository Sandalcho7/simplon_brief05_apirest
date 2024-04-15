import numpy as np
import pandas as pd

from config import NPZ_PATH, MAIN_CSV_PATH



# Conversion
arrays = dict(np.load(NPZ_PATH))
data = {k: [s.decode("utf-8") for s in v.tobytes().split(b"\x00")] if v.dtype == np.uint8 else v for k, v in arrays.items()}
transactions = pd.DataFrame.from_dict(data)

transactions.to_csv(MAIN_CSV_PATH, header=True)