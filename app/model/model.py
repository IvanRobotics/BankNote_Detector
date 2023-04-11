from pydantic import BaseModel
from pathlib import Path
import pickle


__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

with open(f"{BASE_DIR}/BankNote_trained-{__version__}.pkl", "rb") as f:
    classifiers = pickle.load(f)

class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy : float