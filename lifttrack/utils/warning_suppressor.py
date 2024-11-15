import os
import warnings
from cryptography.utils import CryptographyDeprecationWarning

def suppress_warnings():
    """Suppresses specific warnings to clean up the log output."""
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning) 