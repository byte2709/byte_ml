import subprocess
import sys

# Required libraries
packages = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn"
]

print("Installing ML Lab Dependencies...")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "--upgrade"
] + packages)

print("\nDownloading California Housing Dataset...")

from sklearn.datasets import fetch_california_housing

fetch_california_housing()

print("\nChecking installations...")

import numpy
import pandas
import matplotlib
import seaborn
import sklearn

print("NumPy:", numpy.__version__)
print("Pandas:", pandas.__version__)
print("Matplotlib:", matplotlib.__version__)
print("Seaborn:", seaborn.__version__)
print("Scikit-learn:", sklearn.__version__)

print("\nALL DEPENDENCIES AND DATASETS READY!")