import openml
import os


# Switch to the read-only server
openml.config.server = "http://145.38.195.79/api/v1/xml"

# Set the cache dir (for colab)
openml.config.set_root_cache_directory(os.path.expanduser('openml_cache_dir'))

# Download
dataset = openml.datasets.get_dataset(44317, download_data=True, download_all_files=True)
