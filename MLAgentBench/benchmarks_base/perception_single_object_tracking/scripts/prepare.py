# download and preprocess any large-size data/model if needed
import os
import requests
import zipfile

def download_and_unzip(url: str, destination: str):
  """Downloads and unzips a .zip file to a destination.

  Downloads a file from the specified URL, saves it to the destination
  directory, and then extracts its contents.

  If the file is larger than 1GB, it will be downloaded in chunks,
  and the download progress will be displayed.

  Args:
    url (str): The URL of the file to download.
    destination (str): The destination directory to save the file and
      extract its contents.
  """
  if not os.path.exists(destination):
    os.makedirs(destination)

  filename = url.split('/')[-1]
  file_path = os.path.join(destination, filename)

  if os.path.exists(file_path):
    print(f'{filename} already exists. Skipping download.')
    return

  response = requests.get(url, stream=True)
  total_size = int(response.headers.get('content-length', 0))
  gb = 1024*1024*1024

  if total_size / gb > 1:
    print(f'{filename} is larger than 1GB, downloading in chunks')
    chunk_flag = True
    chunk_size = int(total_size/100)
  else:
    chunk_flag = False
    chunk_size = total_size

  with open(file_path, 'wb') as file:
    for chunk_idx, chunk in enumerate(
        response.iter_content(chunk_size=chunk_size)):
      if chunk:
        if chunk_flag:
          print(f"""{chunk_idx}% downloading
          {round((chunk_idx*chunk_size)/gb, 1)}GB
          / {round(total_size/gb, 1)}GB""")
        file.write(chunk)
  print(f"'{filename}' downloaded successfully.")

  with zipfile.ZipFile(file_path, 'r') as zip_ref:
    zip_ref.extractall(destination)
  print(f"'{filename}' extracted successfully.")

  os.remove(file_path)

data_path = '../env/data'

# This is the Eval.ai challenge subset of object tracking annotations
challenge_valid_annot_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sot_valid_annotations_challenge2023.zip'
# download_and_unzip(challenge_valid_annot_url, data_path) # Removed this line

# Define URLs and destination folders
valid_data_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sot_valid_annotations_challenge2023.zip'
test_data_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sot_test_annotations_challenge2023.zip'
valid_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sot_valid_videos_challenge2023.zip'
test_videos_url = 'https://storage.googleapis.com/dm-perception-test/zip_data/sot_test_videos_challenge2023.zip'

dev_folder = os.path.join(data_path, 'dev')
test_folder = os.path.join(data_path, 'test')

# Download and unzip validation data
print("Downloading validation data...")
download_and_unzip(valid_data_url, dev_folder)

# Download and unzip test data
print("Downloading test data...")
download_and_unzip(test_data_url, test_folder)

# # Download and unzip validation videos
# print("Downloading validation videos...")
# download_and_unzip(valid_videos_url, dev_folder)

# # Download and unzip test videos
# print("Downloading test videos...")
# download_and_unzip(test_videos_url, test_folder)

print("All datasets downloaded and extracted successfully.")