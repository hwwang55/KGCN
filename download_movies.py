import requests
import os
import zipfile
import tarfile
from tqdm import tqdm
import shutil


dataset_dir = "resources"

# File name : url
dataset_urls = {'ml-20m.zip': 'http://files.grouplens.org/datasets/movielens/ml-20m.zip'}

def extract(file_name):
  name_stripped, extension = os.path.splitext(file_name)
  if extension == '.zip':
      with zipfile.ZipFile(file_name, 'r') as zip_ref:
          zip_ref.extractall(os.path.join(dataset_dir, name_stripped))
      print("ZIP extraction complete.")
  elif file_name.endswith('.tar.gz'):
      with tarfile.open(file_name, 'r:gz') as tar:
          tar.extractall(os.path.join(dataset_dir, os.path.splitext(name_stripped)[0]))
      print("TAR.GZ extraction complete.")
  else:
      print(f"Unsupported file extension {extension}.")


if __name__ == "__main__":
  # Download the file
  for file_name, url in dataset_urls.items():
    print(f"Downloading file {file_name} from {url}")
    response = requests.get(url, stream=True)

    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)

    with open(file_name, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()

    print(f"File '{file_name}' downloaded successfully.")
    print(f"Unpacking file {file_name}")
    extract(file_name)

    shutil.move("ml-20m/ml-20m/ratings.csv", "data/movie/")