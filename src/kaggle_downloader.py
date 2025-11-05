#Download Dataset from Kaggle

import os
import subprocess
from typing import Optional
from dataclasses import dataclass

@dataclass
class KaggleDatasetDownload:

  kgl_dataset: str
  download_dir: str

  def install_kaggle(self):
    try:
      import kagglehub
    except ImportError:
      print("Installing kaggle...")
      subprocess.run(['pip','install','-q', 'kaggle'], check=True)
      import kagglehub

  def kgl_api_auth(self):
    os.makedirs("~/.kaggle/", exists_ok=True)
    kgl_dir = os.path.expanduser("~/.kaggle/")
    json_path = os.path.join(kgl_dir, "kaggle.json")
    try:
      from google.colab import files
      upload = files.upload()
      if "kaggle.json" in upload:
        os.system(f"cp kaggle.json {kgl_dir}")
        os.chmod(json_path, 0o600)
        print("kaggle.json successfully uploaded!!")
      else:
        raise FileNotFoundError("kaggle.json file not uploaded")
    except Exception as e:
      raise RuntimeError("Kaggle API Authentication failed.") from e

  def download(self):
    self.install_kaggle()

    try:
      import kagglehub
      print("Downloading via Kagglehub...")
      os.environ["DISABLE_COLAB_CACHE"] = "True"
      os.environ["KAGGLEHUB_CACHE"] = self.download_dir
      path = kagglehub.dataset_download(self.kgl_dataset)
      print(f"Downloaded Successfully at: {path}")
      return path
    except Exception as e:
      print("Downloading failed, trying via Kaggle API")
      print(e)
      self.kgl_api_auth()

    print(f"Downloading {self.kgl_dataset}...")
    command = ["kaggle","datasets","download","-d",self.kgl_dataset,
               "-p",self.download_dir,"--unzip"]
    subprocess.run(command, check=True)
    print(f"Dataset downloaded and extracted in: {self.download_dir}")
    return os.path.abspath(self.download_dir)
