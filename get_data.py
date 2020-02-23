import os
import urllib.request

DATASET_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter/train.zip"
DATA_DIR = "data"
DATA_FILENAME = "train.zip"

if __name__ == "__main__":

    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    req = urllib.request.urlopen(DATASET_URL)
    data = req.read()

    with open(os.path.join(DATA_DIR, DATA_FILENAME), "wb") as f:
        f.write(data)
