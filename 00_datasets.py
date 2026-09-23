#!/usr/bin/env python
# This script downloads the datasets we need from the filelist.
# It is mainly a wrapper on top of `crcns-downloader` and tar


import subprocess
from pathlib import Path
from glob import glob
import tarfile
import os


DOWN_LIB = Path("./crcns-downloader/")
DOWNLOADER = "crcns-download.py"
ACCOUNT_FILE = DOWN_LIB / "crcns-account.txt"
FILE_LIST = Path("filelist.txt")
DATA_OUT = Path("data_raw")
DATASET = "hc-11"


def download():
    if not ACCOUNT_FILE.exists():
        raise ValueError(
            f"You need to create an account and create a file {ACCOUNT_FILE}"
        )
    Path.mkdir(DATA_OUT / DATASET, exist_ok=True)
    subprocess.run(["cp", str(FILE_LIST), str(DATA_OUT / DATASET / FILE_LIST)])
    subprocess.run(
        ["python", str(DOWNLOADER), DATASET, "-d", DATA_OUT.absolute()],
        cwd=DOWN_LIB,
        check=True,
    )


def _only_xml_filter(members):
    for tarinfo in members:
        if os.path.splitext(tarinfo.name)[1] in [".xml", ".mat"]:
            print(f"found metadata {tarinfo.name}")
            yield tarinfo


def extract():
    print("------------------------------------------------")
    print(" Extraction will take a long time, please wait ")
    print("------------------------------------------------")
    for file in glob(str(DATA_OUT / DATASET / "data" / "*.tar.gz")):
        print(f"Extracting {file}...")
        tf = tarfile.open(file)
        if "eeg" not in file:
            tf.extractall(DATA_OUT, filter="data", members=_only_xml_filter(tf))
        else:
            tf.extractall(DATA_OUT, filter="data")
        tf.close()
        print("Extraction complete")


if __name__ == "__main__":
    print(
        "This script will download and extract the datasets. Depending on your Internet and hard drive speed, it may take a very long time."
    )
    download()
    extract()
