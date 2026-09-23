# crcns-downloader
Batch downloader and validator for CRCNS.org

A pair of lightweight Python command-line scripts to batch download files from the CRCNS.org repository and verify their integrity using MD5 checksums.

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## ⚡ Minimal Dependencies

These scripts rely primarily on the **Python Standard Library** and work out of the box on macOS and Linux with no additional packages required.

**Windows users may need one optional package:**
On some Windows installations Python's built-in SSL certificate store may be incomplete, causing HTTPS connections to fail. If you see a certificate-related error, install `certifi`:

```bash
# Windows users only (if you experience SSL certificate errors)
python -m pip install certifi
```

The download script automatically uses `certifi` when available, and falls back to system SSL defaults otherwise. Installing `certifi` is harmless on any platform and may also be useful as a precaution on Windows.

---

## 📁 Repository Structure
* `crcns-download.py` — Initializes datasets and batch downloads files based on user selection.
* `crcns-validate.py` — Computes MD5 hashes of downloaded files and validates them against expected values.
* `crcns-account-template.txt` — A template used to create file `crcns-account.txt` which stores the account information for the CRCNS.org account used to download the files.  If you don't have an account you can [create one for free](https://crcns.org/register).

* `README.md` — Project documentation.
* `LICENSE` — Terms of use (MIT License).
---

## 🚀 Initial Setup
Rename (or copy) file `crcns-account-template.txt` to `crcns-account.txt` and then edit that file to specify the username and password for the account at CRCNS.org used to download the data.  If you don't have an account use the above link to create one for free.


## 🚀 How to use the scripts (Step-by-Step)
Downloading and verifying the files in a dataset involves an initialization phase, an optional step to select which files should be downloaded, and the final steps to download and validate the files.
### Step 1: Initialization
Run the download script and specify a dataset ID. On this first run, the script will download a file (named `filelist.txt`) which has a list of all file in the dataset and then stop so you can you select which files to download.
```bash
python crcns-download.py DATASET_ID
```
Where DATASET_ID is the ID for the dataset at CRCNS.org (for example, `pvc-1` or `hc-2`).

**What this does:**
1. The script creates a local folder named the DATASET_ID.
2. It downloads file  `filelist.txt` into that folder.
3. The script prints instructions to your terminal and exits.

**Command-line options:**

You can use the `-a ACCOUNT_FILE` and `-d OUTPUT_DIR` command-line flags to specify the location of the
`crcns-account.txt` file and the directory within which the folder named DATASET_ID is created.  And you can use
the `--help` or `-h` option to get help about each script.

---
### Step 2: Select Files to download
There are two ways to select which files in the dataset should be downloaded.  The first (recommended for most usage) is by making small edits to file `filelist.txt`.  The second (useful in some cases) is by using the `--pattern` option on the command line.  If neither of these are done all the files are downloaded (because the initial contents of file `filelist.txt` selects all the files).

To use the first method, open the newly downloaded `filelist.txt` file in any text editor. By default, every file is listed with a leading space. 

You can control how files are selected by changing the **Mode** line near the top of the file, allowing you to make minimal edits depending on your needs.

#### Option A: Default Mode (`# mode='default'`)
Use this mode if you want to download **most** files in the dataset.
* **To download a file:** Leave it as-is (with the leading space).
* **To skip a file:** Change the leading space to a `#` to comment it out.
```text
# mode='default'
 data_file_01.csv   <-- Will be downloaded
#data_file_02.csv   <-- Will be SKIPPED
 data_file_03.csv   <-- Will be downloaded
```

#### Option B: Selection Mode (`# mode='+'`)
Use this mode if you only want to download a **few** specific files from a dataset containing many files.
* **To download a file:** Change the leading space to a `+` sign.
* **To skip a file:** Leave it as-is (with the leading space).
```text
# mode='+'
+data_file_01.csv   <-- Will be downloaded
 data_file_02.csv   <-- Will be SKIPPED
 data_file_03.csv   <-- Will be SKIPPED
```

* *(Advanced: You can use multiple `# mode='...'` lines throughout the file to switch selection modes mid-list.)*

If using the second method to select files (which is to use the `--patterns` command line flag) there is no need to edit `filelist.txt`.  Go to the next step (Step 3).

---
### Step 3: Download the files
If using file `filelist.txt` to specify the files to download, run the exact same command you ran in [Step 2](#step-2-select-files-to-download) again. The script reads your customized `filelist.txt` and displays the list of files to be downloaded and the total size and prompts you to confirm to proceed.  If you confirm, the script downloads your selected files.
```bash
python crcns-download.py DATASET_ID
```

If instead you are specifying which files to download using patterns on the command line, 
include the  `-p` or  `--patterns` option followed by the patterns.   This will download all files with a name (full path name) matching any of the specified patterns. Patterns use the glob format.  Examples are `docs/*` (to downlad all files in the docs directory) and `**.pdf` (to download all pdf files regardless of location).  A full path name without wildcards can also be given to download just that file.
```bash
python crcns-download.py DATASET_ID -p PATTERN [PATTERN ...] 
```

---
### Step 4: Verify the Files
Once your downloads are complete, run the validation script with the same dataset ID. It reads the `checksums.md5` file and checks every downloaded file to guarantee nothing was corrupted.
```bash
python crcns-verify.py DATASET_ID
```

The `patterns` option can also be given to specify which files should be verified (instead of veryifying every downloaded file).  This allows verifying some files (but not others) which could be useful if some already-verified data files were exceptionally large or numerous and thus take a long time to unnecessarily verify again.

```bash
python crcns-validate.py DATASET_ID -p PATTERN [PATTERN ...] 
```


---
## 🤝 Contributing
Contributions, bug reports, and feature suggestions are welcome!
1. Fork the repository.
2. Create your feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.
---
## 📄 License
Distributed under the MIT License. See the `LICENSE` file in this repository for more details.

---

## Security Notes

- **Never commit credentials** to version control. The `crcns-account.txt` file is included in `.gitignore` for this reason.
- **Use `chmod 600 crcns-account.txt`** on macOS/Linux to ensure only your user account can read the file.



