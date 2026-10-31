from pathlib import Path

ROOT = Path(__file__).resolve().parent / "datasets"

DATASETS = ["BIDMC", "CAPNO", "DALIA", "MIMIC-AFib", "WESAD"]

# Raw files go to datasets/<DATASET_NAME>/raw/
SOURCES = {
    "WESAD": [
        ("WESAD.zip", "https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download"),
    ],
    "DALIA": [
        ("ppg_dalia.zip", "https://archive.ics.uci.edu/static/public/495/ppg+dalia.zip"),
    ],
    "CAPNO": [
        ("capnobase.zip", "https://borealisdata.ca/api/access/dataset/:persistentId/"
                          "?persistentId=doi:10.5683/SP2/NLB8IT&format=original"),
    ],
    "BIDMC": [
        ("bidmc.zip", "https://physionet.org/content/bidmc/get-zip/1.0.0/"),
    ],
    "MIMIC-AFib": [
        ("mimic_perform_af_data.mat",
         "https://zenodo.org/records/6973963/files/mimic_perform_af_data.mat?download=1"),
        ("mimic_perform_non_af_data.mat",
         "https://zenodo.org/records/6973963/files/mimic_perform_non_af_data.mat?download=1"),
    ],
}


def get_raw_dir(name):
    return ROOT / name / "raw"
