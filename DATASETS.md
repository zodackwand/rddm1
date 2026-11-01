# Raw datasets on the cluster

Downloaded by `download.py` into `datasets/<name>/raw/` (`datasets/` is not tracked by git).
The listings are partial: they come from `find -maxdepth 3 | head -25`.

| Dataset | Size | Files | Format |
|---|---|---|---|
| BIDMC | 419M | 486 | WFDB records: `.dat`, `.hea`, `.breath` |
| CAPNO | 192M | 340 | CSV per recording |
| DALIA | 5.4G | 4 | `data.zip` inside `ppg_dalia.zip`, not extracted yet |
| MIMIC-AFib | 37M | 2 | `.mat` |
| WESAD | 19G | 78 | One folder per subject, `.pkl` |

## BIDMC

```
raw/bidmc.zip
raw/bidmc-ppg-and-respiration-dataset-1.0.0/
    RECORDS
    bidmcNN.dat, bidmcNN.hea, bidmcNN.breath
    bidmcNNn.dat, bidmcNNn.hea
```

## CAPNO

```
raw/README.txt
raw/MANIFEST.TXT
raw/capnobase.zip
raw/data/csv/
    NNNN_8min_signal.csv
    NNNN_8min_labels.csv
    NNNN_8min_reference.csv
    NNNN_8min_param.csv
    NNNN_8min_meta.csv
    NNNN_8min_SFresults.csv
```

## DALIA

```
raw/readme.pdf
raw/ppg_dalia.zip
raw/data.zip
```

## MIMIC-AFib

```
raw/mimic_perform_af_data.mat
raw/mimic_perform_non_af_data.mat
```

## WESAD

```
raw/WESAD/SN/
    SN.pkl
    SN_E4_Data.zip
    SN_quest.csv
    SN_readme.txt
    SN_respiban.txt
```
