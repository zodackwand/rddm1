#!/bin/bash
unzip -q -n datasets/DALIA/raw/data.zip -d datasets/DALIA/raw/
find datasets/DALIA/raw -maxdepth 3 | head -15

cat datasets/BIDMC/raw/bidmc-ppg-and-respiration-dataset-1.0.0/bidmc01.hea
head -2 datasets/CAPNO/raw/data/csv/0105_8min_signal.csv | cut -c1-300
ls datasets/CAPNO/raw/data/csv/*_signal.csv | wc -l
ls datasets/WESAD/raw/WESAD | tr '\n' ' '; echo

.venv/bin/python - <<'EOF'
import glob, pickle, h5py, scipy.io

def show(obj, indent=0):
    for k, v in obj.items():
        if isinstance(v, dict):
            print(" " * indent + str(k)); show(v, indent + 2)
        else:
            print(" " * indent + str(k), getattr(v, "shape", type(v).__name__))

for name, pattern in [("WESAD", "datasets/WESAD/raw/WESAD/S3/S3.pkl"),
                      ("DALIA", "datasets/DALIA/raw/**/S1.pkl")]:
    path = glob.glob(pattern, recursive=True)[0]
    print("===", name, path)
    show(pickle.load(open(path, "rb"), encoding="latin1"))

for path in glob.glob("datasets/MIMIC-AFib/raw/*.mat"):
    print("===", path)
    try:
        m = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
        for k, v in m.items():
            if k.startswith("__"):
                continue
            print(k, getattr(v, "shape", None), getattr(v, "_fieldnames", None))
            first = v[0] if getattr(v, "shape", ()) else v
            print(" first:", getattr(first, "_fieldnames", type(first).__name__))
    except NotImplementedError:
        with h5py.File(path) as f:
            f.visititems(lambda n, o: print(n, getattr(o, "shape", "")) if n.count("/") < 3 else None)
EOF
