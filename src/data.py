"""Download the OpenDengue dataset and extract monthly dengue cases for Bulacan.

Source: OpenDengue V1.3 spatial extract (CC BY 4.0), https://opendengue.org
Bulacan has monthly province-level counts for 1994-2010.
"""
import urllib.request
import zipfile

import pandas as pd

from . import config as C


def download_opendengue(force=False):
    """Fetch the OpenDengue zip (~55 MB) into data/raw/ unless it is already there."""
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    if C.OPENDENGUE_ZIP.exists() and not force:
        return C.OPENDENGUE_ZIP
    print(f"Downloading {C.OPENDENGUE_URL} ...")
    urllib.request.urlretrieve(C.OPENDENGUE_URL, C.OPENDENGUE_ZIP)
    return C.OPENDENGUE_ZIP


def build_bulacan_csv():
    """Filter the national extract down to Bulacan monthly rows and save a small CSV."""
    with zipfile.ZipFile(download_opendengue()) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        df = pd.read_csv(z.open(name), low_memory=False)
    b = df[(df.adm_2_name == C.PROVINCE) & (df.T_res == "Month")]
    b = b[["calendar_start_date", "calendar_end_date", "Year", "dengue_total"]].copy()
    b["month"] = pd.to_datetime(b.calendar_start_date).dt.month
    b = b.sort_values("calendar_start_date").reset_index(drop=True)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    b.to_csv(C.BULACAN_CSV, index=False)
    return b


def load_bulacan():
    """Bulacan monthly table (built from the raw download on first use)."""
    if not C.BULACAN_CSV.exists():
        return build_bulacan_csv()
    return pd.read_csv(C.BULACAN_CSV)


def load_season(year):
    """Monthly case counts for one outbreak season (May-December of `year`)."""
    df = load_bulacan()
    s = df[(df.Year == year) & (df.month >= C.SEASON_FIRST_MONTH)].sort_values("month")
    return s.month.to_numpy(), s.dengue_total.to_numpy(dtype=float)
