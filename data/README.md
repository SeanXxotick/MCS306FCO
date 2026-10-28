
# Data

## processed/bulacan_monthly_1994_2010.csv

Monthly reported dengue cases for the Province of Bulacan (Central Luzon, Region III),
January 1994 – December 2010 (200 months; a few months are missing in 1994, 1995 and 2002).

| Column | Meaning |
|---|---|
| calendar_start_date | first day of the month |
| calendar_end_date | last day of the month |
| Year | calendar year |
| dengue_total | reported dengue cases in that month |
| month | month number (1 = January) |

**Source:** OpenDengue V1.3 spatial extract (Clarke et al., 2024), https://opendengue.org,
licensed CC BY 4.0. Created by `src/data.py`, which downloads the release into `raw/`
(not committed, ~55 MB) and filters the Bulacan rows.

## Synthetic data

Generated on the fly by `src/sir.py` (`synthetic_data`) with β = 0.3, γ = 0.1 and seed 0;
no file is stored.
