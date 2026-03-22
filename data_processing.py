import re
import pandas as pd
import numpy as np
from pathlib import Path


def parse_year(filename: str) -> int:
    """extract 4-digit year from filename like '07zpallagi.csv' -> 2007"""
    m = re.match(r'^(\d{2})', Path(filename).name)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {filename}")
    return 2000 + int(m.group(1))


def load_year(filepath: Path, year: int) -> pd.DataFrame:
    """load source CSV, keep columns needed for processing"""
    df = pd.read_csv(filepath, dtype=str)
    
    # standardize column names across IRS years
    df.columns = [c.upper() for c in df.columns]
    std_map = {
        'STATE': 'STATE',
        'ZIPCODE': 'zipcode',
        'AGI_STUB': 'agi_stub',
        'N1': 'N1',
        'A00100': 'A00100',
        'A19700': 'A19700',
        'N19700': 'N19700'
    }
    df = df.rename(columns=std_map)
    needed = ['STATE', 'zipcode', 'agi_stub', 'N1', 'A00100', 'A19700', 'N19700']
    df = df[needed]
    df['year'] = year
    return df


def aggregate_brackets(group: pd.DataFrame) -> pd.DataFrame:
    """
    return one row per (STATE, zipcode, year) with summed raw columns
    uses agi_stub=0 rows if present (pre-aggregated), else sums bracket rows 1-6
    note: if adding source columns in future, extend sum list here
    """
    raw_cols = ['N1', 'A00100', 'A19700', 'N19700']
    keys     = ['STATE', 'zipcode', 'year']

    if 0 in group['agi_stub'].values:
        return group[group['agi_stub'] == 0][keys + raw_cols]
    return group.groupby(keys, as_index=False)[raw_cols].sum()


def clean_data():
    src_dir = Path('data/original')

    csv_files = sorted(src_dir.glob('*.csv'))
    if not csv_files:
        print("No CSV files found in data/original/")
        return

    print(f"Source files: {[f.name for f in csv_files]}\n")

    # load all years
    frames = []
    for f in csv_files:
        year = parse_year(f.name)
        df   = load_year(f, year)
        print(f"  {f.name} -> year={year}, {len(df)} rows loaded")
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows across all years: {len(df)}")

    # shared cleaning

    # remove invalid / placeholder zips
    df = df[~df['zipcode'].isin(['00000', '0', '99999'])]
    df['zipcode'] = df['zipcode'].astype(str).str.zfill(5)
    df['STATE']   = df['STATE'].astype(str)

    # cast numerics
    numeric_cols = ['N1', 'A00100', 'A19700', 'N19700', 'agi_stub']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    df = df.dropna(subset=['N1', 'A00100', 'A19700', 'N19700'])

    # aggregate brackets per year
    # 2007/2009 use agi_stub=0 (pre-totalled), modern files use agi_stub 1-6 (must be summed)
    df = pd.concat(
        [aggregate_brackets(g) for _, g in df.groupby('year')],
        ignore_index=True
    )

    # filters
    df = df[
        (df['N1']     >= 100) &
        (df['A00100'] >  0)   &
        (df['A19700'] >  0)   &
        (df['N19700'] >  0)
    ]

    # derived metrics
    df['generosity_index']  = df['A19700'] / df['A00100']
    df['participation_rate'] = df['N19700'] / df['N1']

    # constrain to valid ratio range
    df = df[df['generosity_index'].between(0, 1,  inclusive='right')]
    df = df[df['participation_rate'].between(0, 1, inclusive='right')]

    # drop any remaining inf / NaN in metrics
    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['generosity_index', 'participation_rate']
    )

    # output
    print(f"\nRetained {len(df)} ZIP-year rows after cleaning.")
    print(f"Years present: {sorted(df['year'].unique())}")
    print(f"ZIP rows per year:\n{df.groupby('year').size().to_string()}")

    out_path = Path('data/zpallagi_cleaned.csv')
    df.to_csv(out_path, index=False)
    print(f"\nCleaned data exported to {out_path}")


if __name__ == '__main__':
    clean_data()
