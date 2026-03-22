import re
import pandas as pd
import numpy as np
from pathlib import Path

# column layout confirmed by header inspection
#
# 2007 (per-state XLS, e.g. "ZIP Code 2007 AL.xls")
#   col 0 = agi_stub text (NaN on total row)
#   col 1 = zip code
#   col 2 = number of returns (N1)
#   col 7 = adjusted gross income (A00100) [AL = $100.7B]
#   col 41 = contributions - number of returns (N19700)
#   col 42 = contributions - amount (A19700)
#   data rows begin at row 8
#
# 2009 (per-state XLS, e.g. "09zp01al.xls")
#   col 0 = zip code
#   col 1 = agi_stub text (NaN on total row)
#   col 2 = number of returns (N1)
#   col 7 = adjusted gross income (A00100) [AL = $92.7B]
#   col 43 = contributions - number of returns (N19700)
#   col 44 = contributions - amount (A19700)
#   data rows begin at row 6
#
# 2010 (per-state XLS, e.g. "10zp01al.xls")
#   identical layout to 2009, confirmed by header inspection
#   col 0 = zip code
#   col 1 = agi_stub text (NaN on total row)
#   col 2 = number of returns (N1)
#   col 7 = adjusted gross income (A00100)
#   col 43 = contributions - number of returns (N19700)
#   col 44 = contributions - amount (A19700)
#   data rows begin at row 6

XLS_CONFIGS = {
    2007: {
        'folder':    '2007zipcode',
        'output':    '07zpallagi.csv',
        'col_zip':      1,
        'col_stub':     0,
        'col_n1':       2,
        'col_a00100':   7,
        'col_n19700':  41,
        'col_a19700':  42,
        'data_start':   8,
        'get_state': lambda name: Path(name).stem.split()[-1].upper(),
        # "ZIP Code 2007 AL" -> "AL"
    },
    2009: {
        'folder':    '2009zipcode',
        'output':    '09zpallagi.csv',
        'col_zip':      0,
        'col_stub':     1,
        'col_n1':       2,
        'col_a00100':   7,
        'col_n19700':  43,
        'col_a19700':  44,
        'data_start':   6,
        'get_state': lambda name: re.search(r'([a-z]{2})\.xls$', name, re.I).group(1).upper(),
        # "09zp01al.xls" -> "AL"
    },
    2010: {
        'folder':    '2010zipcode',
        'output':    '10zpallagi.csv',
        'col_zip':      0,
        'col_stub':     1,
        'col_n1':       2,
        'col_a00100':   7,
        'col_n19700':  43,
        'col_a19700':  44,
        'data_start':   6,
        'get_state': lambda name: re.search(r'([a-z]{2})\.xls$', name, re.I).group(1).upper(),
        # "10zp01al.xls" -> "AL"
    },
}

# skip national total files
SKIP_STATES = {'US'}


def is_valid_zip(val) -> bool:
    """true if val is numeric zip in plausible range (100-99999)"""
    try:
        n = int(float(str(val).strip()))
        return 100 <= n <= 99999
    except (ValueError, TypeError):
        return False


def process_file(filepath: Path, cfg: dict) -> pd.DataFrame | None:
    state = cfg['get_state'](filepath.name)
    if state in SKIP_STATES:
        return None

    df = pd.read_excel(filepath, header=None)
    data = df.iloc[cfg['data_start']:].copy().reset_index(drop=True)

    zc  = cfg['col_zip']
    sc  = cfg['col_stub']
    n1  = cfg['col_n1']
    agi = cfg['col_a00100']
    nc  = cfg['col_n19700']
    ac  = cfg['col_a19700']

    valid_zip = data[zc].apply(is_valid_zip)
    is_total  = data[sc].isna()

    # primary path: use pre-aggregated total row per ZIP (agi_stub == NaN)
    rows = data[valid_zip & is_total].copy()

    if rows.empty:
        # fallback: sum bracket rows per ZIP
        bracket_rows = data[valid_zip & ~is_total].copy()
        if bracket_rows.empty:
            print(f"    WARNING: no usable rows in {filepath.name}")
            return None
        for col in [n1, agi, nc, ac]:
            bracket_rows[col] = pd.to_numeric(bracket_rows[col], errors='coerce')
        rows = (
            bracket_rows.groupby(zc)[[n1, agi, nc, ac]]
            .sum()
            .reset_index()
        )

    result = pd.DataFrame({
        'STATE':   state,
        'zipcode': rows[zc].astype(str).str.strip().str.zfill(5),
        'agi_stub': 0,
        'N1':     pd.to_numeric(rows[n1],  errors='coerce'),
        'A00100': pd.to_numeric(rows[agi], errors='coerce'),
        'N19700': pd.to_numeric(rows[nc],  errors='coerce'),
        'A19700': pd.to_numeric(rows[ac],  errors='coerce'),
    })
    return result


def convert_year(year: int, base_dir: Path, output_dir: Path):
    cfg    = XLS_CONFIGS[year]
    folder = base_dir / cfg['folder']
    output = output_dir / cfg['output']

    xls_files = sorted(folder.glob('*.xls'))
    print(f"[{year}] {len(xls_files)} XLS files found in {folder.name}/")

    frames = []
    for f in xls_files:
        df = process_file(f, cfg)
        if df is not None and not df.empty:
            frames.append(df)

    if not frames:
        print(f"[{year}] ERROR: no data collected - verify column indices")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(output, index=False)

    print(f"[{year}] {len(combined)} ZIP rows written -> {output.name}")
    print(f"       States: {sorted(combined['STATE'].unique())}")
    print(f"       Sample:")
    print(combined.head(5).to_string(index=False))
    print()


def main():
    base_dir   = Path('data/original/convertXLSData')
    output_dir = Path('data/original')

    for year in [2007, 2009, 2010]:
        convert_year(year, base_dir, output_dir)


if __name__ == '__main__':
    main()
