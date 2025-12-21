import pandas as pd
from pathlib import Path
from functools import reduce

# =========================
# 配置：改路径
# =========================
CSV_DIR = Path(r"C:\Users\lyk\Desktop\data\data_300csv")  # 你的CSV文件夹
EACH_XLSX_DIR = Path(r"C:\Users\lyk\Desktop\data\data_300xls")  # 每个CSV转xlsx输出目录
OUT_MERGED_XLSX = Path(r"C:\Users\lyk\Desktop\data\merged_limday300.xls")  # 最终合并文件
STEP_COL = "step"

EACH_XLSX_DIR.mkdir(parents=True, exist_ok=True)

# 是否在列名重复时自动加后缀（推荐 True，避免 merge 后列名冲突）
DISAMBIGUATE_DUP_COLS = True

def read_csv_auto(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    seps = [",", ";", "\t", "|"]
    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(path, encoding=enc, sep=sep)
                if df.shape[1] >= 2:
                    return df
            except Exception as e:
                last_err = e
    raise RuntimeError(f"Failed to read CSV: {path}\nLast error: {last_err}")

csv_files = sorted([f for f in CSV_DIR.glob("*.csv") if not f.name.startswith("~$")])
print("Found", len(csv_files), "csv files:")
for f in csv_files:
    print(" -", f.name)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found in: {CSV_DIR}")

used_value_colnames = set()
tables = []
excel_paths = []

for f in csv_files:
    df = read_csv_auto(f)
    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    if STEP_COL not in df.columns:
        raise ValueError(f"{f.name} missing '{STEP_COL}'. columns={list(df.columns)}")

    value_cols = [c for c in df.columns if c != STEP_COL]
    if len(value_cols) != 1:
        raise ValueError(f"{f.name} expected 1 value col besides '{STEP_COL}', got {value_cols}")

    value_col = value_cols[0]  # 这里就是你的中文列名

    # 若多个文件的 value_col 名字相同，自动加后缀避免冲突
    final_value_col = value_col
    if DISAMBIGUATE_DUP_COLS and final_value_col in used_value_colnames:
        final_value_col = f"{value_col}__{f.stem}"
    used_value_colnames.add(final_value_col)

    out_df = df[[STEP_COL, value_col]].copy()
    out_df = out_df.rename(columns={value_col: final_value_col})

    out_df = out_df.sort_values(STEP_COL).drop_duplicates(subset=[STEP_COL], keep="last")

    out_xlsx = EACH_XLSX_DIR / f"{f.stem}.xlsx"
    if out_xlsx.exists():
        try:
            out_xlsx.unlink()
        except PermissionError:
            raise PermissionError(
                f"Cannot overwrite {out_xlsx}. Close it in Excel/WPS or change output folder."
            )

    out_df.to_excel(out_xlsx, index=False, engine="openpyxl")
    print("Saved single xlsx:", out_xlsx)

    excel_paths.append(out_xlsx)
    tables.append(out_df)

merged = reduce(lambda l, r: l.merge(r, on=STEP_COL, how="outer"), tables).sort_values(STEP_COL)
merged.to_excel(OUT_MERGED_XLSX, index=False, engine="openpyxl")
print("Merged saved:", OUT_MERGED_XLSX)
