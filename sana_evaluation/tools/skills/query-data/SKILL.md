---
name: query-data
description: Column-name gotchas and extraction recipes. Load before your first query_file, grep_file, read_file, or execute_code call.
---

## Column names rarely match the question

Government files use codes. Check columns before writing SQL — `peek_file` for one file, `peek_multiple` for 2+, or `SELECT * FROM t LIMIT 1`.

| You might guess | It might actually be |
|---|---|
| `state` | `STFIP`, `ST_FIPS`, `state_code`, `STATE` |
| `county` | `NMCNTY`, `COUNTY`, `CTY_NAME`, `CTYNAME` |
| `year` | `YEAR`, `fiscal_year`, `SURVEY_YEAR` |
| `school_name` | `SCHNAM`, `SCH_NAME`, `INSTNM` |

---

## Error recipes

**"Referenced column X not found"** → guessed a column. Check with `peek_file` or `SELECT * FROM t LIMIT 1`, then rewrite.

**"maximum_object_size exceeded" (~100MB)** → file too large for DuckDB. Download, then `pd.read_csv(path, usecols=[...])` with only the columns you need.

**Empty DataFrame after filter** → type mismatch. FIPS codes often load as integers (`6`) but are stored as strings (`"06"`):
```python
df['STFIP'] = df['STFIP'].astype(str).str.zfill(2)
```

**Encoding errors** → try `encoding='utf-8-sig'`, then `encoding='latin-1'`.

---

## execute_code checklist

1. Reload the file at the top — state does NOT persist between calls
2. `print(df.columns.tolist())` right after loading
3. `print(df.head(2))` before any filtering
4. Cast FIPS codes to `str` before comparing
5. `print()` the final result — silent code produces nothing
