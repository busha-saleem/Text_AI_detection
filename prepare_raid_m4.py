"""
Data Preparation Script — RAID (AI samples) + M4 (Human samples)
=================================================================
Outputs:
  - raid_ai_15k.csv     → 15,000 AI-generated samples from RAID
  - m4_human_15k.csv    → 15,000 human-written samples from M4
  - merged_raid_m4.csv  → combined, shuffled, ready to merge with HC3 cleaned CSV

Both output CSVs have exactly two columns: 'answers', 'label'
  label = 1  → AI-generated
  label = 0  → Human-written

Usage:
  pip install kagglehub pandas
  python prepare_raid_m4.py
"""

import ast
import os
import random

import kagglehub
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES    = 15_000
MIN_WORDS    = 20          # drop samples shorter than this (neglect very short text)

random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(x):
    """Unwrap serialised list/dict strings; return plain string."""
    if isinstance(x, list):
        return ' '.join(map(str, x))
    if isinstance(x, dict):
        return ' '.join(map(str, x.values()))
    if isinstance(x, str):
        try:
            parsed = ast.literal_eval(x)
            if isinstance(parsed, list):
                return ' '.join(map(str, parsed))
            return str(parsed)
        except (ValueError, SyntaxError):
            return x
    return str(x)


def drop_short(df, col='answers', min_words=MIN_WORDS):
    """Remove rows where the text has fewer than min_words words."""
    before = len(df)
    df = df[df[col].str.split().str.len() >= min_words].reset_index(drop=True)
    print(f'  Dropped {before - len(df)} short samples (< {min_words} words). Remaining: {len(df)}')
    return df


def list_files(path, extensions=('.csv', '.json', '.jsonl')):
    """Recursively list files with given extensions under path."""
    found = []
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(extensions):
                found.append(os.path.join(root, f))
    return sorted(found)


# ─────────────────────────────────────────────────────────────────────────────
# RAID — AI-generated samples only
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('STEP 1 — Download RAID dataset')
print('='*60)
raid_path = kagglehub.dataset_download("ardava/raid-dataset")
print(f'Downloaded to: {raid_path}')

raid_files = list_files(raid_path)
print(f'Files found: {raid_files}')

# Load all RAID files and concatenate
raid_dfs = []
for f in raid_files:
    if f.endswith('.csv'):
        tmp = pd.read_csv(f, low_memory=False)
    elif f.endswith('.jsonl'):
        tmp = pd.read_json(f, lines=True)
    elif f.endswith('.json'):
        tmp = pd.read_json(f)
    else:
        continue
    raid_dfs.append(tmp)
    print(f'  Loaded {f}: {tmp.shape}  columns={tmp.columns.tolist()}')

raid_raw = pd.concat(raid_dfs, ignore_index=True)
print(f'\nRAID total rows: {len(raid_raw)}')
print(f'Columns: {raid_raw.columns.tolist()}')
print(f'Sample:\n{raid_raw.head(2)}')

# ── Identify text column ──────────────────────────────────────────────────────
# RAID typically has a column called 'generation' or 'text' for AI output
# and a 'label' or 'model' column indicating if it's AI-generated
TEXT_COL_RAID = None
for candidate in ['generation', 'text', 'answer', 'content', 'response']:
    if candidate in raid_raw.columns:
        TEXT_COL_RAID = candidate
        break

if TEXT_COL_RAID is None:
    # fallback: use the first string column
    TEXT_COL_RAID = raid_raw.select_dtypes(include='object').columns[0]

print(f'\nUsing text column: "{TEXT_COL_RAID}"')

# ── Keep only AI-generated rows ───────────────────────────────────────────────
# RAID labels: if there's a 'label' column with 0/1, keep label==1
# If there's a 'model' column, all rows in the generation col are AI-generated
if 'label' in raid_raw.columns:
    raid_ai = raid_raw[raid_raw['label'] == 1].copy()
    print(f'AI-labelled rows (label==1): {len(raid_ai)}')
elif 'model' in raid_raw.columns:
    # All rows with a non-null model name are AI-generated
    raid_ai = raid_raw[raid_raw['model'].notna()].copy()
    print(f'AI rows (model not null): {len(raid_ai)}')
else:
    # Assume entire dataset is AI-generated (RAID is an AI-detection benchmark)
    raid_ai = raid_raw.copy()
    print(f'Assuming all rows are AI-generated: {len(raid_ai)}')

# ── Clean & filter ────────────────────────────────────────────────────────────
raid_ai['answers'] = raid_ai[TEXT_COL_RAID].astype(str).apply(clean_text).str.lower()
raid_ai = raid_ai[['answers']].drop_duplicates(subset='answers')
raid_ai = drop_short(raid_ai)

if len(raid_ai) < N_SAMPLES:
    print(f'WARNING: Only {len(raid_ai)} AI samples available after filtering (need {N_SAMPLES}). Using all.')
    raid_sample = raid_ai.copy()
else:
    raid_sample = raid_ai.sample(n=N_SAMPLES, random_state=RANDOM_STATE)

raid_sample['label'] = 1
raid_sample = raid_sample[['answers', 'label']].reset_index(drop=True)

print(f'\nRAID AI sample: {raid_sample.shape}')
print(raid_sample.head(2))

raid_sample.to_csv('raid_ai_15k.csv', index=False)
print('Saved: raid_ai_15k.csv')


# ─────────────────────────────────────────────────────────────────────────────
# M4 — Human-written samples only
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('STEP 2 — Download M4 dataset')
print('='*60)
m4_path = kagglehub.dataset_download("serjhenrique/m4-multi-generator-domain")
print(f'Downloaded to: {m4_path}')

m4_files = list_files(m4_path)
print(f'Files found: {m4_files}')

m4_dfs = []
for f in m4_files:
    if f.endswith('.csv'):
        tmp = pd.read_csv(f, low_memory=False)
    elif f.endswith('.jsonl'):
        tmp = pd.read_json(f, lines=True)
    elif f.endswith('.json'):
        tmp = pd.read_json(f)
    else:
        continue
    m4_dfs.append(tmp)
    print(f'  Loaded {f}: {tmp.shape}  columns={tmp.columns.tolist()}')

m4_raw = pd.concat(m4_dfs, ignore_index=True)
print(f'\nM4 total rows: {len(m4_raw)}')
print(f'Columns: {m4_raw.columns.tolist()}')
print(f'Sample:\n{m4_raw.head(2)}')

# ── Identify text column ──────────────────────────────────────────────────────
# M4 typically has 'text', 'human_text', or 'original_text' for human-written
TEXT_COL_M4 = None
for candidate in ['human_text', 'original_text', 'text', 'answer', 'content']:
    if candidate in m4_raw.columns:
        TEXT_COL_M4 = candidate
        break

if TEXT_COL_M4 is None:
    TEXT_COL_M4 = m4_raw.select_dtypes(include='object').columns[0]

print(f'\nUsing text column: "{TEXT_COL_M4}"')

# ── Keep only human-written rows ──────────────────────────────────────────────
# M4 label convention: label==0 is human, label==1 is AI
if 'label' in m4_raw.columns:
    m4_human = m4_raw[m4_raw['label'] == 0].copy()
    print(f'Human-labelled rows (label==0): {len(m4_human)}')
elif 'source' in m4_raw.columns:
    m4_human = m4_raw[m4_raw['source'].str.lower().str.contains('human', na=False)].copy()
    print(f'Human rows (source contains human): {len(m4_human)}')
else:
    # If text col is 'human_text', all rows are human
    m4_human = m4_raw.copy()
    print(f'Assuming all rows in "{TEXT_COL_M4}" are human-written: {len(m4_human)}')

# ── Clean & filter ────────────────────────────────────────────────────────────
m4_human['answers'] = m4_human[TEXT_COL_M4].astype(str).apply(clean_text).str.lower()
m4_human = m4_human[['answers']].drop_duplicates(subset='answers')
m4_human = drop_short(m4_human)

if len(m4_human) < N_SAMPLES:
    print(f'WARNING: Only {len(m4_human)} human samples available after filtering (need {N_SAMPLES}). Using all.')
    m4_sample = m4_human.copy()
else:
    m4_sample = m4_human.sample(n=N_SAMPLES, random_state=RANDOM_STATE)

m4_sample['label'] = 0
m4_sample = m4_sample[['answers', 'label']].reset_index(drop=True)

print(f'\nM4 Human sample: {m4_sample.shape}')
print(m4_sample.head(2))

m4_sample.to_csv('m4_human_15k.csv', index=False)
print('Saved: m4_human_15k.csv')


# ─────────────────────────────────────────────────────────────────────────────
# Merge RAID + M4 into one clean CSV
# ─────────────────────────────────────────────────────────────────────────────

print('\n' + '='*60)
print('STEP 3 — Merge RAID + M4')
print('='*60)

merged = pd.concat([raid_sample, m4_sample], ignore_index=True)
merged = merged.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)  # shuffle

print(f'Merged shape: {merged.shape}')
print(f'Label distribution:\n{merged["label"].value_counts()}')
print(merged.head(5))

merged.to_csv('merged_raid_m4.csv', index=False)
print('\nSaved: merged_raid_m4.csv')

print('\n' + '='*60)
print('DONE. Next step: concatenate merged_raid_m4.csv with your HC3 cleaned_data.csv')
print('='*60)
print("""
  import pandas as pd

  hc3   = pd.read_csv('cleaned_data.csv')          # your existing HC3 CSV
  extra = pd.read_csv('merged_raid_m4.csv')         # RAID + M4

  final = pd.concat([hc3, extra], ignore_index=True)
  final = final.sample(frac=1, random_state=42).reset_index(drop=True)

  print(final['label'].value_counts())
  final.to_csv('final_training_data.csv', index=False)
""")
