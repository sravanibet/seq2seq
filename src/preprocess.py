import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import re

pairs = []

# Resolve dataset path relative to script
script_dir = Path(__file__).resolve().parent

candidates = [
    script_dir.parent / "eng-fra.txt",
    script_dir.parent / "data" / "eng-fra.txt",
    script_dir.parent.parent / "eng-fra.txt",
    script_dir.parent.parent / "data" / "eng-fra.txt",
]

file_path = None
for p in candidates:
    if p.exists():
        file_path = p
        break

if file_path is None:
    tried = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Dataset file not found. Tried: {tried}")

def clean_text(text):
    """Clean and normalize text"""
    text = text.strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    return text

print("="*60)
print("ENGLISH-FRENCH DATASET PREPROCESSING")
print("="*60)

# Load TSV dataset
with file_path.open(encoding="utf-8") as f:
    for line in f:
        if "\t" in line:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                eng = clean_text(parts[0])
                fra = clean_text(parts[1])
                if eng and fra:  # Only keep non-empty pairs
                    pairs.append((eng, fra))

print(f"\nTotal Dataset Size: {len(pairs)}")

# Save cleaned data as TSV with headers
output_tsv = script_dir.parent / "data" / "eng-fra_cleaned.tsv"
output_tsv.parent.mkdir(parents=True, exist_ok=True)

with output_tsv.open("w", encoding="utf-8", newline="") as f:
    f.write("English\tFrench\n")  # Header row
    for eng, fra in pairs:
        f.write(f"{eng}\t{fra}\n")

print(f"✓ Cleaned TSV saved to: {output_tsv}")

# Train/Test Split (80/20)
train_data, test_data = train_test_split(
    pairs,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Size: {len(train_data)}")
print(f"Test Size: {len(test_data)}")

# Save train/test splits as TSV
output_train = script_dir.parent / "data" / "eng-fra_train.tsv"
output_test = script_dir.parent / "data" / "eng-fra_test.tsv"

with output_train.open("w", encoding="utf-8", newline="") as f:
    f.write("English\tFrench\n")
    for eng, fra in train_data:
        f.write(f"{eng}\t{fra}\n")

with output_test.open("w", encoding="utf-8", newline="") as f:
    f.write("English\tFrench\n")
    for eng, fra in test_data:
        f.write(f"{eng}\t{fra}\n")

print(f"✓ Training TSV saved to: {output_train}")
print(f"✓ Test TSV saved to: {output_test}")

# Representative Examples
print("\n" + "="*60)
print("Representative Test Examples:\n")

for i in range(5):
    eng, fra = random.choice(test_data)
    print("English:", eng)
    print("French :", fra)
    print("-" * 40)