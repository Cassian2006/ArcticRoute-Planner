import json
from pathlib import Path

json_file = Path("tests/data/ais_sample.json")
with json_file.open("r") as f:
    data = json.load(f)

print("JSON structure:")
print(json.dumps(data, indent=2))

# 提取所有记录
records = []
if isinstance(data, list):
    for item in data:
        if isinstance(item, dict) and "data" in item:
            records.extend(item["data"])
        else:
            records.append(item)

print("\n\nExtracted records:")
for i, record in enumerate(records):
    print(f"\nRecord {i}:")
    print(f"  Keys: {list(record.keys())}")
    print(f"  Data: {record}")

