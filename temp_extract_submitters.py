import csv
from collections import defaultdict

data = defaultdict(list)

with open('Input/2025-10-07T13-26_export_SoilHealthClusters.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    sub_idx = headers.index('Submitter')
    title_idx = headers.index('Title')
    for row in reader:
        if len(row) > max(sub_idx, title_idx):
            sub = row[sub_idx]
            title = row[title_idx]
            if title not in data[sub]:
                data[sub].append(title)

for sub in sorted(data):
    print(f"{sub}: {', '.join(data[sub][:3])}")

