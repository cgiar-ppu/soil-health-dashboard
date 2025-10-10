import csv

uniques = set()

with open('Input/2025-10-07T13-26_export_SoilHealthClusters.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    col_idx = headers.index('Submitter')
    for row in reader:
        if len(row) > col_idx:
            uniques.add(row[col_idx])

print('\n'.join(sorted(uniques)))

