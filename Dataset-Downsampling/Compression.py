# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 12:15:53 2024

@author: Tabish Ali Ansari

"""

import pyarrow as pa
import pyarrow.parquet as pq

# Load the Parquet file
file_path = r"D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\dataset.parquet"
table = pq.read_table(file_path)

# Define the compression parameters
compression = 'brotli'  # or 'zstd'
compression_level = 11  # Higher level can mean better compression, but slower

# Write the compressed Parquet file
compressed_file_path = r"D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\compressed_dataset.parquet"
pq.write_table(table, compressed_file_path, compression=compression, compression_level=compression_level)

# Check the size of the compressed file
import os
print(f"Compressed file size: {os.path.getsize(compressed_file_path) / (1024 * 1024)} MB")

import pandas as pd

# Load the Parquet file
file_path = r"D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\compressed_dataset.parquet"
data = pd.read_parquet(file_path)

# Save as CSV
csv_file_path = r"D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\compressed_dataset.csv"
data.to_csv(csv_file_path, index=False)

# Load the Parquet file
file_path = r"D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\compressed_dataset.parquet"
data = pd.read_parquet(file_path)

# Define the number of chunks
chunk_size = 100000  # adjust the size based on your needs
num_chunks = len(data) // chunk_size + 1

for i in range(num_chunks):
    start_row = i * chunk_size
    end_row = start_row + chunk_size
    chunk = data[start_row:end_row]
    chunk.to_parquet(f'D:\AISSMS IOIT - AI&DS (628299510)\General\Hackathons\Prasunethon\chunk_{i}.parquet', index=False)
