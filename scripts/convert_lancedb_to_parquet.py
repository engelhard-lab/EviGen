#!/usr/bin/env python3
"""
Convert a LanceDB table to Parquet file.

Usage:
    python scripts/convert_lancedb_to_parquet.py --db-uri ../ --table death_age_chunksize200_qwen --output notes.parquet
"""

import argparse
import lancedb
import pandas as pd
import os


def main():
    parser = argparse.ArgumentParser(
        description="Convert a LanceDB table to Parquet format"
    )

    parser.add_argument(
        "--db-uri",
        type=str,
        required=True,
        help="Path to LanceDB database directory"
    )
    parser.add_argument(
        "--table",
        type=str,
        required=True,
        help="Name of the table to convert"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("LanceDB to Parquet Conversion")
    print("=" * 80)
    print(f"Database:  {args.db_uri}")
    print(f"Table:     {args.table}")
    print(f"Output:    {args.output}")
    print("=" * 80)

    # Connect to LanceDB
    print("\nConnecting to LanceDB...")
    db = lancedb.connect(args.db_uri)

    # Open table
    print(f"Opening table '{args.table}'...")
    tbl = db.open_table(args.table)
    print(f"Schema: {tbl.schema.names}")

    # Convert to pandas
    print("Loading to pandas (this may take a while)...")
    df = tbl.to_pandas()
    print(f"Rows: {len(df):,}")

    # Calculate size
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"Memory usage: {memory_mb:.2f} MB")

    # Save to parquet
    print("Writing to parquet...")
    df.to_parquet(args.output, compression='snappy', index=False)

    # Check file size
    file_size_mb = os.path.getsize(args.output) / 1024**2
    print(f"\n✓ Conversion complete!")
    print(f"  File: {args.output}")
    print(f"  Size: {file_size_mb:.2f} MB")
    print(f"  Compression ratio: {memory_mb / file_size_mb:.1f}x")


if __name__ == "__main__":
    main()
