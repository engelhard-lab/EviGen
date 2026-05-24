import argparse
from pathlib import Path
import lancedb
import pyarrow as pa
import pandas as pd
import duckdb
from sentence_transformers import SentenceTransformer
import torch
import time

PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_DIR / "data"
DEFAULT_DB_DIR = Path("/hpc/dctrl/fl105/EviGen")
DEFAULT_CSV_NAME = "dataset4death_code.csv"
DEFAULT_TABLE_NAME = "death_age_codes_qwen"
DEFAULT_BATCH_SIZE = 5000
DEFAULT_ENCODE_BATCH_SIZE = 16
VECTOR_DIM = 4096


def parse_args():
    parser = argparse.ArgumentParser(description="Build code LanceDB table from local CSV data.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    parser.add_argument("--csv-file", type=str, default=DEFAULT_CSV_NAME)
    parser.add_argument("--table-name", type=str, default=DEFAULT_TABLE_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--encode-batch-size", type=int, default=DEFAULT_ENCODE_BATCH_SIZE)
    return parser.parse_args()


def load_model():
    print("loading model ...")
    cache_dir = "/hpc/home/fl105/engelhardlab/fl105/cache"
    # Omit device_map so no dispatch wrapper is added — model is 16.5GB fp16, fits trivially
    # on a single GPU. Without device_map the post-load config patch works cleanly.
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-8B",
        cache_folder=cache_dir,
        model_kwargs={
            "attn_implementation": "flash_attention_2",
            "dtype": torch.float16,
        },
        tokenizer_kwargs={"padding_side": "left"},
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Disable KV cache: not needed for embedding-only inference. With use_cache=True (default),
    # all 36 layers accumulate KV tensors, causing OOM around layer 15.
    for mod in model.modules():
        if hasattr(mod, "config") and hasattr(mod.config, "use_cache"):
            mod.config.use_cache = False
            break
    print(f"use_cache = {model._first_module().auto_model.config.use_cache}")  # must print False

    print("model loaded")
    return model
    
def make_batches_from_csv(csv_file, model, batch_size, encode_batch_size):
    prev_time = time.time()
    for num_batch, chunk in enumerate(pd.read_csv(csv_file, chunksize=batch_size), start=1):
        if num_batch > 1:
            batch_time = time.time() - prev_time
            print(f"batch {num_batch-1} processed in {batch_time:.3f}s")
        prev_time = time.time()

        chunk['DaysToLastNotes'] = chunk['DaysToLastNotes'].astype(int)
        chunk['long_title'] = chunk['long_title'].fillna('').astype(str)

        # Collect long titles for encoding
        long_titles = chunk['long_title'].tolist()

        # Batch encode long titles at once
        embeddings = model.encode(long_titles, batch_size=encode_batch_size, show_progress_bar=False)

        # Create rows with batch-encoded embeddings
        rows = []
        for i, row in enumerate(chunk.itertuples(index=False)):
            rows.append({
                "subject_id": row.subject_id,
                "long_title": str(row.long_title),
                "icd_code_ver": str(row.icd_code_ver),
                "days_to_prediction": row.DaysToLastNotes,
                "vector": embeddings[i].tolist()
            })

        yield pd.DataFrame(rows)

def main():
    args = parse_args()
    csv_file = Path(args.csv_file)
    if not csv_file.is_absolute():
        csv_file = args.data_dir / csv_file
    db_path = args.db_dir
    table_name = args.table_name

    model = load_model()
    db = lancedb.connect(str(db_path))

    schema = pa.schema([
        pa.field("subject_id", pa.int64()),
        pa.field("long_title", pa.utf8()),
        pa.field("icd_code_ver", pa.utf8()),
        pa.field("days_to_prediction", pa.int64()),
        pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    ])

    print(f"csv_file={csv_file}")
    print(f"db_path={db_path}")
    print(f"table_name={table_name}")

    print("start encoding ...")
    db.create_table(
        table_name,
        make_batches_from_csv(
            csv_file=csv_file,
            model=model,
            batch_size=args.batch_size,
            encode_batch_size=args.encode_batch_size,
        ),
        schema=schema,
        mode="overwrite",
    )

    # display the first 10 rows
    tbl = db.open_table(table_name)
    lance_tbl = tbl.to_lance()
    df = duckdb.sql("SELECT subject_id, long_title, days_to_prediction, icd_code_ver FROM lance_tbl LIMIT 10").to_df()
    print(df)

    # create search index
    # choosing hyperparameters:
    # num_partitions: keeping each partition 1K-4K rows
    # num_sub_vectors: dimension / num_sub_vectors should be a multiple of 8
    # for optimum SIMD efficiency
    # default is the square root of number of rows
    # comment this line for exact search
    # tbl.create_index(num_partitions=1, num_sub_vectors=256, accelerator="cuda")


if __name__ == "__main__":
    main()
