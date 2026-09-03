"""snapshot the outcome histogram to s3, the one table nothing can rebuild"""
from datetime import UTC, datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src import config, db
from src.process import archive

CHUNK_ROWS = 250_000
PREFIX = "outcome_histogram"


def export(path) -> int:
    """streamed in chunks so a 1.9 gb box never holds the whole table"""
    writer = None
    rows = 0
    try:
        for chunk in pd.read_sql_query(
            "SELECT route_id, stop_id, lead_bucket, month, error_bin, n FROM outcome_histogram",
            db.connect(),
            chunksize=CHUNK_ROWS,
        ):
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
            rows += len(chunk)
    finally:
        if writer is not None:
            writer.close()
    return rows


def main() -> None:
    if not archive.enabled():
        print("S3_ARCHIVE_BUCKET is not set, skipping the histogram snapshot")
        return

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    archive.STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = archive.STAGING_DIR / f"outcome_histogram-{stamp}.parquet"

    rows = export(path)
    size = path.stat().st_size
    key = f"{PREFIX}/outcome_histogram-{stamp}.parquet"
    archive.s3_client().upload_file(str(path), config.S3_ARCHIVE_BUCKET, key)
    path.unlink(missing_ok=True)

    print(f"snapshotted {rows:,} bins, {size / 1e6:.1f} MB "
          f"-> s3://{config.S3_ARCHIVE_BUCKET}/{key}")


if __name__ == "__main__":
    main()
