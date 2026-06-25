#!/usr/bin/env python3
"""Index a JSONL manifest without loading the whole catalog into memory."""
import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path


def encode_form(fields: dict[str, str], image_path: Path) -> tuple[bytes, str]:
    boundary = f"----visual-search-{time.time_ns()}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        if value:
            chunks.extend([
                f"--{boundary}\r\n".encode(),
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                value.encode("utf-8"), b"\r\n",
            ])
    content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    chunks.extend([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        image_path.read_bytes(), b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ])
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def encode_batch(records: list[dict[str, object]]) -> tuple[bytes, str]:
    boundary = f"----visual-search-{time.time_ns()}"
    metadata = [{key: value for key, value in record.items() if key != "image_path"}
                for record in records]
    chunks: list[bytes] = [
        f"--{boundary}\r\n".encode(),
        b'Content-Disposition: form-data; name="metadata"\r\n\r\n',
        json.dumps(metadata, ensure_ascii=False).encode("utf-8"), b"\r\n",
    ]
    for record in records:
        image_path = Path(str(record["image_path"]))
        content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        chunks.extend([
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="images"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {content_type}\r\n\r\n".encode(),
            image_path.read_bytes(), b"\r\n",
        ])
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def send_batch(records: list[dict[str, object]], args: argparse.Namespace) -> str | None:
    body, content_type = encode_batch(records)
    error = None
    for attempt in range(args.retries):
        request = urllib.request.Request(
            f"{args.url.rstrip('/')}/v1/index/batch", data=body, method="POST",
            headers={"Content-Type": content_type, "X-API-Key": args.api_key},
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                response.read()
            return None
        except (urllib.error.URLError, TimeoutError) as exc:
            error = str(exc)
            time.sleep(2 ** attempt)
    return error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, help="每行一个 JSON 对象")
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batch-bytes", type=int, default=200 * 1024 * 1024)
    args = parser.parse_args()

    if args.batch_size < 1 or args.batch_size > 16:
        parser.error("--batch-size must be between 1 and 16")

    succeeded = failed = 0
    batch: list[dict[str, object]] = []
    batch_bytes = 0

    def flush() -> None:
        nonlocal succeeded, failed, batch, batch_bytes
        if not batch:
            return
        error = send_batch(batch, args)
        if error is None:
            succeeded += len(batch)
        else:
            failed += len(batch)
            print(json.dumps({"file_ids": [item["file_id"] for item in batch],
                              "error": error}, ensure_ascii=False))
        print(f"processed={succeeded + failed} succeeded={succeeded} failed={failed}")
        batch = []
        batch_bytes = 0

    with args.manifest.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image_path"])
            file_bytes = image_path.stat().st_size
            if batch and (len(batch) >= args.batch_size or
                          batch_bytes + file_bytes > args.max_batch_bytes):
                flush()
            batch.append({
                "product_id": str(record["product_id"]),
                "product_code": str(record["product_code"]),
                "file_id": str(record["file_id"]),
                "category": str(record.get("category", "")),
                "image_url": str(record.get("image_url", "")),
                "image_path": str(image_path),
            })
            batch_bytes += file_bytes
        flush()
    print(f"done succeeded={succeeded} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
