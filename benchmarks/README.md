# Benchmarks

`benchmark_attendance.py` measures the durable local-storage path with synthetic
128-dimensional embeddings. It never loads face-recognition models, opens a
camera, or reads production data.

Run it from the repository root:

```text
python benchmarks/benchmark_attendance.py --events 1000 --users 8
```

The JSON output contains elapsed time for registry writes and attendance
writes, plus events per second. Results depend on CPU, filesystem, and
antivirus configuration; compare runs on the same machine and record the Python
version included in the output.

This benchmark measures persistence throughput, not recognition accuracy or
anti-spoof performance. Use the protocol in the main README for those metrics.
