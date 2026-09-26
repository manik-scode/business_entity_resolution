import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_blocking_v4 import evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--source1", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--db-records", type=int, required=True)
    parser.add_argument("--limit", type=int, default=1000)
    args = parser.parse_args()
    evaluate(args)
