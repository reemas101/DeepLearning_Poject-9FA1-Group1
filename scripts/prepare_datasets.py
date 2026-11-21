import argparse
from pathlib import Path

from traffic_signs.data import prepare_gtsdb, prepare_gtsrb, verify_label_compatibility


def main():
    parser = argparse.ArgumentParser(description="Download GTSRB and convert extracted GTSDB")
    parser.add_argument("--gtsrb-root", default="data/raw/gtsrb")
    parser.add_argument("--gtsdb-root", default="data/raw/gtsdb/FullIJCNN2013")
    parser.add_argument("--processed", default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-gtsrb", action="store_true")
    parser.add_argument("--skip-gtsdb", action="store_true")
    args = parser.parse_args()
    if not args.skip_gtsrb:
        print(prepare_gtsrb(args.gtsrb_root, Path(args.processed), seed=args.seed))
    if not args.skip_gtsdb:
        print(prepare_gtsdb(args.gtsdb_root, Path(args.processed) / "gtsdb_yolo", seed=args.seed))
    if not args.skip_gtsrb and not args.skip_gtsdb:
        print(verify_label_compatibility(Path(args.processed) / "gtsrb_manifest.csv",
                                         Path(args.processed) / "gtsdb_yolo" / "gtsdb_annotations.csv",
                                         Path(args.processed)))


if __name__ == "__main__":
    main()

