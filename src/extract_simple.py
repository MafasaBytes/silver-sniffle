"""
Simple extraction script without tqdm - for terminals that have issues with progress bars
"""
import sys
sys.path.insert(0, str(__file__.rsplit('\\', 1)[0]))

from extract_features_yolov8_gpu import YOLOv8Extractor
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolov8m-pose.pt")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-hands", action="store_true")
    parser.add_argument("--checkpoint-interval", type=int, default=100)

    args = parser.parse_args()

    print("="*80)
    print("SIMPLE GPU EXTRACTION (NO PROGRESS BAR)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Split: {args.split}")
    print(f"Batch size: {args.batch_size}")
    print(f"Hands: {not args.no_hands}")
    print("="*80)
    print()

    # Initialize
    extractor = YOLOv8Extractor(
        yolo_model=args.model,
        use_hands=not args.no_hands,
        device='cuda',
        batch_size=args.batch_size
    )

    # Extract
    if args.split == "all":
        for split in ["train", "dev", "test"]:
            print(f"\n{'#'*80}")
            print(f"# EXTRACTING {split.upper()}")
            print(f"{'#'*80}\n")
            extractor.extract_dataset(
                split=split,
                checkpoint_interval=args.checkpoint_interval
            )
    else:
        extractor.extract_dataset(
            split=args.split,
            checkpoint_interval=args.checkpoint_interval
        )

    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
