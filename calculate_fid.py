import argparse
import subprocess
from pathlib import Path

# meant to be used with the edm2 repository (https://github.com/NVlabs/edm2)


def compute_metrics(
    gpus: int,
    *,
    images: Path,
    ref: str,
    metrics: str,
    outdir: Path,
) -> None:
    out = subprocess.check_output(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={gpus}",
            "calculate_metrics.py",
            "calc",
            f"--images={images}",
            f"--ref={ref}",
            f"--metrics={metrics}",
        ],
        text=True,
    )
    print(out)
    outdir.mkdir(exist_ok=True, parents=True)
    with open(outdir / f"{images.name}.txt", "w") as f:
        f.write(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--images", type=Path)
    parser.add_argument("--ref", type=str)
    parser.add_argument("--metrics", type=str)
    parser.add_argument("--outdir", type=Path, default=Path("fid"))
    args = parser.parse_args()

    dirnames = []
    for _, dirnames, _ in args.images.walk():
        break

    for dirname in sorted(dirnames):
        compute_metrics(
            gpus=args.gpus,
            images=args.images / dirname,
            ref=args.ref,
            metrics=args.metrics,
            outdir=args.outdir,
        )
