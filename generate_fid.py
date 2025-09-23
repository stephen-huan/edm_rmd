import argparse
import itertools
import subprocess
from pathlib import Path


def generate(
    gpus: int,
    network: str,
    outdir: Path,
    *,
    seeds: str,
    subdirs: bool = True,
    steps: int,
    solver: str,
    disc: str,
    schedule: str,
    scaling: str,
) -> None:
    subprocess.run(
        [
            "torchrun",
            "--standalone",
            f"--nproc_per_node={gpus}",
            "generate.py",
            f"--outdir={outdir}",
            *(["--subdirs"] if subdirs else []),
            f"--seeds={seeds}",
            f"--steps={steps}",
            f"--solver={solver}",
            f"--disc={disc}",
            f"--schedule={schedule}",
            f"--scaling={scaling}",
            f"--network={network}",
        ]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--network", type=str)
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--seeds", type=str)
    parser.add_argument("--steps", type=str)
    parser.add_argument("--solver", type=str)
    parser.add_argument("--disc", type=str)
    parser.add_argument("--schedule", type=str)
    parser.add_argument("--scaling", type=str)
    args = parser.parse_args()

    for seeds, steps, solver in itertools.product(
        args.seeds.split(","), args.steps.split(","), args.solver.split(",")
    ):
        net_name = args.network.split("/")[-1]
        outdir = args.outdir / (
            f"{net_name}_{solver}_{args.disc}_{args.schedule}_{args.scaling}"
            + f"_{steps:0>4}_{seeds}"
        )
        outdir.mkdir(exist_ok=True, parents=True)
        generate(
            gpus=args.gpus,
            network=args.network,
            outdir=outdir,
            seeds=seeds,
            steps=int(steps),
            solver=solver,
            disc=args.disc,
            schedule=args.schedule,
            scaling=args.scaling,
        )
