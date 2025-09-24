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
    rel_score: bool,
    S_churn: float,
    S_min: float,
    S_max: float,
    S_noise: float,
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
            *(["--rel_score"] if rel_score else ["--no-rel_score"]),
            f"--S_churn={S_churn}",
            f"--S_min={S_min}",
            f"--S_max={S_max}",
            f"--S_noise={S_noise}",
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
    parser.add_argument("--rel_score", action=argparse.BooleanOptionalAction)
    parser.add_argument("--S_churn", type=str, default="0")
    parser.add_argument("--S_min", type=float, default=0)
    parser.add_argument("--S_max", type=float, default=float("inf"))
    parser.add_argument("--S_noise", type=float, default=1)
    args = parser.parse_args()

    for seeds, steps, solver, S_churn in itertools.product(
        args.seeds.split(","),
        args.steps.split(","),
        args.solver.split(","),
        args.S_churn.split(","),
    ):
        net_name = args.network.split("/")[-1]
        outdir = args.outdir / (
            f"{net_name}_{solver}_{args.disc}_{args.schedule}_{args.scaling}"
            + f"_{args.rel_score}"
            + f"_{S_churn}_{args.S_min}_{args.S_max}_{args.S_noise}"
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
            rel_score=args.rel_score,
            S_churn=float(S_churn),
            S_min=args.S_min,
            S_max=args.S_max,
            S_noise=args.S_noise,
        )
