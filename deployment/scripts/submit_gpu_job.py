#!/usr/bin/env python3
"""Dispatch container workloads to the GPU runner Compute Engine VM.

Example:
  python deployment/scripts/submit_gpu_job.py \
      --image northamerica-northeast1-docker.pkg.dev/my-project/textflow-jobs/train:latest \
      --env RUN_ID=local-test --workdir /workspace -- python train.py --epochs 10
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import textwrap


def _key_value(text: str) -> str:
    if "=" not in text:
        raise argparse.ArgumentTypeError("expected KEY=VALUE")
    return text


def build_parser() -> argparse.ArgumentParser:
    description = "Submit a containerized job to the TextFlow GPU runner instance"
    epilog = textwrap.dedent(
        """
        Separate runner arguments from the container command with "--".
        Example: submit_gpu_job.py --image IMAGE --env KEY=value -- python script.py --epochs 10
        """
    ).strip()

    parser = argparse.ArgumentParser(
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("image", help="Container image reference to run on the GPU host")
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to execute inside the container (prefix with -- to separate)",
    )
    parser.add_argument(
        "--instance",
        default=os.environ.get("TF_GPU_RUNNER_INSTANCE", "textflow-gpu-runner"),
        help="Target Compute Engine instance name (default: textflow-gpu-runner)",
    )
    parser.add_argument(
        "--zone",
        default=os.environ.get("TF_GPU_RUNNER_ZONE"),
        help="Compute Engine zone for the target instance",
    )
    parser.add_argument(
        "--project",
        default=os.environ.get("TF_GPU_RUNNER_PROJECT"),
        help="GCP project ID hosting the instance",
    )
    parser.add_argument(
        "--gpus",
        default=os.environ.get("TF_GPU_COUNT", "all"),
        help="Value passed to docker --gpus (default: all)",
    )
    parser.add_argument(
        "--env",
        metavar="KEY=VALUE",
        action="append",
        type=_key_value,
        default=[],
        help="Container environment variables (repeatable)",
    )
    parser.add_argument(
        "--volume",
        metavar="HOST:CONTAINER[:MODE]",
        action="append",
        default=[],
        help="Bind mounts for docker run (repeatable)",
    )
    parser.add_argument(
        "--workdir",
        help="Working directory inside the container",
    )
    parser.add_argument(
        "--name",
        help="Optional container name override",
    )
    parser.add_argument(
        "--ssh-flag",
        action="append",
        default=[],
        help="Additional flag to forward to gcloud compute ssh (repeatable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the gcloud command without executing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    container_cmd = args.cmd
    if container_cmd and container_cmd[0] == "--":
        container_cmd = container_cmd[1:]

    runner_args: list[str] = ["--image", args.image, "--gpus", args.gpus]

    for env_entry in args.env:
        runner_args.extend(["--env", env_entry])

    for volume in args.volume:
        runner_args.extend(["--volume", volume])

    if args.workdir:
        runner_args.extend(["--workdir", args.workdir])

    if args.name:
        runner_args.extend(["--name", args.name])

    if container_cmd:
        runner_args.append("--")
        runner_args.extend(container_cmd)

    remote_body = "set -euo pipefail\n" + shlex.join(
        ["sudo", "/usr/local/bin/run_gpu_container_job.sh", *runner_args]
    )
    remote_command = f"bash -lc {shlex.quote(remote_body)}"

    gcloud_cmd = ["gcloud", "compute", "ssh", args.instance, "--command", remote_command]

    if args.zone:
        gcloud_cmd.extend(["--zone", args.zone])

    if args.project:
        gcloud_cmd.extend(["--project", args.project])

    for flag in args.ssh_flag:
        gcloud_cmd.extend(["--ssh-flag", flag])

    if args.dry_run:
        print(shlex.join(gcloud_cmd))
        return 0

    try:
        subprocess.run(gcloud_cmd, check=True)
    except subprocess.CalledProcessError as exc:
        return exc.returncode

    return 0


if __name__ == "__main__":
    sys.exit(main())
