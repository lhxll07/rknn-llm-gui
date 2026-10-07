"""Build a pinned open iwagumi engine into a new, isolated probe bundle.

The upstream source must already be checked out at REVISION. No Rockchip
user-space library or Toolkit is linked. Requires CMake, a C compiler and
libdrm headers; --cross-compiler supports an existing AArch64 toolchain.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

REVISION = "8fafdbd28cce6eabb0ec2072bb549beb1035a141"
REPOSITORY = "https://github.com/fukumori/iwagumi"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cross-compiler", help="For example aarch64-linux-gnu-gcc")
    parser.add_argument("--drm-include", type=Path, help="Directory providing drm/drm.h")
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Expected pinned upstream revision {REVISION}, got {revision}")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Upstream source checkout must be clean for recorded provenance")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    build = output / "build"
    command = ["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DIWAGUMI_BUILD_TESTS=OFF"]
    if args.cross_compiler:
        command += ["-DCMAKE_SYSTEM_NAME=Linux", "-DCMAKE_SYSTEM_PROCESSOR=aarch64",
                    f"-DCMAKE_C_COMPILER={args.cross_compiler}"]
    if args.drm_include:
        command += [f"-DLIBDRM_INCLUDE_DIRS={args.drm_include.resolve()}"]
    with (output / "build.log").open("w") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run(["cmake", "--build", str(build), "--target", "iwagumi", "iwagumi-probe",
                        "run_matmul", "-j", str(args.jobs)], check=True, stdout=log, stderr=subprocess.STDOUT)
    lib = output / "lib"
    lib.mkdir()
    shutil.copy2(build / "src/api/libiwagumi.so.1.0.0", lib / "libiwagumi.so.1.0.0")
    (lib / "libiwagumi.so.1").symlink_to("libiwagumi.so.1.0.0")
    (lib / "libiwagumi.so").symlink_to("libiwagumi.so.1.0.0")
    shutil.copy2(build / "tools/probe/iwagumi-probe", output / "iwagumi-probe")
    shutil.copy2(build / "examples/c/run_matmul", output / "run_matmul")
    for name in ["LICENSE", "NOTICE"]:
        shutil.copy2(source / name, output / name)
    shutil.copytree(source / "include/iwagumi", output / "include/iwagumi")
    manifest = {"repository": REPOSITORY, "revision": revision, "license": "Apache-2.0",
                "cross_compiler": args.cross_compiler, "official_runtime_linked": False,
                "library_sha256": hashlib.sha256((lib / "libiwagumi.so.1.0.0").read_bytes()).hexdigest()}
    (output / "open-npu-build.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"bundle": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
