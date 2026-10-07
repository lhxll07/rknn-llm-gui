#!/usr/bin/env bash
# Run inside the bundle produced by prepare_native_board_bundle.py.
set -euo pipefail

bundle_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$bundle_root"
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) echo 'This check requires an ARM64 Linux host with an RK3588 NPU.' >&2; exit 1 ;;
esac
board_python="${RKLLM_NATIVE_PYTHON:-python3}"
"$board_python" -c 'import sys; assert sys.version_info >= (3, 10), "Python 3.10+ required"; import numpy'
mkdir -p "$bundle_root/runs"
run_dir="$(mktemp -d "$bundle_root/runs/run-XXXXXXXX")"
echo "Results and logs: $run_dir"

"$board_python" - "$run_dir" <<'PY'
import hashlib, json, pathlib, platform, sys
import numpy
root = pathlib.Path.cwd()
manifest = json.loads((root / "bundle.json").read_text())
for name, expected in manifest["files_sha256"].items():
    with (root / name).open("rb") as file:
        digest = hashlib.sha256()
        while block := file.read(1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit("Bundle checksum mismatch: " + name)
record = {"machine": platform.machine(), "kernel": platform.release(),
          "python": sys.version, "numpy": numpy.__version__, "platform": platform.platform()}
compatible = pathlib.Path("/proc/device-tree/compatible")
if compatible.is_file():
    record["device_compatible"] = compatible.read_bytes().replace(b"\0", b", ").decode(errors="replace")
    if "rockchip,rk3588" not in record["device_compatible"]:
        raise SystemExit("This Runtime check is for RK3588; device-tree reports " + record["device_compatible"])
pathlib.Path(sys.argv[1], "environment.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
PY

"$board_python" -m native_converter convert --plan fixtures/model.rkplan \
  --model fixtures/model --output "$run_dir/native.rkllm" 2>&1 | tee "$run_dir/conversion.log"
cmp fixtures/reference.rkllm "$run_dir/native.rkllm"
echo 'ARM64 conversion: complete file matches the official reference.'

smoke_binary="$bundle_root/rkllm_native_smoke"
if [ ! -x "$smoke_binary" ]; then
  board_cxx="${CXX:-g++}"
  command -v "$board_cxx" >/dev/null || { echo 'g++ is required, or supply a prebuilt AArch64 verification program.' >&2; exit 1; }
  smoke_binary="$run_dir/rkllm_native_smoke"
  "$board_cxx" -std=c++17 -O2 -Wall -Wextra -I "$bundle_root/include" \
    "$bundle_root/src/rkllm_native_smoke.cpp" -L "$bundle_root/lib" \
    -lrkllmrt -o "$smoke_binary" 2>&1 | tee "$run_dir/build.log"
fi
max_context="$("$board_python" -c 'import json; print(json.load(open("bundle.json"))["max_context"])')"
vocab_size="$("$board_python" -c 'import json; print(json.load(open("bundle.json"))["vocab_size"])')"
LD_LIBRARY_PATH="$bundle_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$smoke_binary" fixtures/reference.rkllm "$run_dir/native.rkllm" "$max_context" "$vocab_size" \
  2>&1 | tee "$run_dir/runtime.log"

"$board_python" - "$run_dir" <<'PY'
import hashlib, json, pathlib, sys
directory = pathlib.Path(sys.argv[1])
prefix = "NATIVE_SMOKE_RESULT="
results = [line[len(prefix):] for line in (directory / "runtime.log").read_text(errors="replace").splitlines()
           if line.startswith(prefix)]
if len(results) != 1:
    raise SystemExit("Runtime verification result missing; inspect runtime.log")
report = json.loads(results[0])
if report.get("runtime_verified") is not True:
    raise SystemExit("Runtime did not confirm inference")
with (directory / "native.rkllm").open("rb") as file:
    digest = hashlib.sha256()
    while block := file.read(1024 * 1024):
        digest.update(block)
report.update(arm64_conversion_verified=True, official_byte_identical=True,
              sha256=digest.hexdigest(), environment=json.loads((directory / "environment.json").read_text()))
(directory / "board-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
PY
echo "Passed. Report: $run_dir/board-report.json"
