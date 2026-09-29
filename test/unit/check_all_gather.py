# See LICENSE.txt for license information.
"""Run the AllGather regression; reject timeouts, fallback and missing checks."""

import argparse
from collections import Counter
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile


MODES = {
    "mc-poison": ("AllGather_LLBufferMC", "0", r"AllGather_LLBufferMC"),
    "uc-poison": ("AllGather_LLBuffer", "0", r"AllGather_LLBuffer(?:_R[48])?"),
    "uc-ll16": ("AllGather_LLBuffer", "1", r"AllGather_LLBuffer_LL16(?:_R[48])?"),
}
SIZES = [8, 1, 2, 7, 9, 14, 42, 40960, 40962, 327680, 2621442]


def verify(log, debug, options):
    sizes = SIZES if options.bytes is None else [options.bytes]
    checks = len(sizes) * (options.eager + options.graphs)
    expected = [(str(rank), str(checks)) for rank in range(options.ranks)]
    passes = re.findall(r"^PASS rank=(\d+) checks=(\d+) ", log, re.MULTILINE)
    complete = re.findall(r"^ALL_PASS ranks=(\d+) total_checks=(\d+)$", log, re.MULTILINE)
    if (sorted(passes, key=lambda row: int(row[0])) != expected
            or complete != [(str(options.ranks), str(options.ranks * checks))]
            or "FAIL" in log):
        raise ValueError("missing or inconsistent rank checks")
    actual = re.findall(r"AllGather \[Symmetric\]: (\d+) Bytes -> Kernel (\S+)", debug)
    pattern = MODES[options.mode][2]
    # Only rank zero logs tuning decisions. Graph capture logs once per case;
    # replay execution is checked by the binary's GPU epoch and byte verifier.
    expected_dispatches = Counter({size: options.eager + (options.graphs > 0) for size in sizes})
    if (Counter(int(size) for size, _ in actual) != expected_dispatches
            or any(re.fullmatch(pattern, kernel) is None for _, kernel in actual)
            or re.search(r"AllGather: \d+ Bytes -> Algo", debug)):
        raise ValueError("missing target-kernel dispatch or unexpected fallback")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/all_gather_llbuffer_test"))
    parser.add_argument("--mode", choices=MODES, default="mc-poison")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--eager", type=int, default=3)
    parser.add_argument("--graphs", type=int, default=16)
    parser.add_argument("--bytes", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--log-dir", type=Path,
                        help="new directory; logs are retained on success/failure")
    options = parser.parse_args()
    if options.timeout <= 0:
        parser.error("--timeout must be positive")
    if options.offset and options.bytes is None:
        parser.error("--offset requires --bytes")
    if options.log_dir:
        options.log_dir.mkdir(parents=True, exist_ok=False)
        directory = options.log_dir.resolve()
    else:
        directory = Path(tempfile.mkdtemp(prefix="nccl-allgather-"))
    print(f"Logs: {directory}", flush=True)
    kernel, sync, _ = MODES[options.mode]
    env = dict(os.environ, NCCL_CUMEM_ENABLE="1", NCCL_WIN_ENABLE="1", NCCL_NVLS_ENABLE="1",
               NCCL_GIN_ENABLE="0", NCCL_GRAPH_MIXING_SUPPORT="1", NCCL_IB_DISABLE="1",
               NCCL_NET="Socket", NCCL_SOCKET_IFNAME="lo", NCCL_SYM_KERNEL=kernel,
               NCCL_SYM_LLBUFFER_SYNC=sync, NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="TUNING",
               NCCL_DEBUG_FILE=str(directory / "nccl-%p.log"))
    command = [str(options.binary.resolve()), "--ranks", str(options.ranks),
               "--eager", str(options.eager), "--graphs", str(options.graphs)]
    if options.bytes is not None:
        command += ["--bytes", str(options.bytes), "--offset", str(options.offset)]
    if options.in_place:
        command.append("--in-place")
    try:
        with (directory / "test.log").open("x") as stream:
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            try:
                code = process.wait(timeout=options.timeout)
            except BaseException:
                # The unreaped group leader keeps its PID reserved. Kill this
                # run's forked GPU ranks too, not just the parent, on interruption.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
                raise
        if code != 0:
            raise ValueError(f"test exited {code}")
        log = (directory / "test.log").read_text()
        debug = "\n".join(path.read_text() for path in directory.glob("nccl-*.log"))
        verify(log, debug, options)
    except (OSError, ValueError, subprocess.TimeoutExpired) as error:
        print(f"FAIL: {error}; see {directory}", file=sys.stderr)
        return 1
    print(f"CHECK_PASS mode={options.mode} ranks={options.ranks}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
