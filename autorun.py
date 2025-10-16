import subprocess
import re
import requests
import os
import time
import signal
from colorama import Fore, Style, init
from multiprocessing import Process
import argparse

# --- Initialize colorama ---
init(autoreset=True)

# --- Color definitions ---
COLOR_INFO = Fore.GREEN
COLOR_WARN = Fore.YELLOW
COLOR_ERROR = Fore.RED
COLOR_DEBUG = Fore.CYAN
COLOR_HEADER = Fore.MAGENTA + Style.BRIGHT
COLOR_RESET = Style.RESET_ALL

# --- Paths ---
BASE_DIR = "/home/dongtv/userspace_test"
LOG_SYSTEM_PATH = os.path.join(BASE_DIR, "autorun.log")
XDP_STATS_LOG = os.path.join(BASE_DIR, "xdp_stats")
USERSPACE_LOG = os.path.join(BASE_DIR, "userspace_log")
BPF_DIR = os.path.join(BASE_DIR, "log_bpf")
LANFORGE_DIR = os.path.join("/home/lanforge/Desktop/app", "results_userspace")

# --- Parse CLI arguments ---
parser = argparse.ArgumentParser(description="Automated XDP profiling runner")
parser.add_argument(
    "--model",
    default="all",
    help="Tên model (knn, lof, isoforest, randforest, svm hoặc 'all' để chạy toàn bộ)"
)
parser.add_argument("--max-time", type=int, default=120, help="Thời gian chạy mỗi lần (mặc định: 120s)")
parser.add_argument("--num-runs", type=int, default=5, help="Số lần lặp lại mỗi mức PPS (mặc định: 5)")
args = parser.parse_args()

# model = args.model
MAX_TIME = args.max_time
NUM_RUNS = args.num_runs

ALL_MODELS = ["knn", "isoforest", "randforest"]

# Nếu người dùng chỉ định model cụ thể
if args.model.lower() != "all":
    models_to_run = [args.model.lower()]
else:
    models_to_run = ALL_MODELS

os.makedirs(XDP_STATS_LOG, exist_ok=True)
os.makedirs(USERSPACE_LOG, exist_ok=True)
os.makedirs(BPF_DIR, exist_ok=True)

# --- Input params ---
# branch = "featureA"
# param = "knn200"
api_url = "http://192.168.101.238:20168/run"
iface = "eno3"
# MAX_TIME = 20
# NUM_RUNS = 5

# --- System-wide log file ---
g_system_log = open(LOG_SYSTEM_PATH, "a", buffering=1)
g_log_file = None  # profiling log (per-run)

# --- Logging function ---
def log(level, message, to_file=True):
    global g_system_log
    if level == 'INFO': color, prefix = COLOR_INFO, '[INFO]'
    elif level == 'WARN': color, prefix = COLOR_WARN, '[WARN]'
    elif level == 'ERROR': color, prefix = COLOR_ERROR, '[ERROR]'
    elif level == 'DEBUG': color, prefix = COLOR_DEBUG, '[DEBUG]'
    elif level == 'HEADER': color, prefix = COLOR_HEADER, '==='
    else: color, prefix = COLOR_RESET, ''
    msg = f"{prefix} {message}"
    print(f"{color}{msg}{COLOR_RESET}")
    timestamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    # Always log to system log
    if g_system_log:
        g_system_log.write(f"{timestamp} {msg}\n")
        g_system_log.flush()
    # Also log to per-profiling log if active
    if to_file and g_log_file:
        g_log_file.write(f"{timestamp} {msg}\n")
        g_log_file.flush()

# --- Run shell command ---
def run_cmd(cmd, desc, check=True):
    log('DEBUG', f"{desc}: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout.strip():
            for line in result.stdout.splitlines():
                log('INFO', f"  {line}", to_file=False)
        if result.stderr.strip():
            for line in result.stderr.splitlines():
                log('WARN', f"  {line}", to_file=False)
        return result
    except subprocess.CalledProcessError as e:
        log('ERROR', f"Command failed ({desc}): {e}")
        log('ERROR', f"STDOUT: {e.stdout.strip()}")
        log('ERROR', f"STDERR: {e.stderr.strip()}")
        if check: raise
        return None

# --- Get loaded XDP program ID ---
def get_prog_id():
    log('DEBUG', "Getting XDP program ID for 'xdp_anomaly_detector'...")
    try:
        bpftool_out = subprocess.check_output(["sudo", "bpftool", "prog", "show"], text=True)
    except subprocess.CalledProcessError as e:
        log('ERROR', f"Failed to run bpftool: {e}")
        raise RuntimeError("bpftool failed.")
    match = re.search(r'^(\d+):\s+(xdp|ext)\s+name\s+xdp_anomaly_detector', bpftool_out, re.MULTILINE)
    if not match:
        log('ERROR', "No XDP program named 'xdp_anomaly_detector' found!")
        raise RuntimeError("No XDP program found!")
    prog_id = match.group(1)
    log('INFO', f"Found xdp_anomaly_detector ID: {prog_id}")
    return prog_id

# --- Unload all XDP programs ---
def unload_xdp():
    run_cmd(["sudo", "xdp-loader", "unload", iface, "--all"], "Unload all XDP programs", check=False)
    time.sleep(2)

# --- Call tcpreplay API ---
def call_tcpreplay_api(api_url, log_file, speed, duration):
    payload = {"log": log_file, "speed": speed, "duration": duration}
    log('DEBUG', f"Calling tcpreplay API at {api_url} (duration={duration}s)...")
    try:
        resp = requests.post(api_url, json=payload, timeout=duration + 10)
        if resp.status_code == 200:
            log('INFO', f"API OK -> {resp.json()}")
        else:
            log('WARN', f"API returned {resp.status_code}: {resp.text}")
    except Exception as e:
        log('ERROR', f"Failed to call API: {e}")

def run_xdp_stats(log_file_path, iface="eno3", duration=60):
    log('DEBUG', f"Starting xdp_stats on {iface} (duration={duration}s)...", to_file=False)
    cmd = ["sudo", "/home/dongtv/dtuan/xdp-program/xdp_prog/xdp_stats", "--dev", iface]
    with open(log_file_path, "a", buffering=1) as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        log('INFO', f"[XDP_STATS] Started (PID={proc.pid})", to_file=False)
        try:
            time.sleep(duration)
        finally:
            if proc.poll() is None:
                log('DEBUG', f"[XDP_STATS] Stopping after {duration}s...", to_file=False)
                os.killpg(proc.pid, signal.SIGINT)
                time.sleep(1)
                if proc.poll() is None:
                    log('WARN', "[XDP_STATS] Forcing kill...", to_file=False)
                    os.killpg(proc.pid, signal.SIGKILL)
            log('INFO', "[XDP_STATS] Finished.", to_file=False)
    
def run_userspace_test(model, base_dir, log_file, max_time=120, cpu_core=1):
    venv_python = os.path.join(base_dir, ".venv/bin/python3")
    if not os.path.exists(venv_python):
        raise FileNotFoundError(f"Không tìm thấy python venv: {venv_python}")

    cmd = ["sudo", "taskset", "-c", str(cpu_core), venv_python, "main.py", "--model", model]
    print(f"[USERSPACE] Running: {' '.join(cmd)}")
    print(f"[USERSPACE] Logging to: {log_file}")

    # Mở log file
    with open(log_file, "a", buffering=1) as f:
        # preexec_fn=os.setsid để tạo process group riêng => dễ SIGINT/SIGKILL toàn nhóm
        proc = subprocess.Popen(cmd, cwd=base_dir, stdout=f, stderr=subprocess.STDOUT, preexec_fn=os.setsid)
        start_time = time.time()

        try:
            # Giám sát thời gian chạy
            while (time.time() - start_time) < max_time:
                if proc.poll() is not None:
                    print("[USERSPACE] main.py kết thúc sớm.")
                    return
                time.sleep(1)

            print(f"[USERSPACE] Hết {max_time}s, gửi SIGINT để dừng.")
            os.killpg(proc.pid, signal.SIGINT)
            time.sleep(3)
            if proc.poll() is None:
                print("[USERSPACE] Vẫn chạy, gửi SIGKILL.")
                os.killpg(proc.pid, signal.SIGKILL)
        except KeyboardInterrupt:
            print("[USERSPACE] Nhận Ctrl+C, dừng tiến trình.")
            os.killpg(proc.pid, signal.SIGINT)
        finally:
            proc.wait(timeout=5)
            print("[USERSPACE] Đã dừng.")
    
# --- Initial Cleanup ---
unload_xdp()
run_cmd(["sudo", "rm", "-rf", f"/sys/fs/bpf/{iface}"], "Remove old BPF maps", check=False)
run_cmd(["sudo", "pkill", "-9", "bpftool"], "Kill stray bpftool", check=False)
run_cmd(["sudo", "pkill", "-9", "perf"], "Kill stray perf", check=False)

# --- Main loop ---

for model in models_to_run:
    log('HEADER', f"### Bắt đầu test model: {model} ###")
    for pps in range(10000, 100001, 10000):
        for run_idx in range(1, NUM_RUNS + 1):
            log_file_lanforge = os.path.join(LANFORGE_DIR, f"log_{model}_{pps}_{run_idx}.txt")
            log_file_bpf = os.path.join(BPF_DIR, f"log_{model}_{pps}_{run_idx}.txt")
            log_run_xdp_stats = os.path.join(XDP_STATS_LOG, f"log_{model}_{pps}_{run_idx}.txt")
            log_userspace = os.path.join(USERSPACE_LOG, f"log_{model}_{pps}_{run_idx}.txt")
            
            g_log_file = open(log_file_bpf, "a")
            log('HEADER', f"=== PPS={pps}, Run {run_idx}/{NUM_RUNS} ===")
            g_log_file.write(f"=== PPS={pps}, RUN={run_idx}, BRANCH={model}, TIME={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

            run_cmd([
                "sudo", "xdp-loader", "load", iface,
                "-m", "skb",
                "-n", "xdp_anomaly_detector",
                "-p", f"/sys/fs/bpf/{iface}",
                "/home/dongtv/dtuan/xdp-program/xdp_prog/xdp_prog_kern.o"
            ], "Load XDP program")

            p_xdp_stats = Process(target=run_xdp_stats, args=(log_run_xdp_stats, iface, MAX_TIME))
            p_xdp_stats.start()
            try:
                prog_id = get_prog_id()
            except RuntimeError:
                unload_xdp()
                g_log_file.close()
                continue

            call_tcpreplay_api(api_url, log_file_lanforge, pps, MAX_TIME+5)
            p_userspace = Process(target=run_userspace_test, args=(model, BASE_DIR, log_userspace, MAX_TIME, 1))
            p_userspace.start()
            p_userspace.join()
            p_xdp_stats.join()
            unload_xdp()
            run_cmd(["sudo", "rm", "-rf", f"/sys/fs/bpf/{iface}"], "Remove old BPF maps", check=False)
            log('INFO', f"Completed PPS={pps}, Run={run_idx}")
            g_log_file.write(f"=== DONE PPS={pps}, RUN={run_idx}, TIME={time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")
            # g_log_file.close()
            time.sleep(3)

log('HEADER', "=== HOÀN THÀNH TOÀN BỘ TEST ===")
if g_system_log:
    g_system_log.close()