#!/usr/bin/env python3
import ctypes
import socket
import struct
from bcc import libbcc

# ---- DEFINITIONS: must MATCH the kernel eBPF struct layout exactly ----
class FlowKey(ctypes.Structure):
    _fields_ = [
        ("src_ip", ctypes.c_uint32),
        ("src_port", ctypes.c_uint16),
        ("proto", ctypes.c_uint32),
    ]

class DataPoint(ctypes.Structure):
    _fields_ = [
        ("start_ts", ctypes.c_uint64),
        ("last_seen", ctypes.c_uint64),
        ("flow_duration", ctypes.c_uint32),
        ("total_pkts", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint32),
        ("sum_IAT", ctypes.c_uint64),
        ("flow_IAT_mean", ctypes.c_uint32),
        ("flow_pkts_per_s", ctypes.c_uint32),
        ("flow_bytes_per_s", ctypes.c_uint32),
        ("pkts_len_mean", ctypes.c_uint32),
        ("features", ctypes.c_uint32 * 5),
    ]

def flow_key_to_str(fk: FlowKey):
    """Chuyển IP:Port sang dạng string đúng thứ tự byte"""
    ip = socket.inet_ntoa(struct.pack("<I", fk.src_ip))  # little-endian → network order
    port = socket.ntohs(fk.src_port)
    return f"{ip}:{port}"

def get_data(map_path: str):
    """
    Read all entries from pinned BPF map.
    map_path: đường dẫn map (ví dụ "/sys/fs/bpf/eno3/xdp_flow_tracking")
    Return list of (flow_string, DataPoint) tuples.
    """
    map_fd = libbcc.lib.bpf_obj_get(map_path.encode("utf-8"))
    if map_fd < 0:
        raise OSError(f"Cannot open pinned map at {map_path}")

    key = FlowKey()
    next_key = FlowKey()
    value = DataPoint()
    key_set = False
    results = []

    while True:
        key_ptr = ctypes.c_void_p(0) if not key_set else ctypes.byref(key)
        ret = libbcc.lib.bpf_map_get_next_key(map_fd, key_ptr, ctypes.byref(next_key))
        if ret != 0:
            break

        if libbcc.lib.bpf_map_lookup_elem(map_fd, ctypes.byref(next_key), ctypes.byref(value)) == 0:
            flow = flow_key_to_str(next_key)
            dp_copy = DataPoint()
            ctypes.memmove(ctypes.byref(dp_copy), ctypes.byref(value), ctypes.sizeof(DataPoint))
            results.append((flow, dp_copy))

        ctypes.memmove(ctypes.byref(key), ctypes.byref(next_key), ctypes.sizeof(FlowKey))
        key_set = True

    return results

# Nếu chạy trực tiếp thì in ra, có thể truyền map path
if __name__ == "__main__":
    import sys, time

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <map_path>")
        exit(1)

    map_path = sys.argv[1]

    while True:
        flows = get_data(map_path)
        print("\n=== Flow Table ===")
        for i, (flow, dp) in enumerate(flows, 1):
            feats = [
                dp.flow_duration,
                dp.flow_IAT_mean,
                dp.flow_pkts_per_s,
                dp.flow_bytes_per_s,
                dp.pkts_len_mean
            ]
            print(f"{i:<4d} | {flow:<21} | feats={feats} | total_pkts={dp.total_pkts}")
        time.sleep(1)
