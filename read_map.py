#!/usr/bin/env python3
import ctypes
import socket
import struct
from bcc import libbcc


# ==== Khai báo argtypes / restype để tránh segfault ====
libbcc.lib.bpf_obj_get.argtypes = [ctypes.c_char_p]
libbcc.lib.bpf_obj_get.restype = ctypes.c_int

libbcc.lib.bpf_map_get_next_key.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
]
libbcc.lib.bpf_map_get_next_key.restype = ctypes.c_int

libbcc.lib.bpf_map_lookup_elem.argtypes = [
    ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p
]
libbcc.lib.bpf_map_lookup_elem.restype = ctypes.c_int


# ---- DEFINITIONS: phải khớp với struct C ----
class FlowKey(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("src_ip", ctypes.c_uint32),
        ("src_port", ctypes.c_uint16),
        ("dst_ip", ctypes.c_uint32),
        ("dst_port", ctypes.c_uint16),
        ("proto", ctypes.c_uint8),
    ]


class DataPoint(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("start_ts", ctypes.c_uint64),
        ("last_seen", ctypes.c_uint64),
        ("flow_duration", ctypes.c_uint64),
        ("total_pkts", ctypes.c_uint32),
        ("total_bytes", ctypes.c_uint32),
        ("sum_IAT", ctypes.c_uint64),
        ("flow_IAT_mean", ctypes.c_uint32),
        ("flow_pkts_per_s", ctypes.c_uint32),
        ("flow_bytes_per_s", ctypes.c_uint32),
        ("pkts_len_mean", ctypes.c_uint32),
        ("features", ctypes.c_uint32 * 5),
        ("label", ctypes.c_uint32),
        ("__padding", ctypes.c_uint8 * 16),   # thêm padding cho đủ 96B
    ]


def flow_key_to_str(fk: FlowKey) -> str:
    """Convert FlowKey struct to human-readable string"""
    src_ip = socket.inet_ntoa(struct.pack("<I", fk.src_ip))
    dst_ip = socket.inet_ntoa(struct.pack("<I", fk.dst_ip))
    return f"{src_ip}:{fk.src_port} -> {dst_ip}:{fk.dst_port} proto={fk.proto}"


def read_flows(map_path: str):
    """
    Đọc toàn bộ entries từ pinned BPF map
    Trả về list [(flow_key_str, DataPoint)]
    """
    map_fd = libbcc.lib.bpf_obj_get(map_path.encode("utf-8"))
    if map_fd < 0:
        raise OSError(f"Cannot open pinned map at {map_path}")
    print(f"[DEBUG] map_fd = {map_fd}")

    key = FlowKey()
    next_key = FlowKey()
    value = DataPoint()
    key_set = False
    results = []

    while True:
        key_ptr = None if not key_set else ctypes.byref(key)

        ret = libbcc.lib.bpf_map_get_next_key(map_fd, key_ptr, ctypes.byref(next_key))
        if ret != 0:
            break  # hết map

        if libbcc.lib.bpf_map_lookup_elem(map_fd, ctypes.byref(next_key), ctypes.byref(value)) == 0:
            flow = flow_key_to_str(next_key)
            dp_copy = DataPoint()
            ctypes.memmove(ctypes.byref(dp_copy), ctypes.byref(value), ctypes.sizeof(DataPoint))
            results.append((flow, dp_copy))

        ctypes.memmove(ctypes.byref(key), ctypes.byref(next_key), ctypes.sizeof(FlowKey))
        key_set = True

    return results


if __name__ == "__main__":
    import sys, time

    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <map_path>")
        exit(1)

    map_path = sys.argv[1]
    print(f"[DEBUG] Starting read_map loop for {map_path} ...")

    while True:
        try:
            flows = read_flows(map_path)
            print("\n=== Flow Table ===")
            for i, (flow, dp) in enumerate(flows, 1):
                feats = [
                    dp.flow_duration,
                    dp.flow_IAT_mean,
                    dp.flow_pkts_per_s,
                    dp.flow_bytes_per_s,
                    dp.pkts_len_mean,
                ]
                print(f"{i:<4d} | {flow:<30} | feats={feats} | total_pkts={dp.total_pkts}")
        except Exception as e:
            print(f"[ERROR] {e}")
        time.sleep(1)
