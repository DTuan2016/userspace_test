#!/usr/bin/env python3
import ctypes
import socket
import struct
from bcc import libbcc


# ---- DEFINITIONS: must MATCH the kernel eBPF struct layout exactly ----
# Added _pack_=1 to ensure C-style memory alignment and prevent Segmentation Faults.
class FlowKey(ctypes.Structure):
    _pack_ = 1  # Ensures C-style packing
    _fields_ = [
        ("src_ip", ctypes.c_uint32),
        ("dst_ip", ctypes.c_uint32),
        ("src_port", ctypes.c_uint16),
        ("dst_port", ctypes.c_uint16),
        ("proto", ctypes.c_uint8),
    ]

# Added _pack_=1 to ensure C-style memory alignment and prevent Segmentation Faults.
class DataPoint(ctypes.Structure):
    _pack_ = 1  # Ensures C-style packing
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
        ("label", ctypes.c_uint32),
    ]

def flow_key_to_str(fk: FlowKey) -> str:
    """Convert FlowKey struct to human-readable string"""
    # IP addresses are stored in network byte order in BPF, but are read as host byte order
    # in Python due to ctypes/struct.pack usage. Using '<I' handles the conversion back.
    src_ip = socket.inet_ntoa(struct.pack("<I", fk.src_ip))
    dst_ip = socket.inet_ntoa(struct.pack("<I", fk.dst_ip))
    return f"{src_ip}:{fk.src_port} -> {dst_ip}:{fk.dst_port} proto={fk.proto}"

def read_flows(map_path: str):
    """
    Return all entries from pinned map
    Return list of tuple (flow_tracking, DataPoint).
    """
    map_fd = libbcc.lib.bpf_obj_get(map_path.encode("utf-8"))
    if map_fd < 0:
        raise OSError(f"Cannot open pinned map at {map_path}")

    # Initialize key, next_key, and value structs
    key = FlowKey()
    next_key = FlowKey()
    value = DataPoint()
    key_set = False
    results = []

    # Iterate through the BPF map
    while True:
        # Determine the pointer for the current key
        key_ptr = ctypes.c_void_p(0) if not key_set else ctypes.byref(key)
        
        # Get the next key
        ret = libbcc.lib.bpf_map_get_next_key(map_fd, key_ptr, ctypes.byref(next_key))
        if ret != 0:
            # End of map or error (ret < 0)
            break

        # Look up the element (value) associated with the next key
        if libbcc.lib.bpf_map_lookup_elem(map_fd, ctypes.byref(next_key), ctypes.byref(value)) == 0:
            flow = flow_key_to_str(next_key)
            
            # Create a copy of the DataPoint returned by the kernel
            # This is crucial because the 'value' struct is reused in the lookup call.
            dp_copy = DataPoint()
            ctypes.memmove(ctypes.byref(dp_copy), ctypes.byref(value), ctypes.sizeof(DataPoint))
            results.append((flow, dp_copy))

        # Update 'key' for the next iteration
        ctypes.memmove(ctypes.byref(key), ctypes.byref(next_key), ctypes.sizeof(FlowKey))
        key_set = True

    return results

if __name__ == "__main__":
    import sys, time
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <map_path>")
        exit(1)

    map_path = sys.argv[1]
    while True:
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
            print(f"{i:<4d} | {flow:<21} | feats={feats} | total_pkts={dp.total_pkts}")
        time.sleep(1)
