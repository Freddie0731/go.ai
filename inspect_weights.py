"""Inspect the GoAI weight files.

Usage:
    python inspect_weights.py [path-to-.bin] [path-to-.pt]

Prints:
  * the geometry the C++ engine derives from the .bin
    (channels, input channels, residual blocks, tensor total), so we can tell
    whether nn_weights.bin will actually load;
  * the first tensor names and every shape, to confirm the naming scheme the
    engine's loader requires (conv_in.weight, blocks.N.conv1.weight, ...);
  * the contents of the matching .pt checkpoint.

No third-party packages are needed for the .bin part; torch is only used for
the .pt if it happens to be installed.
"""
import os
import struct
import sys


def read_bin(path):
    with open(path, "rb") as f:
        data = f.read()
    print("=" * 62)
    print("BIN  %s" % path)
    print("     size %d bytes" % len(data))
    off = 0
    n, = struct.unpack_from("<i", data, off)
    off += 4
    print("     tensor count %d" % n)
    if n <= 0 or n > 100000:
        print("     !! implausible tensor count; not this format?")
        return None
    tensors = []
    for i in range(n):
        nl, = struct.unpack_from("<i", data, off)
        off += 4
        if nl <= 0 or nl > 500:
            print("     !! bad name length %d at tensor %d" % (nl, i))
            return None
        name = data[off:off + nl].decode("utf-8", "replace")
        off += nl
        nd, = struct.unpack_from("<i", data, off)
        off += 4
        shape = list(struct.unpack_from("<%di" % nd, data, off))
        off += 4 * nd
        ne, = struct.unpack_from("<i", data, off)
        off += 4
        off += 4 * ne
        tensors.append((name, shape, ne))
    if off != len(data):
        print("     note: %d trailing bytes after the last tensor"
              % (len(data) - off))
    idx = dict((t[0], t) for t in tensors)
    print("-" * 62)
    print("derived geometry")
    print("  names[0..7]      : %s" % ", ".join(t[0] for t in tensors[:8]))
    if "conv_in.weight" in idx:
        sh = idx["conv_in.weight"][1]
        ch = sh[0] if len(sh) > 0 else -1
        in_ch = sh[1] if len(sh) > 1 else -1
        nb = 0
        while ("blocks.%d.conv1.weight" % nb) in idx:
            nb += 1
        print("  ch (filters)     : %d" % ch)
        print("  in_ch (planes)   : %d" % in_ch)
        print("  residual blocks  : %d" % nb)
        total = sum(t[2] for t in tensors)
        print("  total floats     : %d (%.2f MB of float32)"
              % (total, total * 4 / 1048576.0))
    else:
        print("  !! conv_in.weight missing -> the C++ loader will reject it")
    print("all tensors (name, shape, elements):")
    for name, shape, ne in tensors:
        print("  %-34s %-18s %d" % (name, str(shape), ne))
    return tensors


def read_pt(path):
    print("=" * 62)
    print("PT   %s" % path)
    print("     size %d bytes" % os.path.getsize(path))
    try:
        import torch
    except Exception as exc:                      # torch not installed
        print("     torch not available (%s)" % exc)
        print("     -> only the .bin geometry above can be checked")
        return
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print("     torch.load failed: %s" % exc)
        return
    if isinstance(obj, dict):
        print("     top-level keys: %s" % ", ".join(str(k) for k in obj.keys()))
        for key in ("ch", "n_blocks", "in_ch", "blocks", "config", "args",
                    "arch", "epoch", "iters", "best", "step"):
            if key in obj:
                val = obj[key]
                if isinstance(val, (int, float, str, bool)) or val is None:
                    print("       %s = %r" % (key, val))
        for key, val in obj.items():
            if hasattr(val, "shape"):
                print("     %s.shape = %s" % (key, tuple(val.shape)))
        outer = None
        for key in ("model", "state_dict", "net", "weights"):
            if key in obj and isinstance(obj[key], dict):
                outer = key
                break
        if outer:
            inner = obj[outer]
            print("     state_dict '%s' has %d entries" % (outer, len(inner)))
            for k in list(inner.keys())[:12]:
                v = inner[k]
                print("       %-34s %s" % (k, tuple(v.shape) if hasattr(v, "shape") else type(v)))
        if "blocks" in obj and hasattr(obj["blocks"], "__len__"):
            print("     sequence 'blocks' length = %d" % len(obj["blocks"]))
            first = obj["blocks"][0]
            for k, v in first.items():
                if hasattr(v, "shape"):
                    print("       blocks[0].%-24s %s" % (k, tuple(v.shape)))
    else:
        print("     type %s" % type(obj))
        print("     %r" % (obj,))


def main():
    args = sys.argv[1:]
    if not args:
        args = [r"C:\Users\小米\Desktop\goai\nn_weights_v3.bin",
                r"C:\Users\小米\Desktop\goai\nn_weights_v3.bin.pt"]
    for path in args:
        if not os.path.exists(path):
            print("missing: %s" % path)
            continue
        if path.lower().endswith(".pt"):
            read_pt(path)
        else:
            read_bin(path)


if __name__ == "__main__":
    main()
