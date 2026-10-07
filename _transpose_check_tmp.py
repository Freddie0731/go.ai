"""Review harness for 转换权重.py -- NOT executed from the reviewing session
(the review sandbox had no usable process backend), so treat it as a
hand-checked script to run yourself:

    python _transpose_check_tmp.py

It builds a random-but-generic weight set, computes the trainer's forward pass
(layout [c][x=col][y=row], PyTorch semantics) in numpy, and then checks which
conversion of the weights makes the engine's forward pass (the shipped
转换权重.py helpers _conv3x3_cpp/_bn_cpp/_fc_cpp, i.e. go_ai.cpp semantics,
layout [c][y=row][x=col]) reproduce it.

Engine slot k means the physical point (col = k%19, row = k//19); the trainer's
slot m means (col = m//19, row = m%19); both describe the same point iff
m = tau(k) with tau the digit swap.  So the only correct requirement is

    logits_engine[k] == logits_trainer[tau(k)]

for every k, which is exactly the permutation the shipped selfcheck() never
applies.
"""
import importlib.util

import numpy as np

N = 19
HW = N * N
PATH = r"D:\GoAI3\GoAI3\GoAI3\转换权重.py"

spec = importlib.util.spec_from_file_location("convw", PATH)
convw = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convw)


# ---------------------------------------------------------------- weights
def make_weights(ic=5, ch=6, nb=2, seed=7):
    rng = np.random.default_rng(seed)
    f = lambda a: np.asarray(a, dtype=np.float32)
    d = {}
    d["conv_in.weight"] = f(rng.normal(0, 0.4, (ch, ic, 3, 3)))
    d["bn_in.weight"] = f(np.abs(rng.normal(1, 0.2, ch)))
    d["bn_in.bias"] = f(rng.normal(0, 0.2, ch))
    d["bn_in.running_mean"] = f(rng.normal(0, 0.3, ch))
    d["bn_in.running_var"] = f(np.abs(rng.normal(1, 0.2, ch)))
    for i in range(nb):
        p = "blocks.%d." % i
        d[p + "conv1.weight"] = f(rng.normal(0, 0.3, (ch, ch, 3, 3)))
        d[p + "conv2.weight"] = f(rng.normal(0, 0.3, (ch, ch, 3, 3)))
        for bn in ("bn1", "bn2"):
            d[p + bn + ".weight"] = f(np.abs(rng.normal(1, 0.2, ch)))
            d[p + bn + ".bias"] = f(rng.normal(0, 0.2, ch))
            d[p + bn + ".running_mean"] = f(rng.normal(0, 0.3, ch))
            d[p + bn + ".running_var"] = f(np.abs(rng.normal(1, 0.2, ch)))
    d["p_conv.weight"] = f(rng.normal(0, 0.4, (2, ch, 1, 1)))
    d["p_bn.weight"] = f(np.abs(rng.normal(1, 0.2, 2)))
    d["p_bn.bias"] = f(rng.normal(0, 0.2, 2))
    d["p_bn.running_mean"] = f(rng.normal(0, 0.3, 2))
    d["p_bn.running_var"] = f(np.abs(rng.normal(1, 0.2, 2)))
    d["p_fc.weight"] = f(rng.normal(0, 0.05, (HW + 1, 2 * HW)))
    d["p_fc.bias"] = f(rng.normal(0, 0.2, HW + 1))
    d["v_conv.weight"] = f(rng.normal(0, 0.4, (1, ch, 1, 1)))
    d["v_bn.weight"] = f(np.abs(rng.normal(1, 0.2, 1)))
    d["v_bn.bias"] = f(rng.normal(0, 0.2, 1))
    d["v_bn.running_mean"] = f(rng.normal(0, 0.3, 1))
    d["v_bn.running_var"] = f(np.abs(rng.normal(1, 0.2, 1)))
    d["v_fc1.weight"] = f(rng.normal(0, 0.05, (64, HW)))
    d["v_fc1.bias"] = f(rng.normal(0, 0.2, 64))
    d["v_fc2.weight"] = f(rng.normal(0, 0.1, (1, 64)))
    d["v_fc2.bias"] = f(rng.normal(0, 0.2, 1))
    return d


# ------------------------------------------------- trainer side (PyTorch math)
def bn_t(x, w, b, m, v):
    sc = (w / np.sqrt(v + 1e-5)).astype(np.float64)
    return x * sc[:, None, None] + (b - m * sc)[:, None, None]


def conv_train(X, K):
    """X (ic,N,N) axes (c, x=col, y=row); K (oc,ic,kh,kw), kh along x."""
    p = np.zeros((X.shape[0], N + 2, N + 2))
    p[:, 1:-1, 1:-1] = X
    out = np.zeros((K.shape[0], N, N))
    for dh in range(3):
        for dw in range(3):
            out += np.einsum("oi,imn->omn", K[:, :, dh, dw], p[:, dh:dh + N, dw:dw + N])
    return out


def forward_train(X, d, nb):
    g = lambda k: d[k].astype(np.float64)
    cur = np.maximum(bn_t(conv_train(X, g("conv_in.weight")), g("bn_in.weight"),
                          g("bn_in.bias"), g("bn_in.running_mean"),
                          g("bn_in.running_var")), 0)
    for i in range(nb):
        p = "blocks.%d." % i
        t1 = np.maximum(bn_t(conv_train(cur, g(p + "conv1.weight")), g(p + "bn1.weight"),
                             g(p + "bn1.bias"), g(p + "bn1.running_mean"),
                             g(p + "bn1.running_var")), 0)
        t2 = bn_t(conv_train(t1, g(p + "conv2.weight")), g(p + "bn2.weight"),
                  g(p + "bn2.bias"), g(p + "bn2.running_mean"), g(p + "bn2.running_var"))
        cur = np.maximum(cur + t2, 0)
    tp = np.maximum(bn_t(np.einsum("oi,imn->omn", g("p_conv.weight")[:, :, 0, 0], cur),
                         g("p_bn.weight"), g("p_bn.bias"), g("p_bn.running_mean"),
                         g("p_bn.running_var")), 0)
    logits = g("p_fc.weight") @ tp.reshape(-1) + g("p_fc.bias")   # c*361 + col*19+row
    tv = np.maximum(bn_t(np.einsum("oi,imn->omn", g("v_conv.weight")[:, :, 0, 0], cur),
                         g("v_bn.weight"), g("v_bn.bias"), g("v_bn.running_mean"),
                         g("v_bn.running_var")), 0)
    h1 = np.maximum(g("v_fc1.weight") @ tv.reshape(-1) + g("v_fc1.bias"), 0)
    v = np.tanh(g("v_fc2.weight") @ h1 + g("v_fc2.bias"))
    return logits, float(v[0])


# ------------------------- engine side, using the SHIPPED helpers from the file
def forward_engine(x_cpp, d, nb):
    g = lambda k: d[k]
    cur = convw._bn_cpp(convw._conv3x3_cpp(x_cpp, g("conv_in.weight")),
                        g("bn_in.weight"), g("bn_in.bias"),
                        g("bn_in.running_mean"), g("bn_in.running_var"))
    for i in range(nb):
        p = "blocks.%d." % i
        t1 = convw._bn_cpp(convw._conv3x3_cpp(cur, g(p + "conv1.weight")),
                           g(p + "bn1.weight"), g(p + "bn1.bias"),
                           g(p + "bn1.running_mean"), g(p + "bn1.running_var"))
        t2 = convw._bn_cpp(convw._conv3x3_cpp(t1, g(p + "conv2.weight")),
                           g(p + "bn2.weight"), g(p + "bn2.bias"),
                           g(p + "bn2.running_mean"), g(p + "bn2.running_var"),
                           relu=False)
        cur = np.maximum(cur + t2, 0.0)
    Wp = g("p_conv.weight").reshape(g("p_conv.weight").shape[0], -1)
    tp = convw._bn_cpp(Wp @ cur, g("p_bn.weight"), g("p_bn.bias"),
                       g("p_bn.running_mean"), g("p_bn.running_var"))
    logits = convw._fc_cpp(tp.reshape(-1), g("p_fc.weight"), g("p_fc.bias"))
    Wv = g("v_conv.weight").reshape(1, -1)
    tv = convw._bn_cpp(Wv @ cur, g("v_bn.weight"), g("v_bn.bias"),
                       g("v_bn.running_mean"), g("v_bn.running_var"))
    h1 = convw._fc_cpp(tv.reshape(-1), g("v_fc1.weight"), g("v_fc1.bias"), relu=True)
    vout = convw._fc_cpp(h1, g("v_fc2.weight"), g("v_fc2.bias"))
    return logits, float(np.tanh(vout[0]))


# ------------------------------------------------------------ conversions
def t3(W):
    return np.ascontiguousarray(np.asarray(W).transpose(0, 1, 3, 2))


def perm_move_rows(M):
    M = np.asarray(M)
    top = M[:HW].reshape(N, N, M.shape[1]).transpose(1, 0, 2).reshape(HW, M.shape[1])
    return np.concatenate([top, M[HW:]], axis=0)


def perm_pos_cols(M):                      # (R, 722) -> within each 361 block
    M = np.asarray(M)
    return M.reshape(M.shape[0], 2, N, N).transpose(0, 1, 3, 2).reshape(M.shape[0], 2 * HW)


def perm_1x361(M):
    M = np.asarray(M)
    return M.reshape(M.shape[0], N, N).transpose(0, 2, 1).reshape(M.shape[0], HW)


def build_variant(d, mode):
    o = dict((k, np.array(v, copy=True)) for k, v in d.items())
    if mode in ("c_in", "all3", "full"):
        o["conv_in.weight"] = t3(d["conv_in.weight"])
    if mode in ("all3", "full"):
        for name, a in d.items():
            if name.startswith("blocks.") and name.endswith(".weight") and a.ndim == 4:
                o[name] = t3(a)
    if mode == "full":
        o["p_fc.weight"] = perm_move_rows(perm_pos_cols(d["p_fc.weight"]))
        o["p_fc.bias"] = perm_move_rows(d["p_fc.bias"][:, None])[:, 0]
        o["v_fc1.weight"] = perm_1x361(d["v_fc1.weight"])
    return o


def main():
    ic, ch, nb = 5, 6, 2
    d = make_weights(ic, ch, nb)
    rng = np.random.default_rng(11)
    X = (rng.random((ic, N, N)) > 0.6).astype(np.float64)   # trainer layout
    X[:, 3, 7] = 1.0
    X[:, 12, 4] = 0.0
    Xe = X.transpose(0, 2, 1).reshape(ic, HW)               # engine layout
    ref_logits, ref_v = forward_train(X, d, nb)
    M = ref_logits[:HW].reshape(N, N)
    ref_perm = np.concatenate([M.T.reshape(-1), ref_logits[HW:]])
    print("torch in this interpreter:", end=" ")
    try:
        import torch
        print(torch.__version__)
    except Exception as e:
        print("NO (%s) -- the shipped selfcheck() would silently return ok=True" % e)

    K = d["conv_in.weight"].astype(np.float64)
    rhs = conv_train(X, K).transpose(0, 2, 1)
    print("\nlayer-1 identity  max|conv3x3(T X, K^T) - T conv_train(X, K)| = %.3e"
          % np.abs(convw._conv3x3_cpp(Xe, t3(K).astype(np.float32)).reshape(ch, N, N) - rhs).max())
    print("                  max|conv3x3(T X, K  ) - T conv_train(X, K)| = %.3e"
          % np.abs(convw._conv3x3_cpp(Xe, K.astype(np.float32)).reshape(ch, N, N) - rhs).max())

    names = {"raw": "no conversion",
             "c_in": "转换权重.py as written (conv_in only)",
             "all3": "all 3x3 transposed, p_fc/v_fc1 raw",
             "full": "all 3x3 + p_fc both axes + v_fc1 cols"}
    print("\n%-5s %-13s %-13s %-11s  %s" % ("mode", "err vs ref[tau]", "err vs ref", "value err", "meaning"))
    for mode in ("raw", "c_in", "all3", "full"):
        lg, vv = forward_engine(Xe, build_variant(d, mode), nb)
        print("%-5s %-13.3e %-13.3e %-11.3e  %s"
              % (mode, np.abs(lg - ref_perm).max(), np.abs(lg - ref_logits).max(),
                 abs(vv - ref_v), names[mode]))

    print("\nwhat the shipped selfcheck() actually tests (softmax vs raw logits):")
    for mode in ("raw", "c_in", "all3", "full"):
        lg, vv = forward_engine(Xe, build_variant(d, mode), nb)
        sm = np.exp(lg - lg.max())
        sm = sm / sm.sum()
        print("  dp(%-5s) = max|softmax(logits) - torch logits| = %.3e" % (mode, np.abs(sm - ref_logits).max()))

    converted, _ = convw.convert([(k, v) for k, v in d.items()])
    dc = dict(converted)
    p_cpp, v_cpp = convw._forward_cpp(Xe, dc, nb)
    print("\nshipped convert() + _forward_cpp: dp = %.3e  dv = %.3e  (ok flag = %s)"
          % (np.abs(p_cpp - ref_logits).max(), abs(v_cpp - ref_v), False))
    print("convert() rewrote blocks.0.conv1.weight?",
          not np.array_equal(dc["blocks.0.conv1.weight"], d["blocks.0.conv1.weight"]))
    print("convert() rewrote p_fc.weight?         ",
          not np.array_equal(dc["p_fc.weight"], d["p_fc.weight"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
