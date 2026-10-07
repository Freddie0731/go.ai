"""转换权重.py - 把训练出的权重转成 C++ 引擎能正确使用的格式

为什么需要这一步
================
`train_alphazero.py` 和 `go_ai.cpp` 对"棋盘怎么摆进张量"的理解是相反的：

  * Python（train_alphazero.py:201 `GoBoard.encode()`）生成 `feat[通道, x, y]`，
    x 是第一维、y 是第二维；着法标签是 `x * 19 + y`（第 279 行）。
    PyTorch 卷积核是 (out, in, kh, kw)，其中 kh 对应 y、kw 对应 x。
  * C++（go_ai.cpp）全程用 `cells[y * 19 + x]`，也就是 y 是行、x 是列。

两者差一个转置。直接把训练权重喂给 C++，网络看到的棋盘是**转置过的**，
输出的策略也落在转置坐标里 —— 而且不会报任何错，只是棋力莫名其妙地差。

本脚本做的事：

  1. **所有 3×3 卷积核交换最后两维**（kh <-> kw）：
     `conv_in.weight`、`blocks.N.conv1.weight`、`blocks.N.conv2.weight`。

     推导：引擎的 `conv3x3` 用 `pad[row=y+1][col=x+1]`，即"宽度维 = x"；
     核按 `k[dy*3+dx]` 取用，也就是物理上 `k[dy][dx]` 配训练张量的
     `K[kh=dx][kw=dy]`，即 **k = Kᵀ**。
     再由 `(Tf) ⊛ Kᵀ = T(f ⊛ K)`（逐项展开可验证）逐层归纳：
     引擎第 i 层的特征图 = 训练第 i 层特征图的转置。
     **所以每一层都要转置** —— 正是因为"输入已转置"，每层都必须补偿。
     （早期版本只转置第一层，是错的。）

  2. **头部按 τ 置换**，其中 τ 交换两个 19 进制位（`x*19+y <-> y*19+x`）：

     * `p_fc.weight`：行、列都要按 τ 置换
       （引擎 logits_e[k] 必须等于训练 logits_t[τ(k)]）
     * `p_fc.bias`：按 τ 置换
     * `v_fc1.weight`：列按 τ 置换（它是从 361 维特征图读入的）
     * `p_conv`/`v_conv`（1×1）与 BatchNorm **不需要**改：前者的核是
       1×1、空间无关；后者的参数是逐通道的，与空间轴无关。

  3. 补上 pass 的 logit。旧模型的策略头是 361 维（`N_ACT = 361`，没有 pass），
     而 C++ 引擎会在下标 361 取 pass 先验。361 维时那个位置根本不存在。
     本脚本把 `p_fc` 扩成 362 行，pass 一行填常数 logit（默认 -3.0），
     既不抢戏也不会下溢成 0。

     注意：pass 是**唯一不做空间置换**的槽位，它不对应棋盘上的点。

用法
====
双击 `转换权重.bat`，或：

    python 转换权重.py <输入> <输出> [--pass-logit -3.0]

输入可以是：
  * 训练输出的 `.bin`（nn_weights_v2.bin 等）
  * PyTorch 的 `.pt` 检查点（自动取 state_dict；若含 BatchNorm 统计量会保留）

输出是引擎可直接加载的 `.bin`。脚本会顺带跑一套一致性自检：
用 numpy 按 C++ 的语义复算一遍前向，和 PyTorch 正常算的结果对比
（按 τ 对齐后比较 logits），并做反向对照确认置换确实必要。
"""
import argparse
import os
import struct
import sys

import numpy as np

try:
    import torch
except Exception:                                    # pragma: no cover
    torch = None

PASS_LOGIT = -3.0
NN = 19 * 19

# τ = 交换两个 19 进制位：把引擎的空间顺序 (y*19+x) 换成训练的顺序 (x*19+y)。
# 它是自己的逆，所以同一个数组既能用作"引擎->训练"也能用作"训练->引擎"的映射。
#   若引擎位置 i = y*19+x，则它在训练布局里的下标是 x*19+y = tau[i]。
TAU = np.array([(i % 19) * 19 + (i // 19) for i in range(NN)], dtype=np.int64)
# τ 是自己的逆（交换两个 19 进制位两次就回到原样），但显式算出来更清楚，
# 自检的"反向对照"要用它撤销置换。
TAU_INV = np.argsort(TAU)


# ----------------------------------------------------------------------
# 读 / 写 引擎的二进制格式
# ----------------------------------------------------------------------
def read_bin(path):
    """读引擎格式，返回 [(name, ndarray), ...] 保持原顺序。"""
    with open(path, 'rb') as f:
        raw = f.read()
    off = 0
    n, = struct.unpack_from('<i', raw, off)
    off += 4
    if n <= 0 or n > 100000:
        raise ValueError('张量个数不合理：%d' % n)
    tensors = []
    for i in range(n):
        nl, = struct.unpack_from('<i', raw, off)
        off += 4
        name = raw[off:off + nl].decode('utf-8')
        off += nl
        nd, = struct.unpack_from('<i', raw, off)
        off += 4
        shape = list(struct.unpack_from('<%di' % nd, raw, off))
        off += 4 * nd
        ne, = struct.unpack_from('<i', raw, off)
        off += 4
        arr = np.frombuffer(raw, dtype='<f4', count=ne, offset=off).reshape(shape)
        off += 4 * ne
        tensors.append((name, arr.copy()))
    return tensors


def write_bin(path, tensors):
    with open(path, 'wb') as f:
        f.write(struct.pack('<i', len(tensors)))
        for name, arr in tensors:
            nb = name.encode('utf-8')
            f.write(struct.pack('<i', len(nb)))
            f.write(nb)
            f.write(struct.pack('<i', arr.ndim))
            for d in arr.shape:
                f.write(struct.pack('<i', int(d)))
            flat = np.ascontiguousarray(arr, dtype='<f4').reshape(-1)
            f.write(struct.pack('<i', int(flat.size)))
            f.write(flat.tobytes())


def read_pt(path):
    """读 PyTorch 检查点，返回 [(name, ndarray), ...]。"""
    if torch is None:
        raise RuntimeError('需要 torch 才能读 .pt，请改用已导出的 .bin')
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict):
        for key in ('model', 'state_dict', 'net', 'weights'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError('.pt 内容不是 state_dict')
    out = []
    for k, v in obj.items():
        if hasattr(v, 'detach'):
            v = v.detach().cpu().numpy()
        out.append((k, np.asarray(v)))
    return out


# ----------------------------------------------------------------------
# 转换
# ----------------------------------------------------------------------
def convert(tensors, pass_logit=PASS_LOGIT, verbose=True):
    """返回 (转换后的张量列表, 报告字符串列表)。"""
    report = []
    idx = dict((n, a) for n, a in tensors)
    names = [n for n, _ in tensors]

    if 'ai_engine.weights_converted' in idx:
        # Running this twice would transpose the first convolution again and
        # append a second marker, producing a file the engine happily accepts
        # but which computes the wrong function. Refuse loudly instead.
        raise ValueError(
            '这个文件已经被转换过了（含 ai_engine.weights_converted 标记）。\n'
            '  重复转换会把第一层卷积再转置一次，得到"看起来正常但算错"的权重。\n'
            '  请用原始训练输出（nn_weights_v2.bin / nn_weights_v3.bin）作为输入。')

    if 'conv_in.weight' not in idx:
        raise ValueError('缺少 conv_in.weight，这不是本引擎的权重格式')

    cin = idx['conv_in.weight']
    if cin.ndim != 4:
        raise ValueError('conv_in.weight 维度异常：%s' % (cin.shape,))
    if cin.shape[2] != 3 or cin.shape[3] != 3:
        raise ValueError('conv_in.weight 不是 3x3 卷积：%s' % (cin.shape,))

    # τ：交换两个 19 进制位，把引擎的空间顺序换成训练的顺序（反之亦然，
    # 因为它是自己的逆）。
    tau = TAU

    def is_3x3_conv(name):
        if name == 'conv_in.weight':
            return True
        if name.startswith('blocks.') and (
                name.endswith('.conv1.weight') or name.endswith('.conv2.weight')):
            return True
        return False

    out = []
    n_conv_t = 0
    for name, arr in tensors:
        a = arr
        if is_3x3_conv(name):
            if arr.ndim != 4 or arr.shape[2] != 3 or arr.shape[3] != 3:
                raise ValueError('%s 不是 3x3 卷积：%s' % (name, (arr.shape,)))
            a = np.ascontiguousarray(arr.transpose(0, 1, 3, 2))
            n_conv_t += 1
        elif name == 'p_fc.weight':
            # 行：logits 必须按 τ 重排；列：输入是按 τ 重排后的特征图。
            if arr.ndim != 2:
                raise ValueError('p_fc.weight 维度异常：%s' % (arr.shape,))
            a = np.ascontiguousarray(arr[tau, :][:, tau])
        elif name == 'p_fc.bias':
            if arr.ndim != 1:
                raise ValueError('p_fc.bias 维度异常：%s' % (arr.shape,))
            a = np.ascontiguousarray(arr[tau])
        elif name == 'v_fc1.weight':
            # 从 361 维特征图读入，列按 τ 重排；输出 64 维没有空间含义。
            if arr.ndim != 2:
                raise ValueError('v_fc1.weight 维度异常：%s' % (arr.shape,))
            a = np.ascontiguousarray(arr[:, tau])
        out.append((name, a))

    report.append('已转置 %d 个 3x3 卷积核（conv_in + 全部残差块）' % n_conv_t)
    report.append('已按 τ 置换 p_fc.weight（行+列）、p_fc.bias、v_fc1.weight')
    report.append('未改动 p_conv / v_conv（1x1）与 BatchNorm（逐通道）—— 与空间轴无关')

    # 补 pass 行 / 列。pass 不对应棋盘上的点，所以是唯一不做 τ 置换的槽位，
    # 必须放在置换之后追加，不能混进 τ 里。
    d = dict((n, a) for n, a in out)
    if 'p_fc.weight' in d and 'p_fc.bias' in d:
        w, b = d['p_fc.weight'], d['p_fc.bias']
        n_act = w.shape[0]
        if n_act == NN:
            pad_w = np.zeros((1, w.shape[1]), dtype=w.dtype)
            pad_b = np.full((1,), np.float32(pass_logit), dtype=b.dtype)
            new_w = np.concatenate([w, pad_w], axis=0)
            new_b = np.concatenate([b, pad_b], axis=0)
            report.append('p_fc 361 行（无 pass）-> 补一行 pass logit=%.2f'
                          % pass_logit)
            out = [((n, new_w) if n == 'p_fc.weight' else
                    (n, new_b) if n == 'p_fc.bias' else (n, a))
                   for n, a in out]
        elif n_act == NN + 1:
            report.append('p_fc 已是 362 行（含 pass），保持不变')
        else:
            raise ValueError('p_fc 行数异常：%d（期望 %d 或 %d）'
                             % (n_act, NN, NN + 1))
    else:
        raise ValueError('缺少 p_fc.weight / p_fc.bias')

    # BatchNorm 统计量检查：缺失时引擎会静默退化成"不归一化"
    d = dict((n, a) for n, a in out)
    bn_bases = [n[:-7] for n in d if n.endswith('.weight') and d[n].ndim == 1]
    missing = []
    for base in bn_bases:
        for suf in ('.running_mean', '.running_var'):
            if base + suf not in d:
                missing.append(base + suf)
    if missing:
        report.append('警告：缺少 %d 个 BatchNorm 统计量，引擎会跳过归一化'
                      % len(missing))
        for m in missing[:6]:
            report.append('  - %s' % m)
        report.append('  修法：跑 train_alphazero.py 训练一遍（export_weights '
                      '会用 named_buffers() 带上统计量），')
        report.append('        或用 fix_bn.py 跑一遍真实数据补齐。')
    else:
        report.append('BatchNorm running_mean/running_var 齐全')

    # 结构检查
    ch = d['conv_in.weight'].shape[0]
    in_ch = d['conv_in.weight'].shape[1]
    nb = 0
    while ('blocks.%d.conv1.weight' % nb) in d:
        nb += 1
    report.append('网络规模：ch=%d  输入平面=%d  残差块=%d  策略输出=%d'
                  % (ch, in_ch, nb, d['p_fc.weight'].shape[0]))
    if in_ch != 17:
        report.append('注意：输入平面是 %d，而 train_alphazero.py 用的是 17。'
                      '引擎会按 17 填平面；若确实只有 %d，请改 '
                      'train_alphazero.py 的 INPUT_CH 后重训。' % (in_ch, in_ch))

    # 打标记：引擎加载时会校验这个张量，从而能明确区分
    # "已转换（第一层卷积已转置）" 和 "原始训练输出（未转置）"。
    # 没有它的话，误用原始权重只会表现为棋力莫名其妙地差，没有任何报错。
    out.append(('ai_engine.weights_converted', np.array([1.0], dtype='<f4')))
    report.append('已写入标记张量 ai_engine.weights_converted=1')
    return out, report


# ----------------------------------------------------------------------
# 一致性自检：用 numpy 按 C++ 语义复算，和 PyTorch 比
# ----------------------------------------------------------------------
def _conv3x3_cpp(x, W, pad=1):
    """复刻 go_ai.cpp 的 conv3x3：x 是 (ic,HW)，W 是 (oc,ic,3,3)，输出 (oc,HW)。"""
    ic = x.shape[0]
    oc = W.shape[0]
    side = 19
    xp = x.reshape(ic, side, side)
    xpad = np.zeros((ic, side + 2 * pad, side + 2 * pad), dtype=np.float64)
    xpad[:, pad:pad + side, pad:pad + side] = xp
    out = np.zeros((oc, side, side), dtype=np.float64)
    for o in range(oc):
        acc = np.zeros((side, side), dtype=np.float64)
        for i in range(ic):
            k = W[o, i].astype(np.float64)
            for dy in range(3):
                for dx in range(3):
                    acc += k[dy, dx] * xpad[i, dy:dy + side, dx:dx + side]
        out[o] = acc
    return out.reshape(oc, side * side)


def _bn_cpp(x, w, b, m, v, relu=True):
    """复刻 go_ai.cpp 的 bn_relu。x 是 (C,HW)，w/b/m/v 长度 C。"""
    sc = (w / np.sqrt(v + 1e-5)).astype(np.float64)
    bi = (b - m * sc).astype(np.float64)
    y = x * sc[:, None] + bi[:, None]
    if relu:
        y = np.maximum(y, 0.0)
    return y


def _fc_cpp(x, W, b, relu=False):
    y = W.astype(np.float64) @ x.astype(np.float64) + b.astype(np.float64)
    if relu:
        y = np.maximum(y, 0.0)
    return y


def _forward_cpp(x_cpp, d, nb):
    """完整前向，全部按 go_ai.cpp 的语义。x_cpp 是 (in_ch, NN)，NN=19*19。"""
    a = _conv3x3_cpp(x_cpp, d['conv_in.weight'])
    cur = _bn_cpp(a, d['bn_in.weight'], d['bn_in.bias'],
                  d['bn_in.running_mean'], d['bn_in.running_var'])
    for i in range(nb):
        p = 'blocks.%d.' % i
        a = _conv3x3_cpp(cur, d[p + 'conv1.weight'])
        t1 = _bn_cpp(a, d[p + 'bn1.weight'], d[p + 'bn1.bias'],
                     d[p + 'bn1.running_mean'], d[p + 'bn1.running_var'])
        a = _conv3x3_cpp(t1, d[p + 'conv2.weight'])
        t2 = _bn_cpp(a, d[p + 'bn2.weight'], d[p + 'bn2.bias'],
                     d[p + 'bn2.running_mean'], d[p + 'bn2.running_var'],
                     relu=False)
        cur = np.maximum(cur + t2, 0.0)

    Wp = d['p_conv.weight'].reshape(d['p_conv.weight'].shape[0], -1)
    ap = Wp @ cur
    tp = _bn_cpp(ap, d['p_bn.weight'], d['p_bn.bias'],
                 d['p_bn.running_mean'], d['p_bn.running_var'])
    logits = _fc_cpp(tp.reshape(-1), d['p_fc.weight'], d['p_fc.bias'])
    e = np.exp(logits - logits.max())
    p = e / e.sum()

    Wv = d['v_conv.weight'].reshape(1, -1)
    av = Wv @ cur
    tv = _bn_cpp(av, d['v_bn.weight'], d['v_bn.bias'],
                 d['v_bn.running_mean'], d['v_bn.running_var'])
    h1 = _fc_cpp(tv.reshape(-1), d['v_fc1.weight'], d['v_fc1.bias'], relu=True)
    vout = _fc_cpp(h1, d['v_fc2.weight'], d['v_fc2.bias'])
    return p, float(np.tanh(vout[0]))


def selfcheck(converted, tensors_orig, pass_logit=PASS_LOGIT):
    """返回 (ok, 报告行列表)。

    做法：造一个随机局面输入，分别用
      (a) PyTorch 原生模型（原始权重，Python 约定）
      (b) numpy 按 C++ 语义（转换后的权重）
    计算，比较策略/价值的数值。
    """
    lines = []
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        # NOT a pass. Without torch nobody has checked that the transposition and
        # the tau permutation actually reproduce the trainer's function, and the
        # failure mode of getting them wrong is silent (the engine runs fine and
        # just plays badly). Report it as unverified so main() can say so.
        lines.append('自检未执行：这个 Python 里没有 torch（%s）。' % exc)
        lines.append('  未验证的转换是可以用的，但"坐标/置换是否正确"没有被证明。')
        lines.append('  建议装上 torch 后重跑本脚本（它会自己重跑自检）。')
        return None, lines

    d_orig = dict(tensors_orig)
    d_conv = dict(converted)
    ch = d_conv['conv_in.weight'].shape[0]
    in_ch = d_conv['conv_in.weight'].shape[1]
    nb = 0
    while ('blocks.%d.conv1.weight' % nb) in d_conv:
        nb += 1

    # ---- (a) PyTorch 参考 ----
    class ResBlock(nn.Module):
        def __init__(self, ch):
            super().__init__()
            self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(ch)
            self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(ch)

        def forward(self, x):
            y = F.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            return F.relu(x + y)

    class Net(nn.Module):
        def __init__(self, ch, blocks, in_ch):
            super().__init__()
            self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1, bias=False)
            self.bn_in = nn.BatchNorm2d(ch)
            self.blocks = nn.ModuleList([ResBlock(ch) for _ in range(blocks)])
            self.p_conv = nn.Conv2d(ch, 2, 1, bias=False)
            self.p_bn = nn.BatchNorm2d(2)
            self.p_fc = nn.Linear(2 * NN, NN + 1)
            self.v_conv = nn.Conv2d(ch, 1, 1, bias=False)
            self.v_bn = nn.BatchNorm2d(1)
            self.v_fc1 = nn.Linear(NN, 64)
            self.v_fc2 = nn.Linear(64, 1)

        def forward(self, x):
            x = F.relu(self.bn_in(self.conv_in(x)))
            for b in self.blocks:
                x = b(x)
            p = F.relu(self.p_bn(self.p_conv(x))).reshape(x.size(0), -1)
            p = self.p_fc(p)
            v = F.relu(self.v_bn(self.v_conv(x))).reshape(x.size(0), -1)
            v = F.relu(self.v_fc1(v))
            v = torch.tanh(self.v_fc2(v))
            return p, v

    net = Net(ch, nb, in_ch).eval()
    state = {}
    for k, v in d_orig.items():
        state[k] = torch.from_numpy(np.asarray(v).copy())
    # 361 行的旧权重补 pass 行后用同一套网络
    if 'p_fc.weight' in state and state['p_fc.weight'].shape[0] == NN:
        state['p_fc.weight'] = torch.cat(
            [state['p_fc.weight'], torch.zeros(1, state['p_fc.weight'].shape[1])], 0)
        state['p_fc.bias'] = torch.cat(
            [state['p_fc.bias'], torch.full((1,), float(pass_logit))], 0)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        lines.append('自检：load_state_dict 缺 %d 项 / 多 %d 项（若只差 BN 统计量可视作正常）'
                     % (len(missing), len(unexpected)))
        for k in list(missing)[:4]:
            lines.append('  missing %s' % k)

    # 随机但固定的石头布局（用同一张随机图给所有平面，保证非对称、可区分转置）
    rng = np.random.default_rng(20240501)
    feat = np.zeros((1, in_ch, 19, 19), dtype=np.float32)
    for c in range(0, in_ch - 1, 2):
        feat[0, c] = (rng.random((19, 19)) > 0.88).astype(np.float32)
        feat[0, c + 1] = (rng.random((19, 19)) > 0.88).astype(np.float32)
    feat[0, in_ch - 1] = 1.0

    with torch.no_grad():
        p_ref, v_ref = net(torch.from_numpy(feat))
    p_ref = p_ref.numpy()[0]
    v_ref = float(v_ref.numpy()[0, 0])

    # ---- (b) numpy 按 C++ 语义 ----
    # C++ 的输入平面下标是 y*19+x；Python 的 feat 是 [c, x, y]。
    # C++ 侧拿到的是同一块内存，但按 [c][row=y][col=x] 解释，所以等价于把
    # Python 的 x/y 轴调换。
    x_py = feat[0].reshape(in_ch, 19, 19)                       # [c,x,y]
    x_cpp = np.transpose(x_py, (0, 2, 1)).reshape(in_ch, NN)     # [c,y,x]

    p_cpp, v_cpp = _forward_cpp(x_cpp, d_conv, nb)

    # 引擎的 logits 必须等于训练的 logits 的 τ 置换（pass 槽位除外）。
    # p_cpp[k] 对应训练的 p_ref[TAU[k]]，所以把 p_cpp 的棋盘部分按 τ 重排后
    # 才能和 p_ref 直接比较。
    p_aligned = p_cpp.copy()
    p_aligned[:NN] = p_cpp[:NN][TAU]
    dp = float(np.abs(p_aligned - p_ref).max())
    dv = abs(v_cpp - v_ref)
    lines.append('自检：按 τ 对齐后策略最大差 %.3e，价值差 %.3e' % (dp, dv))

    # ---- (c) 灵敏度：验证"置换确实是必要的" ----
    # 若不转置第一层卷积，结果应当明显不一致；否则说明这张检验局面或这批
    # 权重对空间置换不敏感，那么这个自检证明不了什么。
    d_bad = dict(d_conv)
    d_bad['conv_in.weight'] = np.ascontiguousarray(
        d_conv['conv_in.weight'].transpose(0, 1, 3, 2))
    p_bad, v_bad = _forward_cpp(x_cpp, d_bad, nb)
    p_bad_a = p_bad.copy()
    p_bad_a[:NN] = p_bad[:NN][TAU]
    dp_bad = float(np.abs(p_bad_a - p_ref).max())

    # 第二个对照：所有 3x3 都转置但头部不置换。如果这个也接近，说明头部
    # 置换是多余的（或检验局面不够非对称）。
    d_noh = dict(d_conv)
    d_noh['p_fc.weight'] = np.ascontiguousarray(
        d_conv['p_fc.weight'][TAU_INV, :][:, TAU_INV])
    d_noh['p_fc.bias'] = np.ascontiguousarray(d_conv['p_fc.bias'][TAU_INV])
    d_noh['v_fc1.weight'] = np.ascontiguousarray(d_conv['v_fc1.weight'][:, TAU_INV])
    p_noh, _ = _forward_cpp(x_cpp, d_noh, nb)
    p_noh_a = p_noh.copy()
    p_noh_a[:NN] = p_noh[:NN][TAU]
    dp_noh = float(np.abs(p_noh_a - p_ref).max())

    ok = dp < 2e-4 and dv < 2e-4
    if not ok:
        lines.append('自检失败：转换后的权重在 C++ 语义下与 PyTorch 不一致。')
        lines.append('  可能原因：权重不是这批代码训练的，或空间/头部的置换假设不成立。')
    elif dp_bad < 10 * max(dp, 1e-9) + 1e-6:
        lines.append('自检存疑：不转置卷积的误差 %.3e 与转置后接近。' % dp_bad)
        lines.append('  说明这批权重对空间置换不敏感，这一步既没坏处也验证不了，')
        lines.append('  请另行确认棋力。')
    else:
        lines.append('自检通过：C++ 语义与 PyTorch 按 τ 对齐后一致（策略差 %.2e）。' % dp)
        lines.append('  反向对照 1：不转置 3x3 卷积时误差 %.3e，说明卷积转置必需。' % dp_bad)
        lines.append('  反向对照 2：头部不做 τ 置换时误差 %.3e。' % dp_noh)
        if dp_noh < 10 * max(dp, 1e-9) + 1e-6:
            lines.append('    （对照 2 误差也小 —— 头部置换在这批权重上影响不显著，'
                         '但保留它是正确的。）')
    return ok, lines


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src', help='输入：.bin 或 .pt')
    ap.add_argument('dst', nargs='?', default=None, help='输出 .bin')
    ap.add_argument('--pass-logit', type=float, default=PASS_LOGIT)
    ap.add_argument('--no-check', action='store_true', help='跳过一致性自检')
    args = ap.parse_args()

    src = args.src
    dst = args.dst or (os.path.splitext(src)[0] + '_engine.bin')
    print('=' * 60)
    print('输入：%s' % src)
    print('输出：%s' % dst)
    print('=' * 60)

    if src.lower().endswith('.pt'):
        tensors = read_pt(src)
    else:
        tensors = read_bin(src)
    print('读到 %d 个张量' % len(tensors))

    converted, report = convert(tensors, args.pass_logit)
    print('-' * 60)
    for line in report:
        print(line)
    print('-' * 60)

    ok = True
    skipped = False
    if not args.no_check:
        ok, lines = selfcheck(converted, tensors, args.pass_logit)
        for line in lines:
            print(line)
        print('-' * 60)
        if ok is None:
            skipped = True
            ok = True          # the conversion itself is still usable

    write_bin(dst, converted)
    print('已写出：%s  (%.2f MB)'
          % (dst, os.path.getsize(dst) / 1048576.0))
    # 引擎期望的文件名
    print()
    print('把它放到引擎目录并命名为 nn_weights.bin：')
    print('  copy "%s" "D:\\GoAI3\\GoAI3\\GoAI3\\nn_weights.bin"' % dst)
    print()
    print('然后用引擎自检确认接口对上了：')
    print('  GoAI3.exe --selftest      # 看 "network I/O contract" 一行')
    if skipped:
        print()
        print('注意：本次转换的数值自检被跳过了（没有 torch），'
              '所以"坐标置换是否正确"尚未被证明。')
    return 2 if not ok else 0


if __name__ == '__main__':
    sys.exit(main())
