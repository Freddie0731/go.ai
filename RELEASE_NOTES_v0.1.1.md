# GoAI3 v0.1.1

**首次编译验证通过的版本。** v0.1.0 是一个"代码写完但从未编译"的版本；这一版把那份代码真正跑起来了，并修掉三个只有真机运行才会暴露的问题。

---

## 本版重点

### 1. 首次通过编译与自检 ✅

在 Visual Studio 2026（工具集 `v145`）x64 上实测：

| 项目 | 结果 |
|---|---|
| 编译 | **零错误** |
| `--selftest` | **5/5 通过**（对称置换、坐标变换一致性、**合法着法枚举精确 205 手**、先验归一化、空盘搜索） |
| 权重加载 | `Neural net loaded: ch=32 blocks=4 in_ch=17 layout=converted-in-memory` |
| 权重接口契约 | `policy=362 (want 362) in_ch=17 (want 17)` |
| 网络参与搜索 | `nn=yes`，先验 `max=0.3105` |
| `nn/sim` 性能契约 | **1.00–1.02**（`symmetries=1` 时理论值 1.0） |
| 搜索速度 | 约 320–360 visits/s（2 线程，19×19） |

其中 **`nn/sim = 1.00`** 尤其值得注意：它证明 v0.1.0 里"每次模拟对叶子做两次网络前向"那个性能缺陷确实被修好了——若缺陷仍在，这里会是 `2.00`。

**`layout=converted-in-memory`** 则证明引擎的自动布局转换正确工作：未带转换标记的原始权重也能被正确使用。

### 2. 修复：命令行构建缺少 `gdi32.lib` / `user32.lib`

裸 `cl` 命令行下链接失败，报出大量 `LNK2019 无法解析的外部符号`：

- `CreateFontA`、`LineTo`、`Ellipse`、`BitBlt` … → **`gdi32.lib`**
- `CreateWindowEx`、`BeginPaint`、`MessageBoxA`、`GetSystemMetrics` … → **`user32.lib`**

Visual Studio 会通过默认库集合隐式链接这些 Windows 导入库，**裸 `cl` 不会**。这正是"VS 里能过、命令行过不了"的原因。

修法：在源码里显式声明全部所需导入库，使任何编译方式都自动生效——

```cpp
#pragma comment(lib, "gdiplus.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "comctl32.lib")
```

### 3. 修复：权重加载失败的原因被吞掉

`NeuralNet::load()` 过去只返回 `bool`，因此**所有**失败——文件不存在、张量缺失、尺寸不符、文件截断——都打印同一句：

```
No nn_weights.bin: the search will use heuristic priors ...
```

这句话会把人引向错误的方向：文件往往就在那里，问题却在别处。（开发过程中它确实导致了误判。）

现在 `load()` 记录**具体原因**，调用方还会先探测文件是否真的存在，据此给出完全不同的指引：

```
nn_weights.bin exists but could not be used, so no network is loaded. Reported reason:
  'nn_weights.bin' p_fc.weight has 361 policy outputs, expected 362 (361 board points
  + pass). Convert a 361-row file with the weight converter first.
```

覆盖的诊断包括：无法打开（带系统错误码）、文件过短、张量损坏（报出第几个/共几个）、缺失具体张量名、`p_fc.weight` 列数与行数不符、`p_fc.bias` 长度不符。

### 4. 修复：批处理脚本编译失败却报成功

`构建并自检.bat` 把 `set "BLD=!ERRORLEVEL!"` 放在 `if defined ... ( ... )` 块内，随后的 `if not "!BLD!"=="0"` 会被 `call` 污染成 0，于是**编译失败也打印 `[OK] build + self test finished`**。

现在改为**检查产物文件是否真的生成**，不再依赖退出码——链接器因文件被占用而失败时退出码可能是 0，只看退出码骗不了这个检查。

### 5. 新增 `编译验证.bat`

一个更可靠的一键验证脚本：

- 自动定位 MSVC（`vswhere` + 多个回退路径），再回退 `g++`
- 产物写成 `GoAI3_new.exe`，**避开旧 exe 被占用导致的链接失败**
- 失败判定基于产物文件，而非退出码
- 自动运行 `--selftest`，并把 ANSI 日志转成 UTF-8 写入 `build_log.txt`
- 用 `pushd` 切到输出目录再运行，使引擎能找到同目录的 `nn_weights.bin`

### 6. 新增 GitHub Actions CI

`.github/workflows/build.yml`：每次 push / PR 在 `windows-latest` 上编译并运行自检，日志作为 artifact 上传。

CI **不需要** `nn_weights.bin`（100+ MB，不在仓库内）：自检在无网络时跳过网络契约检查，只验证搜索不变量与规则枚举，因此可以在没有训练权重的机器上运行。

---

## 从源码构建

```bat
:: 一键（推荐）
编译验证.bat

:: 或手动 MSVC
cl /O2 /EHsc /std:c++17 /W3 /DNDEBUG /Fe:GoAI3.exe go_ai.cpp /link gdiplus.lib gdi32.lib user32.lib

:: 或 MinGW
g++ -O3 -march=native -std=c++17 -pthread go_ai.cpp -o GoAI3.exe -mwindows -static -lgdiplus -lgdi32 -luser32
```

## 权重

本仓库**不包含**神经网络权重。用你自己的训练产物转换后，把 `nn_weights.bin` 放在**当前工作目录**（引擎按工作目录查找，不是 exe 所在目录）：

```bat
python 转换权重.py nn_weights_v3.bin nn_weights.bin
```

---

## ⚠️ 仍未验证的部分

**编译与自检通过，不等于棋力可用。** 以下都还没有实测：

1. **对局棋力完全未知。** `--selftest` 检查搜索不变量与规则枚举，**不衡量棋力**。需要与基准引擎跑 SGF 对局才能量化。
2. **价值头偏弱。** 基准中网络给出的胜率在 0.5%–57% 之间摆动（visits 仅 45–1080，且权重只经 SGF 监督学习、无 RL 闭环）。
3. **GUI 交互与 GTP 对接未运行过。** 只验证了 `--selftest` 与 `--benchmark` 两条路径。
4. **贴目不进入网络**（输入平面无贴目，训练固定 `komi=7.5`）。
5. **棋谱坐标约定未验证**：训练脚本与另一 Python 引擎的 SGF 映射方向相反，交叉使用会导致棋谱转置。

---

## 下一步

1. 验证网络真的提升搜索质量：`--benchmark --symmetries 1` 与 `8` 对比，`--benchmark --no-nn` 与加载网络对比（目前只确认了 `nn/sim` 契约）。
2. 实战验证：`--cli` 接入 Sabaki，或 `--selfplay` 生成棋谱检查着法质量。
3. 走向完整 KataGo 路线：自对弈产生带 policy 目标的训练数据、加 ownership 辅助头、迭代式训练闭环。

---

## 许可

MIT
