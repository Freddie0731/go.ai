// go_ai.cpp - C++ Go AI with NN + Selfplay + Random Color
// MSVC: cl /O2 /EHsc /std:c++17 /Fe:go_ai.exe go_ai.cpp /link gdiplus.lib gdi32.lib user32.lib
// GCC:  g++ -O3 -march=native -std=c++17 -pthread go_ai.cpp -o go_ai.exe -mwindows -static -lgdiplus -lgdi32 -luser32
// (gdiplus for Gdiplus::*, gdi32 for the native GDI drawing calls in the GUI)

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <objidl.h>
#include <gdiplus.h>
#pragma comment(lib, "gdiplus.lib")
// The GUI calls the Win32 API directly, and Visual Studio links the standard
// Windows import libraries implicitly through its default library set. A bare
// "cl" command line does not, so every Win32 import must be named explicitly or
// it surfaces as LNK2019 "unresolved external symbol" (one round per missing
// library, which is why they are all listed here up front):
//   gdiplus -> Gdiplus::*            gdi32   -> CreateFontA, LineTo, Ellipse, BitBlt, ...
//   user32  -> CreateWindowEx, BeginPaint, MessageBoxA, GetSystemMetrics, ...
//   kernel32 -> GetModuleHandle, QueryPerformanceCounter, CreateThread, ...
//   shell32 -> ShellExecute*, SHGetFolderPath, ...   comctl32 -> InitCommonControlsEx, ...
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "comctl32.lib")
#undef near
#undef far
#undef min
#undef max
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// Bumped by hand whenever the search or the weight contract changes. Printed at
// startup so that "am I running the binary I just built?" is answerable from the
// console output alone -- a stale exe looks exactly like a logic bug otherwise.
#define GOAI_BUILD_TAG "2025-06-14-katago-mcts-3"

// ============================================================
// Constants
// ============================================================
const int N = 19;
const int NN = N * N;
const int8_t EMPTY = 0, BLACK = 1, WHITE = 2;
const int PASS = -1;
const int DX[4] = { -1, 1, 0, 0 };
const int DY[4] = { 0, 0, -1, 1 };

int NEIGHBORS[NN][4];
int NCOUNT[NN];
uint64_t ZB[NN], ZW[NN], ZT;

int STAR_POINTS[9] = {
    3 * N + 3, 3 * N + 9, 3 * N + 15,
    9 * N + 3, 9 * N + 9, 9 * N + 15,
    15 * N + 3, 15 * N + 9, 15 * N + 15
};

// Handicap stones placement (19x19)
int HANDICAP_POINTS[9] = {
    3 * N + 15, 15 * N + 3, 15 * N + 15, 3 * N + 3,   // 4 corner star
    3 * N + 9, 9 * N + 3, 15 * N + 9, 9 * N + 15,      // 4 side star
    9 * N + 9                                // center
};

void init_tables() {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            int i = y * N + x, c = 0;
            for (int d = 0; d < 4; ++d) {
                int nx = x + DX[d], ny = y + DY[d];
                if (nx >= 0 && nx < N && ny >= 0 && ny < N)
                    NEIGHBORS[i][c++] = ny * N + nx;
            }
            NCOUNT[i] = c;
        }
    }
    std::mt19937_64 rng(0x5DEECE66DULL);
    for (int i = 0; i < NN; ++i) { ZB[i] = rng(); ZW[i] = rng(); }
    ZT = rng();
}

// ============================================================
// Board
// ============================================================
struct Board {
    int8_t cells[NN];
    int ko;
    int8_t to_move;
    int passes;
    uint64_t hash;
    int cap_b, cap_w;
    int handicap;
    Board() : ko(-1), to_move(BLACK), passes(0), hash(ZT),
        cap_b(0), cap_w(0), handicap(0) {
        std::memset(cells, EMPTY, sizeof(cells));
    }
    void reset() {
        std::memset(cells, EMPTY, sizeof(cells));
        ko = -1; to_move = BLACK; passes = 0;
        hash = ZT; cap_b = cap_w = 0; handicap = 0;
    }
    void set_handicap(int h) {
        reset();
        handicap = h;
        if (h >= 2 && h <= 9) {
            for (int i = 0; i < h; ++i) {
                int p = HANDICAP_POINTS[i];
                cells[p] = BLACK;
                hash ^= ZB[p];
            }
            to_move = WHITE;   // after handicap, White plays first
            // The convention everywhere else is "ZT present <=> black to move"
            // (see reset() and play_move()). Without this the hash claims black
            // is to move, and ensure_root() compares hashes to decide whether it
            // can reuse the tree, so a handicap position could be mistaken for a
            // black-to-move position with the same stones.
            hash ^= ZT;
        }
    }
};

static thread_local int g_buf[NN], l_buf[NN];
static thread_local bool g_vis[NN], l_vis[NN];

int find_group(const Board& b, int idx, int* gsize, int* lcount) {
    int8_t color = b.cells[idx];
    std::memset(g_vis, 0, sizeof(g_vis));
    std::memset(l_vis, 0, sizeof(l_vis));
    int stack[NN], sp = 0, gs = 0, lc = 0;
    stack[sp++] = idx;
    g_vis[idx] = true;
    while (sp > 0) {
        int cur = stack[--sp];
        g_buf[gs++] = cur;
        for (int d = 0; d < NCOUNT[cur]; ++d) {
            int nb = NEIGHBORS[cur][d];
            if (b.cells[nb] == EMPTY) {
                if (!l_vis[nb]) { l_vis[nb] = true; l_buf[lc++] = nb; }
            }
            else if (b.cells[nb] == color && !g_vis[nb]) {
                g_vis[nb] = true;
                stack[sp++] = nb;
            }
        }
    }
    *gsize = gs; *lcount = lc;
    return gs;
}

bool play_move(Board& b, int idx) {
    if (idx == PASS) {
        b.ko = -1; b.passes++; b.to_move = 3 - b.to_move;
        b.hash ^= ZT; return true;
    }
    if (b.cells[idx] != EMPTY || idx == b.ko) return false;
    int8_t color = b.to_move, opp = 3 - color;
    uint64_t nh = b.hash;
    nh ^= (color == BLACK) ? ZB[idx] : ZW[idx];
    b.cells[idx] = color;
    int cap_list[NN], cap_cnt = 0;
    bool cap_set[NN];
    std::memset(cap_set, 0, sizeof(cap_set));
    for (int d = 0; d < NCOUNT[idx]; ++d) {
        int nb = NEIGHBORS[idx][d];
        if (b.cells[nb] != opp) continue;
        int gs, lc;
        find_group(b, nb, &gs, &lc);
        if (lc == 0) {
            for (int i = 0; i < gs; ++i) {
                int c = g_buf[i];
                if (!cap_set[c]) { cap_set[c] = true; cap_list[cap_cnt++] = c; }
            }
        }
    }
    if (cap_cnt == 0) {
        int gs, lc;
        find_group(b, idx, &gs, &lc);
        if (lc == 0) { b.cells[idx] = EMPTY; return false; }
    }
    for (int i = 0; i < cap_cnt; ++i) {
        int c = cap_list[i];
        nh ^= (b.cells[c] == BLACK) ? ZB[c] : ZW[c];
        b.cells[c] = EMPTY;
    }
    if (color == BLACK) b.cap_b += cap_cnt; else b.cap_w += cap_cnt;
    int new_ko = -1;
    if (cap_cnt == 1) {
        int gs, lc;
        find_group(b, idx, &gs, &lc);
        if (gs == 1 && lc == 1) new_ko = cap_list[0];
    }
    b.ko = new_ko; b.to_move = opp; b.passes = 0;
    nh ^= ZT; b.hash = nh;
    return true;
}

bool is_legal(const Board& b, int idx) {
    if (idx == PASS) return true;
    if (b.cells[idx] != EMPTY || idx == b.ko) return false;
    for (int d = 0; d < NCOUNT[idx]; ++d)
        if (b.cells[NEIGHBORS[idx][d]] == EMPTY) return true;
    int8_t opp = 3 - b.to_move;
    for (int d = 0; d < NCOUNT[idx]; ++d) {
        int nb = NEIGHBORS[idx][d];
        if (b.cells[nb] == opp) {
            int gs, lc;
            find_group(b, nb, &gs, &lc);
            if (lc == 1) return true;
        }
    }
    Board tmp = b;
    tmp.cells[idx] = b.to_move;
    int gs, lc;
    find_group(tmp, idx, &gs, &lc);
    return lc > 0;
}

inline bool is_own_eye(const Board& b, int idx, int color) {
    for (int d = 0; d < NCOUNT[idx]; ++d)
        if (b.cells[NEIGHBORS[idx][d]] != color) return false;
    return true;
}

inline bool is_tiger_mouth(const Board& b, int idx) {
    int8_t opp = 3 - b.to_move;
    int cnt = 0;
    for (int d = 0; d < NCOUNT[idx]; ++d)
        if (b.cells[NEIGHBORS[idx][d]] == opp) cnt++;
    return cnt >= 3;
}

// ============================================================
// Move enumeration (complete, unlike gen_candidates)
//
// gen_candidates() prunes to the 2-ring neighbourhood of existing stones: a
// cheap move generator for random rollouts. A policy-guided search must not
// inherit that pruning, because a single omitted legal move means the search
// can never find a win by it. enumerate_legal_moves() instead visits all 361
// points, so the network policy is the only thing that concentrates the
// search. Own eyes are kept: they are legal, merely unlikely, and the priors
// already give them almost no mass.
// ============================================================
void enumerate_legal_moves(const Board& b, std::vector<int>& out) {
    out.clear();
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] != EMPTY) continue;
        if (i == b.ko) continue;
        if (!is_legal(b, i)) continue;
        out.push_back(i);
    }
    out.push_back(PASS);
}

// ============================================================
// Board symmetry (the 8 dihedral transforms)
//
// KataGo evaluates several symmetries of the same position and averages the
// results, which cancels most of the network's directional bias with no extra
// training. The transform is applied to the board before inference and the
// inverse is applied to the returned policy, so callers always work in
// ordinary board coordinates.
// ============================================================
int transform_move(int idx, int s) {
    int x = idx % N, y = idx / N, nx = x, ny = y;
    if (s & 4) nx = N - 1 - nx;
    int rot = s & 3;
    for (int r = 0; r < rot; ++r) {
        int tx = ny, ty = N - 1 - nx;
        nx = tx; ny = ty;
    }
    return ny * N + nx;
}

void transform_board(const Board& src, int s, Board& dst) {
    dst.reset();
    dst.handicap = src.handicap;
    int rot = s & 3;
    bool flip = (s & 4) != 0;
    for (int i = 0; i < NN; ++i) {
        if (src.cells[i] == EMPTY) continue;
        int nx = i % N, ny = i / N;
        if (flip) nx = N - 1 - nx;
        for (int r = 0; r < rot; ++r) {
            int tx = ny, ty = N - 1 - nx;
            nx = tx; ny = ty;
        }
        int j = ny * N + nx;
        dst.cells[j] = src.cells[i];
        dst.hash ^= (src.cells[i] == BLACK) ? ZB[j] : ZW[j];
    }
    dst.to_move = src.to_move;
    if (src.to_move == WHITE) dst.hash ^= ZT;
    dst.ko = (src.ko >= 0) ? transform_move(src.ko, s) : -1;
    dst.passes = src.passes;
    dst.cap_b = src.cap_b;
    dst.cap_w = src.cap_w;
}

// ============================================================
// Scoring
// ============================================================
// Chinese-style area scoring: stones plus surrounded empty regions, minus komi.
// Returned as a double because komi is fractional (7.5); truncating it to int
// would turn a half-point win into a tie or flip the sign of a jigo.
double area_score(const Board& b, double komi) {
    int black = 0, white = 0;
    bool vis[NN];
    std::memset(vis, 0, sizeof(vis));
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] == BLACK) black++;
        else if (b.cells[i] == WHITE) white++;
    }
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] != EMPTY || vis[i]) continue;
        int rn = 0, border = 0;
        int stack[NN], sp = 0;
        stack[sp++] = i; vis[i] = true;
        while (sp > 0) {
            int cur = stack[--sp]; rn++;
            for (int d = 0; d < NCOUNT[cur]; ++d) {
                int nb = NEIGHBORS[cur][d];
                if (b.cells[nb] == EMPTY) {
                    if (!vis[nb]) { vis[nb] = true; stack[sp++] = nb; }
                }
                else border |= b.cells[nb];
            }
        }
        if (border == BLACK) black += rn;
        else if (border == WHITE) white += rn;
    }
    return (double)(black - white) - komi;
}

int area_winner(const Board& b, double komi) {
    return area_score(b, komi) > 0 ? BLACK : WHITE;
}

// Records a position into the rolling game-history window that feeds the
// network's 8-move history planes. Entries are newest-first and the window
// keeps 7, because the current board supplies the 8th plane.
//
// Call it with the board as it is BEFORE the move is applied; the duplicate
// check makes a repeated call with an unchanged board a no-op.
void push_snapshot(std::deque<Board>& hist, const Board& b) {
    if (!hist.empty() && hist.front().hash == b.hash) return;   // duplicate
    hist.push_front(b);
    while (hist.size() > 7) hist.pop_back();
}

// ============================================================
// SGF parser
// ============================================================
bool parse_sgf_moves(const std::string& path,
    std::vector<int>& moves, int& size) {
    std::ifstream f(path.c_str());
    if (!f) return false;
    std::stringstream ss;
    ss << f.rdbuf();
    std::string data = ss.str();
    size = 19;
    size_t pos = data.find("SZ[");
    if (pos != std::string::npos) {
        try { size = std::stoi(data.substr(pos + 3, 2)); }
        catch (...) {}
    }
    size_t i = 0;
    while (i + 4 < data.size()) {
        if ((data[i] == 'B' || data[i] == 'W') && data[i + 1] == '[') {
            char c1 = data[i + 2];
            if (c1 == ']') { moves.push_back(PASS); i += 3; continue; }
            char c2 = data[i + 3];
            if (i + 4 < data.size() && data[i + 4] == ']'
                && c1 >= 'a' && c1 < 'a' + size
                && c2 >= 'a' && c2 < 'a' + size) {
                int x = c1 - 'a', y = c2 - 'a';
                moves.push_back(y * size + x);
                i += 5; continue;
            }
        }
        i++;
    }
    return true;
}

// ============================================================
// PatternDB
// ============================================================
struct PatternDB {
    std::unordered_map<uint32_t, uint32_t> counts;
    std::unordered_map<uint32_t, uint32_t> totals;
    bool enabled;
    PatternDB() : enabled(false) {}
    static const int OFF[8][2];

    uint32_t encode(const Board& b, int idx, int color) const {
        int x = idx % N, y = idx / N;
        uint32_t code = 0;
        for (int i = 0; i < 8; ++i) {
            int nx = x + OFF[i][0], ny = y + OFF[i][1];
            int v = 0;
            if (nx >= 0 && nx < N && ny >= 0 && ny < N)
                v = b.cells[ny * N + nx];
            code |= (uint32_t)v << (i * 2);
        }
        if (color == WHITE) {
            uint32_t sw = 0;
            for (int i = 0; i < 8; ++i) {
                uint32_t v = (code >> (i * 2)) & 3;
                if (v == 1) v = 2; else if (v == 2) v = 1;
                sw |= v << (i * 2);
            }
            code = sw;
        }
        return code;
    }

    void learn_move(const Board& b, int idx, int color) {
        if (idx == PASS) return;
        uint32_t code = encode(b, idx, color);
        uint32_t key = (code << 2) | color;
        counts[key]++; totals[key]++;
    }

    double query(const Board& b, int idx, int color) const {
        if (!enabled) return -1.0;
        uint32_t code = encode(b, idx, color);
        uint32_t key = (code << 2) | color;
        auto it = totals.find(key);
        if (it == totals.end() || it->second < 3) return -1.0;
        uint32_t c = 0;
        auto itc = counts.find(key);
        if (itc != counts.end()) c = itc->second;
        return (double)c / (double)it->second;
    }

    int learn_sgf(const std::string& path, int max_moves = -1);
    int learn_directory(const std::string& dir, int max_files = -1);

    void save(const std::string& path) const {
        std::ofstream f(path.c_str(), std::ios::binary);
        if (!f) return;
        uint32_t n = (uint32_t)totals.size();
        f.write((const char*)&n, 4);
        for (auto& kv : totals) {
            uint32_t k = kv.first, v = kv.second;
            uint32_t c = 0;
            auto itc = counts.find(k);
            if (itc != counts.end()) c = itc->second;
            f.write((const char*)&k, 4);
            f.write((const char*)&c, 4);
            f.write((const char*)&v, 4);
        }
    }

    bool load(const std::string& path) {
        std::ifstream f(path.c_str(), std::ios::binary);
        if (!f) return false;
        uint32_t n = 0;
        f.read((char*)&n, 4);
        if (!f) return false;
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t k = 0, c = 0, t = 0;
            f.read((char*)&k, 4); f.read((char*)&c, 4); f.read((char*)&t, 4);
            if (!f) break;
            counts[k] = c; totals[k] = t;
        }
        enabled = true;
        return true;
    }

    size_t size() const { return totals.size(); }
    uint32_t total_samples() const {
        uint32_t s = 0;
        for (auto& kv : totals) s += kv.second;
        return s;
    }
};

const int PatternDB::OFF[8][2] = {
    {-1,-1}, {0,-1}, {1,-1},
    {-1, 0},         {1, 0},
    {-1, 1}, {0, 1}, {1, 1}
};

PatternDB g_pattern;

int PatternDB::learn_sgf(const std::string& path, int max_moves) {
    std::vector<int> moves;
    int size = 19;
    if (!parse_sgf_moves(path, moves, size) || size != N) return 0;
    Board b;
    int learned = 0;
    for (size_t step = 0; step < moves.size(); ++step) {
        if (max_moves > 0 && (int)step >= max_moves) break;
        int mv = moves[step];
        if (mv == PASS) { play_move(b, PASS); continue; }
        if (mv < 0 || mv >= NN || b.cells[mv] != EMPTY) break;
        learn_move(b, mv, b.to_move);
        if (!play_move(b, mv)) break;
        learned++;
    }
    return learned;
}

static void collect_sgf_files(const std::string& dir,
    std::vector<std::string>& out, int max_files) {
    if (max_files > 0 && (int)out.size() >= max_files) return;
#ifdef _WIN32
    std::string pattern = dir + "\\*";
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        std::string name = fd.cFileName;
        if (name == "." || name == "..") continue;
        std::string full = dir + "\\" + name;
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) {
            collect_sgf_files(full, out, max_files);
            if (max_files > 0 && (int)out.size() >= max_files) { FindClose(h); return; }
        }
        else if (name.size() > 4) {
            std::string ext = name.substr(name.size() - 4);
            for (size_t k = 0; k < ext.size(); ++k)
                ext[k] = (char)std::tolower((unsigned char)ext[k]);
            if (ext == ".sgf") {
                out.push_back(full);
                if (max_files > 0 && (int)out.size() >= max_files) { FindClose(h); return; }
            }
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR* d = opendir(dir.c_str());
    if (!d) return;
    struct dirent* entry;
    while ((entry = readdir(d)) != NULL) {
        std::string name = entry->d_name;
        if (name == "." || name == "..") continue;
        std::string full = dir + "/" + name;
        struct stat st;
        if (stat(full.c_str(), &st) != 0) continue;
        if (S_ISDIR(st.st_mode)) {
            collect_sgf_files(full, out, max_files);
            if (max_files > 0 && (int)out.size() >= max_files) { closedir(d); return; }
        }
        else if (S_ISREG(st.st_mode) && name.size() > 4) {
            std::string ext = name.substr(name.size() - 4);
            for (size_t k = 0; k < ext.size(); ++k)
                ext[k] = (char)std::tolower((unsigned char)ext[k]);
            if (ext == ".sgf") {
                out.push_back(full);
                if (max_files > 0 && (int)out.size() >= max_files) { closedir(d); return; }
            }
        }
    }
    closedir(d);
#endif
}

int PatternDB::learn_directory(const std::string& dir, int max_files) {
    std::vector<std::string> files;
    collect_sgf_files(dir, files, max_files);
    if (files.empty()) {
        std::cerr << "No SGF files in: " << dir << std::endl;
        return 0;
    }
    std::cerr << "Found " << files.size() << " SGF files" << std::endl;
    int total = 0, done = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        total += learn_sgf(files[i], -1);
        done++;
        if (done % 500 == 0)
            std::cerr << "  learned " << done << "/" << files.size()
            << "  moves=" << total << std::endl;
    }
    enabled = true;
    return total;
}

// ============================================================
// Neural Network
// ============================================================
struct NNTensor {
    std::vector<int> shape;
    std::vector<float> data;
};

// ============================================================
// Weight layout conversion
//
// The trainer (train_alphazero.py) and this engine index the board
// differently: the trainer lays a plane out as [x][y] with move label
// x*19+y, while this engine uses [y][x] (see init_tables). A 19x19 plane is
// square, so such a file loads without any complaint and the network quietly
// evaluates a transposed board and returns a transposed policy. That is the
// worst kind of bug: no error, just bad play.
//
// The fix is a coordinate change:
//   * every 3x3 kernel is transposed in its two spatial axes (k = K^T, which
//     follows from the engine reading tap p = dy*3+dx as k[dy][dx] while
//     PyTorch indexes K[kh][kw]);
//   * the flattened axes of p_fc and v_fc1, and the rows of p_fc, are
//     permuted by tau, where tau[k] = (k%19)*19 + k//19 swaps the two
//     base-19 digits and is its own inverse.
//
// This is done here, at load time, rather than by an external script: it
// removes the need for a working Python environment and makes a raw training
// .bin directly usable. A file produced by 转换权重.py carries
// "ai_engine.weights_converted" and is loaded as-is.
// ============================================================
static inline int tau_index(int k) { return (k % N) * N + (k / N); }

// Splits n elements into blocks of 361 and swaps the two 19 digits inside each
// block, writing whole 19x19 tiles.
static void permute_digit_swap(const std::vector<float>& src,
    std::vector<float>& dst) {
    size_t n = src.size();
    dst.assign(n, 0.0f);
    size_t side = (size_t)NN;
    for (size_t base = 0; base + side <= n; base += side) {
        for (int i = 0; i < NN; ++i) dst[base + tau_index(i)] = src[base + i];
    }
    for (size_t i = n / side * side; i < n; ++i) dst[i] = src[i];
}

// Transposes the last two axes of a 4-D NCHW-shaped tensor.
static void transpose_kernel_2d(const NNTensor& src, NNTensor& dst) {
    dst.shape = src.shape;
    dst.data = src.data;
    if (src.shape.size() != 4) return;
    int a = src.shape[0], b = src.shape[1], h = src.shape[2], w = src.shape[3];
    if (h <= 0 || w <= 0) return;
    dst.data.assign(src.data.size(), 0.0f);
    for (int o = 0; o < a; ++o) {
        for (int i = 0; i < b; ++i) {
            for (int y = 0; y < h; ++y) {
                for (int x = 0; x < w; ++x) {
                    dst.data[(((size_t)o * b + i) * w + y) * h + x] =
                        src.data[(((size_t)o * b + i) * h + y) * w + x];
                }
            }
        }
    }
}

// Converts a tensor that came straight from the trainer into this engine's
// layout. Names not affected by the coordinate change are copied unchanged.
static void convert_tensor_layout(const std::string& name, const NNTensor& in,
    NNTensor& out) {
    // "conv_in.weight", "blocks.N.conv1.weight", "blocks.N.conv2.weight" are
    // the only 3x3 kernels; everything else is 1x1, a per-channel vector, or a
    // fully connected matrix.
    bool is_3x3 = (name == "conv_in.weight");
    if (!is_3x3 && name.compare(0, 7, "blocks.") == 0 && name.size() >= 13) {
        std::string tail = name.substr(name.size() - 13);
        if (tail == "conv1.weight" || tail == "conv2.weight") is_3x3 = true;
    }
    if (is_3x3) {
        transpose_kernel_2d(in, out);
        return;
    }
    if (name == "p_fc.bias") {
        out.shape = in.shape;
        permute_digit_swap(in.data, out.data);
        return;
    }
    if (name == "p_fc.weight") {
        out.shape = in.shape;
        if (in.shape.size() != 2) { out.data = in.data; return; }
        int rows = in.shape[0], cols = in.shape[1];
        std::vector<float> cols_permuted;
        permute_digit_swap(in.data, cols_permuted);          // columns first
        out.data.assign(in.data.size(), 0.0f);
        std::vector<float> rowbuf(cols);
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c)
                rowbuf[c] = cols_permuted[(size_t)r * cols + c];
            int tr = (r < NN) ? tau_index(r) : r;             // rows second
            for (int c = 0; c < cols; ++c)
                out.data[(size_t)tr * cols + c] = rowbuf[c];
        }
        return;
    }
    if (name == "v_fc1.weight") {
        out.shape = in.shape;
        if (in.shape.size() != 2) { out.data = in.data; return; }
        // Only the input (column) axis is spatial: v_fc1 reads the 361-dim
        // pooled feature plane. Its 64 outputs have no spatial meaning.
        permute_digit_swap(in.data, out.data);
        return;
    }
    out.shape = in.shape;
    out.data = in.data;
}

struct NeuralNet {
    int ch = 0, n_blocks = 0, in_ch = 0;
    NNTensor conv_in_W;
    NNTensor bn_in_w, bn_in_b, bn_in_m, bn_in_v;
    struct Block {
        NNTensor conv1_W, conv2_W;
        NNTensor bn1_w, bn1_b, bn1_m, bn1_v;
        NNTensor bn2_w, bn2_b, bn2_m, bn2_v;
    };
    std::vector<Block> blocks;
    NNTensor p_conv_W;
    NNTensor p_bn_w, p_bn_b, p_bn_m, p_bn_v;
    NNTensor p_fc_W, p_fc_b;
    NNTensor v_conv_W;
    NNTensor v_bn_w, v_bn_b, v_bn_m, v_bn_v;
    NNTensor v_fc1_W, v_fc1_b, v_fc2_W, v_fc2_b;
    bool loaded = false;
    // Set by load() from the file's own marker: tells whether the weights were
    // produced by 转换权重.py (first convolution already transposed for this
    // engine's board layout).
    bool weights_converted = false;
    // True when load() had to convert the trainer's board layout in memory
    // because the file carried no marker. Either way the network is usable and
    // correct; this only drives the console report.
    bool layout_converted_at_load = false;
    // Why the last load() failed, in plain words. load() used to return a bare
    // bool, so every failure -- missing file, wrong board layout, missing
    // tensor, size mismatch -- produced the same "No nn_weights.bin" message and
    // sent the reader hunting for a file that was usually right there.
    std::string load_error;

    static bool read_tensor(std::ifstream& f, std::string& name, NNTensor& t) {
        int nl;
        if (!f.read((char*)&nl, 4) || nl <= 0 || nl > 500) return false;
        name.resize(nl);
        if (!f.read(&name[0], nl)) return false;
        int nd;
        if (!f.read((char*)&nd, 4) || nd < 0 || nd > 8) return false;
        t.shape.resize(nd);
        if (nd > 0 && !f.read((char*)t.shape.data(), nd * 4)) return false;
        int ne;
        if (!f.read((char*)&ne, 4) || ne < 0) return false;
        t.data.resize(ne);
        if (ne > 0 && !f.read((char*)t.data.data(), ne * 4)) return false;
        return true;
    }

    bool load(const std::string& path) {
        load_error.clear();
        std::ifstream f(path.c_str(), std::ios::binary);
        if (!f) {
            load_error = "cannot open '" + path + "'";
            if (errno != 0) load_error += std::string(": ") + std::strerror(errno);
            return false;
        }
        int nt;
        if (!f.read((char*)&nt, 4) || nt <= 0) {
            load_error = "'" + path + "' is too short to hold a tensor count";
            return false;
        }
        std::unordered_map<std::string, NNTensor> W;
        for (int i = 0; i < nt; ++i) {
            std::string name;
            NNTensor t;
            if (!read_tensor(f, name, t)) {
                load_error = "'" + path + "' is malformed: tensor " + std::to_string(i + 1)
                    + " of " + std::to_string(nt) + " could not be read "
                    "(truncated file, or not the format written by the trainer)";
                return false;
            }
            W[name] = t;
        }
        if (W.find("conv_in.weight") == W.end()) {
            load_error = "'" + path + "' has no conv_in.weight";
            return false;
        }
        if (W["conv_in.weight"].shape.size() != 4) return false;
        ch = W["conv_in.weight"].shape[0];
        in_ch = W["conv_in.weight"].shape[1];
        if (ch <= 0 || in_ch <= 0) return false;
        // A file that carries no marker came straight from the trainer and has
        // the other board layout, so it is converted in memory, once, here.
        // Loading it unchanged would leave the network evaluating a transposed
        // board and returning a transposed policy: no error, just bad play.
        weights_converted =
            (W.find("ai_engine.weights_converted") != W.end());
        if (!weights_converted) {
            for (std::unordered_map<std::string, NNTensor>::iterator it = W.begin();
                it != W.end(); ++it) {
                NNTensor conv;
                convert_tensor_layout(it->first, it->second, conv);
                it->second = conv;
            }
            layout_converted_at_load = true;
        }
        n_blocks = 0;
        while (W.find("blocks." + std::to_string(n_blocks) + ".conv1.weight") != W.end())
            n_blocks++;
        if (n_blocks == 0) return false;

        // Reject a file that is merely close to the expected format instead of
        // proceeding with default-constructed (empty) tensors, which would turn
        // into zero-width convolutions or out-of-range reads at inference time
        // rather than a clean "cannot load".
        {
            const char* required[] = {
                "bn_in.weight", "bn_in.bias", "bn_in.running_mean",
                "bn_in.running_var", "p_conv.weight", "p_bn.weight",
                "p_bn.bias", "p_bn.running_mean", "p_bn.running_var",
                "p_fc.weight", "p_fc.bias", "v_conv.weight", "v_bn.weight",
                "v_bn.bias", "v_bn.running_mean", "v_bn.running_var",
                "v_fc1.weight", "v_fc1.bias", "v_fc2.weight", "v_fc2.bias"
            };
            for (size_t i = 0; i < sizeof(required) / sizeof(required[0]); ++i)
                if (W.find(required[i]) == W.end()) {
                    load_error = std::string("'") + path + "' is missing required tensor '"
                        + required[i] + "'";
                    return false;
                }
            if (W["p_fc.weight"].shape.size() != 2) {
                load_error = "'" + path + "' p_fc.weight is not 2-D";
                return false;
            }
            if (W["p_fc.weight"].shape[1] != 2 * NN) {
                load_error = "'" + path + "' p_fc.weight has " +
                    std::to_string(W["p_fc.weight"].shape[1]) + " input columns, expected " +
                    std::to_string(2 * NN) + " (2*361). A file written before the trainer "
                    "switched to the 17-plane input will not fit.";
                return false;
            }
            if (W["p_fc.weight"].shape[0] != NN + 1) {
                load_error = "'" + path + "' p_fc.weight has " +
                    std::to_string(W["p_fc.weight"].shape[0]) + " policy outputs, expected " +
                    std::to_string(NN + 1) + " (361 board points + pass). Convert a 361-row "
                    "file with the weight converter first.";
                return false;
            }
            if (W["p_fc.bias"].data.size() != (size_t)(NN + 1)) {
                load_error = "'" + path + "' p_fc.bias has " +
                    std::to_string(W["p_fc.bias"].data.size()) + " entries, expected " +
                    std::to_string(NN + 1);
                return false;
            }
        }

        conv_in_W = W["conv_in.weight"];
        bn_in_w = W["bn_in.weight"];
        bn_in_b = W["bn_in.bias"];
        bn_in_m = W["bn_in.running_mean"];
        bn_in_v = W["bn_in.running_var"];

        blocks.resize(n_blocks);
        for (int i = 0; i < n_blocks; ++i) {
            std::string p = "blocks." + std::to_string(i) + ".";
            blocks[i].conv1_W = W[p + "conv1.weight"];
            blocks[i].conv2_W = W[p + "conv2.weight"];
            blocks[i].bn1_w = W[p + "bn1.weight"];
            blocks[i].bn1_b = W[p + "bn1.bias"];
            blocks[i].bn1_m = W[p + "bn1.running_mean"];
            blocks[i].bn1_v = W[p + "bn1.running_var"];
            blocks[i].bn2_w = W[p + "bn2.weight"];
            blocks[i].bn2_b = W[p + "bn2.bias"];
            blocks[i].bn2_m = W[p + "bn2.running_mean"];
            blocks[i].bn2_v = W[p + "bn2.running_var"];
        }
        p_conv_W = W["p_conv.weight"];
        p_bn_w = W["p_bn.weight"];
        p_bn_b = W["p_bn.bias"];
        p_bn_m = W["p_bn.running_mean"];
        p_bn_v = W["p_bn.running_var"];
        p_fc_W = W["p_fc.weight"];
        p_fc_b = W["p_fc.bias"];
        v_conv_W = W["v_conv.weight"];
        v_bn_w = W["v_bn.weight"];
        v_bn_b = W["v_bn.bias"];
        v_bn_m = W["v_bn.running_mean"];
        v_bn_v = W["v_bn.running_var"];
        v_fc1_W = W["v_fc1.weight"];
        v_fc1_b = W["v_fc1.bias"];
        v_fc2_W = W["v_fc2.weight"];
        v_fc2_b = W["v_fc2.bias"];

        loaded = true;
        return true;
    }

    // conv3x3b - zero-padded direct convolution, blocked for SIMD.
    //
    // The old version tested 9 bounds conditions for every (input channel,
    // output pixel) pair. Here each input channel is first copied into a
    // 21x21 zero-padded plane, so all nine taps become plain contiguous
    // reads from three row pointers and the innermost loop over x is
    // branch free and auto vectorisable. The patch is kept implicit
    // instead of materialising an im2col matrix: at 19x19 an im2col
    // buffer is ic*9*361 floats (0.8 MB at ic=64), ~600x the size of the
    // feature map it would replace, so building it costs more memory
    // traffic than the direct form saves.
    static void conv3x3(const std::vector<float>& x, int ic, int oc,
        const std::vector<float>& W, std::vector<float>& out) {
        const int HW = N * N;        // 361 real positions per channel
        const int WP = N + 2;        // 21: one zero ring around the board
        const int HWP = WP * WP;     // 441 padded positions per channel

        out.assign((size_t)oc * HW, 0.0f);
        if (ic <= 0 || oc <= 0) return;

        // Padded copy: [ic][21][21]. One scratch plane per thread, reused
        // by every convolution of the forward pass.
        static thread_local std::vector<float> pad;
        pad.resize((size_t)ic * HWP);
        for (int i = 0; i < ic; ++i) {
            float* dst = pad.data() + (size_t)i * HWP;
            std::memset(dst, 0, sizeof(float) * HWP);
            const float* src = x.data() + (size_t)i * HW;
            for (int y = 0; y < N; ++y)
                std::memcpy(dst + (y + 1) * WP + 1, src + y * N,
                    sizeof(float) * N);
        }

        for (int o = 0; o < oc; ++o) {
            float* op = out.data() + (size_t)o * HW;
            for (int i = 0; i < ic; ++i) {
                const float* p = pad.data() + (size_t)i * HWP;
                const float* k = W.data() + ((size_t)o * ic + i) * 9;
                const float k0 = k[0], k1 = k[1], k2 = k[2];
                const float k3 = k[3], k4 = k[4], k5 = k[5];
                const float k6 = k[6], k7 = k[7], k8 = k[8];
                for (int y = 0; y < N; ++y) {
                    const float* r0 = p + y * WP;   // padded row y-1
                    const float* r1 = r0 + WP;      // padded row y
                    const float* r2 = r1 + WP;      // padded row y+1
                    float* __restrict a = op + y * N;
                    for (int xx = 0; xx < N; ++xx) {
                        a[xx] += k0 * r0[xx] + k1 * r0[xx + 1] + k2 * r0[xx + 2]
                            + k3 * r1[xx] + k4 * r1[xx + 1] + k5 * r1[xx + 2]
                            + k6 * r2[xx] + k7 * r2[xx + 1] + k8 * r2[xx + 2];
                    }
                }
            }
        }
    }

    static void conv1x1(const std::vector<float>& x, int ic, int oc,
        const std::vector<float>& W, std::vector<float>& out) {
        int HW = N * N;
        out.assign((size_t)oc * HW, 0.0f);
        for (int o = 0; o < oc; ++o) {
            float* op = &out[(size_t)o * HW];
            for (int i = 0; i < ic; ++i) {
                float w = W[o * ic + i];
                const float* xp = &x[(size_t)i * HW];
                for (int j = 0; j < HW; ++j) op[j] += w * xp[j];
            }
        }
    }

    static void bn_relu(const std::vector<float>& x, int ch_,
        const NNTensor& w, const NNTensor& b,
        const NNTensor& m, const NNTensor& v,
        std::vector<float>& out, bool relu = true) {
        int HW = N * N;
        out.resize((size_t)ch_ * HW);
        bool has_bn = (int)w.data.size() == ch_ && (int)v.data.size() == ch_ &&
            (int)b.data.size() == ch_ && (int)m.data.size() == ch_;
        for (int c = 0; c < ch_; ++c) {
            float sc = 1.0f, bi = 0.0f;
            if (has_bn) {
                sc = w.data[c] / std::sqrt(v.data[c] + 1e-5f);
                bi = b.data[c] - m.data[c] * sc;
            }
            const float* xp = &x[(size_t)c * HW];
            float* op = &out[(size_t)c * HW];
            for (int j = 0; j < HW; ++j) {
                float val = xp[j] * sc + bi;
                op[j] = (relu && val < 0) ? 0.0f : val;
            }
        }
    }

    static void fc(const std::vector<float>& x, const NNTensor& W,
        const NNTensor& b, std::vector<float>& out, bool relu) {
        int oc = W.shape[0], ic = W.shape[1];
        out.resize(oc);
        for (int o = 0; o < oc; ++o) {
            float s = b.data[o];
            const float* wp = &W.data[(size_t)o * ic];
            for (int i = 0; i < ic; ++i) s += wp[i] * x[i];
            out[o] = (relu && s < 0) ? 0.0f : s;
        }
    }

    // Builds the input planes exactly the way the training code does
    // (train_alphazero.py, GoBoard.encode):
    //
    //   plane 0..1    : the CURRENT position (black, white)
    //   plane 2..15   : the previous 7 positions, newest first, two planes each
    //   plane in_ch-1 : 1.0 everywhere when black is to move
    //
    // The trainer snapshots the board *after* every move, and encodes the
    // position BEFORE playing the next move, so its plane 0 is the position
    // being evaluated, not the one before it. `history` therefore holds only
    // the predecessors.
    void extract_input(const Board& b,
        const std::vector<const Board*>& history,
        std::vector<float>& input) const {
        int HW = N * N;
        input.assign((size_t)in_ch * HW, 0.0f);
        int hist_planes = 2 * 8;
        int max_planes = in_ch - 1;
        if (hist_planes > max_planes) hist_planes = max_planes;
        // Plane 0/1 is the current position. This is what the network was
        // trained to read first; leaving it out (and putting the previous
        // position there instead) fed the network a position one move stale.
        if (hist_planes >= 2) {
            float* bp = input.data();
            float* wp = bp + HW;
            for (int i = 0; i < HW; ++i) {
                if (b.cells[i] == BLACK) bp[i] = 1.0f;
                else if (b.cells[i] == WHITE) wp[i] = 1.0f;
            }
        }
        int nhist = (int)history.size();
        if (nhist > 7) nhist = 7;
        for (int h = 0; h < nhist; ++h) {
            int p = 2 * (h + 1);              // planes 2,4,6,... for predecessors
            if (p + 1 >= hist_planes) break;
            const Board* hb = history[h];
            if (!hb) continue;
            float* bp = input.data() + (size_t)p * HW;
            float* wp = bp + HW;
            for (int i = 0; i < HW; ++i) {
                if (hb->cells[i] == BLACK) bp[i] = 1.0f;
                else if (hb->cells[i] == WHITE) wp[i] = 1.0f;
            }
        }
        // The colour plane must be written even when the network has only the
        // two original planes, so that a 3-plane net still gets a signal.
        if (in_ch > 2 && b.to_move == BLACK) {
            float* cp = input.data() + (size_t)(in_ch - 1) * HW;
            for (int i = 0; i < HW; ++i) cp[i] = 1.0f;
        }
        // When there is no history at all the predecessor planes stay zero,
        // which is exactly the trainer's convention for the first eight moves
        // of a game.
    }

    bool forward(const Board& b, const std::vector<const Board*>& history,
        std::vector<float>& policy, float& value) {
        std::vector<float> input;
        extract_input(b, history, input);
        std::vector<float> a, t1, t2, t3, cur;

        conv3x3(input, in_ch, ch, conv_in_W.data, a);
        bn_relu(a, ch, bn_in_w, bn_in_b, bn_in_m, bn_in_v, cur);

        for (int i = 0; i < n_blocks; ++i) {
            conv3x3(cur, ch, ch, blocks[i].conv1_W.data, a);
            bn_relu(a, ch, blocks[i].bn1_w, blocks[i].bn1_b,
                blocks[i].bn1_m, blocks[i].bn1_v, t1);
            conv3x3(t1, ch, ch, blocks[i].conv2_W.data, a);
            bn_relu(a, ch, blocks[i].bn2_w, blocks[i].bn2_b,
                blocks[i].bn2_m, blocks[i].bn2_v, t2, false);
            t3.resize(cur.size());
            for (size_t j = 0; j < cur.size(); ++j) {
                float v = cur[j] + t2[j];
                t3[j] = (v > 0) ? v : 0;
            }
            cur = t3;
        }

        conv1x1(cur, ch, p_conv_W.shape[0], p_conv_W.data, a);
        bn_relu(a, p_conv_W.shape[0], p_bn_w, p_bn_b, p_bn_m, p_bn_v, t1);
        std::vector<float> logits;
        fc(t1, p_fc_W, p_fc_b, logits, false);
        if (logits.empty()) return false;
        float mx = logits[0];
        for (size_t i = 0; i < logits.size(); ++i)
            if (logits[i] > mx) mx = logits[i];
        float sum = 0;
        policy.resize(logits.size());
        for (size_t i = 0; i < logits.size(); ++i) {
            policy[i] = std::exp(logits[i] - mx);
            sum += policy[i];
        }
        if (sum <= 0.0f) return false;
        for (size_t i = 0; i < policy.size(); ++i) policy[i] /= sum;

        conv1x1(cur, ch, 1, v_conv_W.data, a);
        bn_relu(a, 1, v_bn_w, v_bn_b, v_bn_m, v_bn_v, t1);
        std::vector<float> h1;
        fc(t1, v_fc1_W, v_fc1_b, h1, true);
        std::vector<float> vout;
        fc(h1, v_fc2_W, v_fc2_b, vout, false);
        if (vout.empty()) value = 0.0f;
        else value = std::tanh(vout[0]);
        return true;
    }
};

NeuralNet g_nn;

// Counts network forward passes, so the benchmark can prove the search really
// uses the network, and that each simulation costs exactly one evaluation
// rather than the two an earlier draft of this engine was paying. Declared here
// (next to the network it counts) because nn_forward() below uses it.
static std::atomic<long long> g_nn_calls(0);

// ============================================================
// Network evaluation helpers (symmetry-averaged)
// ============================================================
// Evaluates b under each listed symmetry and averages the result. Averaging
// independent orientations is the cheap version of KataGo's test-time
// augmentation: it costs nsym forward passes but removes most of the
// network's directional bias. The policy is mapped back to ordinary board
// coordinates, so index i of `policy` always means point i and index NN
// means pass.
//
// `history` is the position history the network was trained to see (newest
// first; see NeuralNet::extract_input). It is optional: an empty vector means
// the history planes stay zero, which is exactly what the trainer produces for
// the first eight moves of a game.
static void nn_forward(const Board& b, const std::vector<int>& syms,
    std::vector<float>& policy, double& winrate,
    const std::vector<const Board*>* history = NULL) {
    policy.clear();
    winrate = 0.5;
    if (!g_nn.loaded || syms.empty()) return;

    static const std::vector<const Board*> no_history;
    const std::vector<const Board*>& hist =
        history ? *history : no_history;

    std::vector<float> acc;
    double vsum = 0.0;
    bool first = true;
    std::vector<Board> hist_transformed;
    std::vector<const Board*> hist_mapped;
    for (size_t k = 0; k < syms.size(); ++k) {
        int s = syms[k] & 7;
        Board tb;
        const Board* bp = &b;
        if (s != 0) { transform_board(b, s, tb); bp = &tb; }
        // The history must be transformed with the SAME symmetry as the
        // position, otherwise the network sees a rotated board paired with
        // unrotated history and the extra symmetry evaluations do more harm
        // than good.
        const std::vector<const Board*>* hp = &hist;
        if (s != 0 && !hist.empty()) {
            hist_transformed.assign(hist.size(), Board());
            hist_mapped.assign(hist.size(), (const Board*)NULL);
            for (size_t h = 0; h < hist.size(); ++h) {
                if (!hist[h]) continue;
                transform_board(*hist[h], s, hist_transformed[h]);
                hist_mapped[h] = &hist_transformed[h];
            }
            hp = &hist_mapped;
        }
        std::vector<float> raw, mapped;
        float v = 0.0f;
        g_nn_calls.fetch_add(1);
        g_nn.forward(*bp, *hp, raw, v);
        if (!(v == v)) v = 0.0f;                 // NaN guard
        mapped.assign((size_t)NN + 1, 0.0f);
        // A policy head that is only 361 wide has no pass output at all. The
        // pass slot then stays 0 and picks up the uniform floor in
        // nn_policy_to_targets; reading raw[NN] regardless would be an
        // out-of-bounds read on the vector.
        if (raw.size() > (size_t)NN) mapped[NN] = raw[NN];
        for (int i = 0; i < NN; ++i) {
            int j = (s == 0) ? i : transform_move(i, s);
            if (j >= 0 && (size_t)j < raw.size()) mapped[i] = raw[j];
        }
        if (first) { acc.swap(mapped); first = false; }
        else for (size_t i = 0; i < acc.size(); ++i) acc[i] += mapped[i];
        vsum += v;
    }
    float inv = 1.0f / (float)syms.size();
    for (size_t i = 0; i < acc.size(); ++i) acc[i] *= inv;
    policy.swap(acc);
    winrate = (vsum / (double)syms.size() + 1.0) * 0.5;
    if (winrate < 0.0) winrate = 0.0;
    if (winrate > 1.0) winrate = 1.0;
}

// Redistributes the raw network policy onto the legal moves of b, then
// renormalises. KataGo does the same thing: the policy head is not
// guaranteed to be zero on illegal points, and its absolute scale is
// meaningless, so only the relative ordering inside the legal set matters.
static void nn_policy_to_targets(const Board& b, const std::vector<int>& legal,
    const std::vector<float>& policy, std::vector<float>& targets) {
    targets.assign(legal.size(), 0.0f);
    double total = 0.0;
    bool have_policy = !policy.empty();
    if (have_policy) {
        for (size_t i = 0; i < legal.size(); ++i) {
            int slot = (legal[i] == PASS) ? NN : legal[i];
            float p = (slot >= 0 && slot < (int)policy.size()) ? policy[slot] : 0.0f;
            if (!(p > 0.0f)) p = 0.0f;
            targets[i] = p;
            total += p;
        }
    }
    double legal_count = (double)legal.size();
    // Uniform floor first: even a confident policy can be wrong about a move
    // the network never considered, and a zero prior makes it invisible to
    // PUCT forever. Keep most of the shaping on top of that floor.
    double floor_p = 0.02 / legal_count;
    for (size_t i = 0; i < legal.size(); ++i) {
        targets[i] = (float)((have_policy && total > 0.0)
            ? 0.98 * (targets[i] / total) + floor_p
            : 1.0 / legal_count);
    }
}

// True when `idx` looks like a one-point eye of `color`: the point is empty,
// all its neighbours are that colour, and at most one diagonal is not.
// Diagonal checking is what separates a real eye from a false one, so the
// test is worth the four extra reads.
static bool looks_like_true_eye(const Board& b, int idx, int color) {
    if (b.cells[idx] != EMPTY || idx == b.ko) return false;
    for (int d = 0; d < NCOUNT[idx]; ++d)
        if (b.cells[NEIGHBORS[idx][d]] != color) return false;
    int x = idx % N, y = idx / N, opp = 0;
    for (int dx = -1; dx <= 1; dx += 2) {
        for (int dy = -1; dy <= 1; dy += 2) {
            int nx = x + dx, ny = y + dy;
            if (nx < 0 || nx >= N || ny < 0 || ny >= N) continue;
            if (b.cells[ny * N + nx] != color) opp++;
        }
    }
    return opp <= 1;
}

// Damps the prior of own-eye moves. Filling a real eye is legal but is almost
// always a wasted move, and a uniform floor on the legal set would otherwise
// guarantee it enough mass to be explored. The factor is deliberately not
// zero: if every remaining legal move were an eye, a zero prior would leave
// the node with no playable child and stall the search.
static void damp_own_eye_priors(const Board& b, std::vector<float>& targets) {
    std::vector<int> legal;
    enumerate_legal_moves(b, legal);
    if (legal.size() != targets.size()) return;   // caller used another set
    bool changed = false;
    for (size_t i = 0; i < legal.size(); ++i) {
        if (legal[i] == PASS) continue;
        if (looks_like_true_eye(b, legal[i], b.to_move)) {
            targets[i] *= 0.25f;
            changed = true;
        }
    }
    if (!changed) return;
    double sum = 0.0;
    for (size_t i = 0; i < targets.size(); ++i) sum += targets[i];
    if (sum <= 0.0) {
        float u = 1.0f / (float)targets.size();
        for (size_t i = 0; i < targets.size(); ++i) targets[i] = u;
        return;
    }
    for (size_t i = 0; i < targets.size(); ++i)
        targets[i] = (float)(targets[i] / sum);
}

// ============================================================
// Candidates and priors
// ============================================================
void gen_candidates(const Board& b, std::vector<int>& out) {
    out.clear();
    int occupied = 0;
    for (int i = 0; i < NN; ++i) if (b.cells[i] != EMPTY) occupied++;
    if (occupied < 4) {
        for (int k = 0; k < 9; ++k)
            if (b.cells[STAR_POINTS[k]] == EMPTY) out.push_back(STAR_POINTS[k]);
        if (out.empty())
            for (int i = 0; i < NN; ++i)
                if (b.cells[i] == EMPTY) out.push_back(i);
        out.push_back(PASS);
        return;
    }
    bool nearby[NN];
    std::memset(nearby, 0, sizeof(nearby));
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] == EMPTY) continue;
        for (int d = 0; d < NCOUNT[i]; ++d) {
            int nb = NEIGHBORS[i][d];
            if (b.cells[nb] != EMPTY) continue;
            nearby[nb] = true;
            for (int d2 = 0; d2 < NCOUNT[nb]; ++d2) {
                int nb2 = NEIGHBORS[nb][d2];
                if (b.cells[nb2] == EMPTY) nearby[nb2] = true;
            }
        }
    }
    for (int i = 0; i < NN; ++i) {
        if (!nearby[i] || i == b.ko) continue;
        if (is_own_eye(b, i, b.to_move)) continue;
        out.push_back(i);
    }
    if (out.empty())
        for (int i = 0; i < NN; ++i)
            if (b.cells[i] == EMPTY && i != b.ko) out.push_back(i);
    out.push_back(PASS);
}

double move_prior(const Board& b, int idx) {
    if (idx == PASS) return 0.05;
    double s = 0.5;
    int8_t opp = 3 - b.to_move;
    for (int d = 0; d < NCOUNT[idx]; ++d) {
        int nb = NEIGHBORS[idx][d];
        if (b.cells[nb] == opp) s += 1.5;
        else if (b.cells[nb] == b.to_move) s += 0.3;
    }
    int min_dist = 99;
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] == EMPTY) continue;
        int dx = (idx % N) - (i % N);
        int dy = (idx / N) - (i / N);
        int dd = std::max(std::abs(dx), std::abs(dy));
        if (dd < min_dist) min_dist = dd;
    }
    if (min_dist >= 6) s += 2.0;
    else if (min_dist >= 4) s += 1.0;
    else if (min_dist <= 1) s -= 0.8;
    int x = idx % N, y = idx / N;
    int edge = std::min(std::min(x, y), std::min(N - 1 - x, N - 1 - y));
    if (edge == 3 || edge == 4) s += 1.5;
    else if (edge == 2) s += 0.3;
    else if (edge <= 1) s -= 2.0;
    if (g_pattern.enabled) {
        double p = g_pattern.query(b, idx, b.to_move);
        if (p >= 0) s += 15.0 * p;
    }
    if (is_tiger_mouth(b, idx)) s *= 0.3;
    return s;
}

// ============================================================
// Rollout
// ============================================================
static thread_local std::mt19937 rng_roll(
    (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count()
    ^ (uint32_t)std::hash<std::thread::id>()(std::this_thread::get_id()));

inline int rnd_int() { return (int)(rng_roll() & 0x7fffffff); }

int tactical_move(const Board& b) {
    int8_t opp = 3 - b.to_move;
    bool checked[NN], targets[NN];
    std::memset(checked, 0, sizeof(checked));
    std::memset(targets, 0, sizeof(targets));
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] != opp || checked[i]) continue;
        int gs, lc;
        find_group(b, i, &gs, &lc);
        for (int k = 0; k < gs; ++k) checked[g_buf[k]] = true;
        if (lc == 1) {
            for (int k = 0; k < gs; ++k) {
                int cell = g_buf[k];
                for (int d = 0; d < NCOUNT[cell]; ++d) {
                    int nb = NEIGHBORS[cell][d];
                    if (b.cells[nb] == EMPTY) targets[nb] = true;
                }
            }
        }
    }
    for (int i = 0; i < NN; ++i) {
        if (!targets[i] || i == b.ko) continue;
        if (is_legal(b, i)) return i;
    }
    return -1;
}

int atari_move(const Board& b) {
    int8_t opp = 3 - b.to_move;
    bool checked[NN], targets[NN];
    std::memset(checked, 0, sizeof(checked));
    std::memset(targets, 0, sizeof(targets));
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] != opp || checked[i]) continue;
        int gs, lc;
        find_group(b, i, &gs, &lc);
        for (int k = 0; k < gs; ++k) checked[g_buf[k]] = true;
        if (lc == 2) {
            for (int k = 0; k < gs; ++k) {
                int cell = g_buf[k];
                for (int d = 0; d < NCOUNT[cell]; ++d) {
                    int nb = NEIGHBORS[cell][d];
                    if (b.cells[nb] == EMPTY) targets[nb] = true;
                }
            }
        }
    }
    int found[NN], fn = 0;
    for (int i = 0; i < NN; ++i) {
        if (!targets[i] || i == b.ko) continue;
        if (is_tiger_mouth(b, i)) continue;
        if (is_legal(b, i)) found[fn++] = i;
    }
    if (fn == 0) return -1;
    return found[rnd_int() % fn];
}

int local_random_move(const Board& b) {
    if ((rnd_int() % 100) < 85) {
        bool nearby[NN];
        std::memset(nearby, 0, sizeof(nearby));
        for (int i = 0; i < NN; ++i) {
            if (b.cells[i] == EMPTY) continue;
            for (int d = 0; d < NCOUNT[i]; ++d) {
                int nb = NEIGHBORS[i][d];
                if (b.cells[nb] != EMPTY) continue;
                nearby[nb] = true;
                for (int d2 = 0; d2 < NCOUNT[nb]; ++d2) {
                    int nb2 = NEIGHBORS[nb][d2];
                    if (b.cells[nb2] == EMPTY) nearby[nb2] = true;
                }
            }
        }
        int cands[NN], cn = 0;
        for (int i = 0; i < NN; ++i)
            if (nearby[i] && i != b.ko && !is_own_eye(b, i, b.to_move))
                cands[cn++] = i;
        if (cn > 0) {
            int start = rnd_int() % cn;
            int limit = cn < 40 ? cn : 40;
            for (int k = 0; k < limit; ++k) {
                int i = cands[(start + k) % cn];
                if (is_tiger_mouth(b, i) && (rnd_int() % 100) < 80) continue;
                if (is_legal(b, i)) return i;
            }
        }
    }
    int start = rnd_int() % NN;
    for (int k = 0; k < NN; ++k) {
        int idx = (start + k) % NN;
        if (b.cells[idx] != EMPTY || idx == b.ko) continue;
        if (is_own_eye(b, idx, b.to_move)) continue;
        if (is_legal(b, idx)) return idx;
    }
    return PASS;
}

int influence_winner(const Board& b, double komi) {
    double infl[NN];
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] == BLACK) infl[i] = 1.0;
        else if (b.cells[i] == WHITE) infl[i] = -1.0;
        else infl[i] = 0.0;
    }
    for (int iter = 0; iter < 3; ++iter) {
        double ni[NN];
        for (int i = 0; i < NN; ++i) {
            if (b.cells[i] != EMPTY) { ni[i] = infl[i]; continue; }
            double s = 0;
            for (int d = 0; d < NCOUNT[i]; ++d) s += infl[NEIGHBORS[i][d]];
            ni[i] = (infl[i] + s / NCOUNT[i]) * 0.55;
        }
        std::memcpy(infl, ni, sizeof(infl));
    }
    double score = 0;
    for (int i = 0; i < NN; ++i) {
        if (b.cells[i] == BLACK) score += 1;
        else if (b.cells[i] == WHITE) score -= 1;
        else if (infl[i] > 0.15) score += (infl[i] < 1.0 ? infl[i] : 1.0);
        else if (infl[i] < -0.15) score -= ((-infl[i]) < 1.0 ? -infl[i] : 1.0);
    }
    return (score - komi > 0) ? BLACK : WHITE;
}

int rollout(Board b, double komi, int max_moves,
    uint64_t* amaf_b, uint64_t* amaf_w) {
    int moves = 0;
    int influence_after = 60;
    while (b.passes < 2 && moves < max_moves) {
        if (moves >= influence_after) return influence_winner(b, komi);
        int idx = tactical_move(b);
        if (idx == -1) idx = atari_move(b);
        if (idx == -1) idx = local_random_move(b);
        if (idx == PASS) {
            play_move(b, PASS);
        }
        else {
            uint64_t* arr = (b.to_move == BLACK) ? amaf_b : amaf_w;
            arr[idx >> 6] |= (1ULL << (idx & 63));
            play_move(b, idx);
        }
        moves++;
    }
    return area_winner(b, komi);
}

// ============================================================
// KataGo-style MCTS
//
// What changed from the previous engine and why:
//
//  * No random rollouts. Every leaf is evaluated by the network; value is a
//    winrate prediction, not the outcome of a random playout. Random playouts
//    were the dominant cost per simulation and the dominant source of noise,
//    which is why the old engine spent its budget on 220-move rollouts instead
//    of on tree depth.
//
//  * PUCT with the network policy as the prior, at every node. The old code
//    used plain UCT exploration for non-root nodes (its prior array was only
//    filled at the root) and a hard-coded +20x rescale of the network policy.
//    PUCT is scale-free because the exploration term is already multiplied by
//    the move prior, so no magic multiplier is needed and the prior actually
//    guides deep search.
//
//  * RAVE/AMAF removed. RAVE was standing in for exactly what the learned
//    policy now provides; keeping both means two competing move orderings, and
//    the old beta-blend was computed from uninitialised visit counters in the
//    first simulations. Removing it also deletes the per-simulation AMAF
//    bitmask bookkeeping.
//
//  * One shared tree instead of one private tree per thread. Independent trees
//    cannot pool their statistics, so the old engine effectively searched with
//    (visits/threads) nodes and then averaged by *vote*, which is a strictly
//    worse use of the same CPU time. This tree uses atomics for the hot
//    selection path, a short lock for expansion, and virtual loss so that
//    concurrent threads do not all descend the same branch.
//
//  * Tree reuse. The subtree of the played move is kept as the next root, so
//    thinking time accumulates across moves instead of being thrown away.
//
//  * Terminal positions are scored directly instead of being estimated.
// ============================================================

struct SearchParams {
    double max_time = 5.0;      // seconds for this move
    long long max_sims = 0;     // >0: hard simulation budget, 0 = time only
    int threads = 2;
    double cpuct = 1.10;        // PUCT exploration constant (sqrt-normalised)
    double fpu_reduction = 0.20;  // unvisited child Q = parent Q - this
    double virtual_loss = 1.0;  // discourages threads from sharing a path
    double root_noise_alpha = 0.0;  // >0 enables Dirichlet root noise
    double root_noise_eps = 0.25;
    double temperature = 0.0;   // 0 = pick the most visited move
    double policy_temp = 1.10;  // prior sharpening exponent
    int symmetries = 1;         // 1..8 test-time augmentations
    size_t max_nodes = 500000;
    bool use_nn = true;         // false: heuristic priors, uniform leaf value
    uint64_t seed = 0x9E3779B97F4A7C15ULL;

    // Positions before the search root, newest first (up to 7), owned by the
    // caller. Needed because the network's input has 8-move history planes.
    // May be NULL: the history planes are then zero, which is what the trainer
    // itself produces for the first eight moves of a game.
    const std::deque<Board>* game_history = NULL;
};

struct SearchStats {
    double winrate = 0.5;    // root player's expected winrate
    double root_value = 0.0; // same number on the [-1, 1] scale
    int visits = 0;
    int depth = 0;
    double seconds = 0.0;
    bool nn_used = false;
};

// The GUI code below is defined after the search, so it needs these.
struct SearchResult;
SearchResult mcts_search(const Board& board, double komi,
    const SearchParams& params);
#ifdef _WIN32
void on_ai_done(WPARAM wParam, LPARAM lParam);
#endif

struct TreeNode {
    Board board;
    int parent;
    int move;                   // move that led here (PASS for pass)
    int8_t color;               // player to move at this node
    std::vector<int> children;
    std::vector<int> legal;     // legal moves here (global indices, PASS last)
    std::vector<float> targets; // prior over `legal`
    double nn_v;                // network winrate from `color`'s view
    bool terminal;
    // Prior of the move that led here, copied out of the parent's `targets` at
    // creation time. The parent's array shrinks as moves get children, so the
    // prior cannot be recovered from it later.
    float child_prior;
    std::atomic<int> visits;    // real completed visits
    std::atomic<int> vloss;     // in-flight visits (virtual loss)
    std::atomic<double> value;  // SUM of backed-up outcomes, from `color`'s view
    bool expanded;

    TreeNode()
        : parent(-1), move(PASS), color(BLACK), nn_v(0.5), terminal(false),
        child_prior(0.0f),
        visits(0), vloss(0), value(0.0), expanded(false) {
        board.reset();
    }
};

struct SearchTree {
    // A vector of raw pointers, which means the container itself must never
    // reallocate while threads are indexing it. create_node() runs under the
    // expand mutex, but readers such as `t->pool[cur]->legal` do not take that
    // mutex, so a mid-search push_back that grew the vector would free the
    // array another thread is reading. reserve_pool() fixes the capacity once,
    // up front, so the data pointer never moves; `cap` is then the hard node
    // limit and `full()` is just a size comparison.
    std::vector<TreeNode*> pool;
    size_t cap = 0;
    int root = -1;
    bool trained = false;       // at least one search has run
    double root_komi = 7.5;

    std::mutex expand_mutex;
    std::atomic<bool> stop_flag;

    // Positions before the current root, newest first (up to 7 of them), owned
    // by the caller. The ancestor chain inside the tree can only supply the
    // moves the search itself saw, so after tree reuse -- every move in a real
    // game -- the history planes at the root would otherwise be all zero. The
    // caller keeps this rolling window; the tree never owns it.
    const std::deque<Board>* game_history = NULL;

    SearchTree() : stop_flag(false) {}
    ~SearchTree() { clear(); }

    void reserve_pool(size_t max_nodes) {
        cap = max_nodes;
        pool.reserve(max_nodes);
        // Make sure the container can actually hold `cap` entries. Comparing
        // against capacity() rather than the caller's number is what keeps the
        // data pointer from ever moving; the explicit guard avoids relying on
        // capacity() being >= cap, which is what reserve() promises but is not
        // worth assuming.
        if (pool.capacity() < cap) cap = pool.capacity();
    }

    void clear() {
        for (size_t i = 0; i < pool.size(); ++i) delete pool[i];
        pool.clear();
        root = -1;
        trained = false;
    }

    bool full(size_t max_nodes) const {
        // Must not exceed the reserved capacity, and note that capacity() can be
        // zero when nothing was ever reserved; `cap` is the authoritative
        // limit set by reserve_pool().
        if (pool.size() >= max_nodes) return true;
        if (cap > 0 && pool.size() >= cap) return true;
        return false;
    }

    int create_node(const Board& b, int parent, int move) {
        TreeNode* n = new TreeNode();
        n->board = b;
        n->parent = parent;
        n->move = move;
        n->color = b.to_move;
        pool.push_back(n);
        return (int)pool.size() - 1;
    }

    // Mean outcome (from the perspective of the player to move at `idx`) in
    // [-1, 1].
    double q_of(int idx) const {
        TreeNode* n = pool[idx];
        int v = n->visits.load();
        return (v > 0) ? (n->value.load() / (double)v) : 0.0;
    }

    double winrate_of(int idx) const {
        double w = (q_of(idx) + 1.0) * 0.5;
        if (w < 0.0) w = 0.0;
        if (w > 1.0) w = 1.0;
        return w;
    }
};

static SearchTree g_tree;

// Picks or samples a move from the root visit distribution.
//   tau <= ~0   -> most visited (strongest, deterministic)
//   tau > 0     -> sample proportional to visits^(1/tau)
static int sample_from_visits(const std::vector<TreeNode*>& kids, double tau,
    std::mt19937_64& rng) {
    if (kids.empty()) return PASS;
    int n = (int)kids.size();
    if (tau < 1e-3) {
        int best = 0;
        double bv = -1.0;
        for (int i = 0; i < n; ++i) {
            double v = (double)kids[i]->visits.load();
            if (v > bv) { bv = v; best = i; }
        }
        return kids[best]->move;
    }
    std::vector<double> w(n);
    bool any = false;
    for (int i = 0; i < n; ++i) {
        double v = (double)kids[i]->visits.load();
        w[i] = std::pow(v, 1.0 / tau);
        if (w[i] > 0) any = true;
    }
    if (!any) {
        std::uniform_int_distribution<int> pick(0, n - 1);
        return kids[pick(rng)]->move;
    }
    double sum = 0;
    for (int i = 0; i < n; ++i) sum += w[i];
    std::uniform_real_distribution<double> uni(0.0, sum);
    double r = uni(rng);
    for (int i = 0; i < n; ++i) {
        r -= w[i];
        if (r <= 0) return kids[i]->move;
    }
    return kids[n - 1]->move;
}

// Chooses a child index at the root, respecting temperature.
// Also performs tree reuse: the subtree of the chosen move becomes the next
// root, so later searches start from accumulated statistics instead of zero.
static int select_move_at_root(const SearchParams& p, std::mt19937_64& rng) {
    SearchTree* t = &g_tree;
    if (t->root < 0 || t->pool[t->root]->children.empty()) return PASS;
    const std::vector<int>& ch = t->pool[t->root]->children;
    std::vector<TreeNode*> kids;
    kids.reserve(ch.size());
    for (size_t i = 0; i < ch.size(); ++i) kids.push_back(t->pool[ch[i]]);
    int mv = sample_from_visits(kids, p.temperature, rng);
    for (size_t i = 0; i < ch.size(); ++i) {
        if (t->pool[ch[i]]->move == mv) {
            t->root = ch[i];
            t->pool[t->root]->parent = -1;
            break;
        }
    }
    return mv;
}

// Ensures the tree root matches `board`; reuses the saved subtree when the
// position is the same one the last search ended on.
static void ensure_root(const Board& board, double komi) {
    SearchTree* t = &g_tree;
    if (t->trained && t->root >= 0 &&
        t->pool[t->root]->board.hash == board.hash &&
        t->root_komi == komi) {
        t->pool[t->root]->parent = -1;
        return;                                  // exact reuse
    }
    t->clear();
    t->root = t->create_node(board, -1, PASS);
    t->root_komi = komi;
}

// Collects the previous positions for the input history planes, newest first.
// Two sources are combined, in this order:
//
//   1. the tree's ancestor chain, which is the exact history of the moves the
//      search itself played;
//   2. the caller's rolling game-history window, which covers the moves played
//      before this search started.
//
// The second source is what makes the planes correct in a real game: tree
// reuse promotes the played move to the root and cuts the chain, so without it
// the root of every search would report "no previous positions" and the 16
// history planes would be zero -- an input the trainer only ever produces for
// the first eight moves of a game.
//
// Entries may be left zero when neither source reaches far enough, which is
// again exactly what the trainer does at the start of a game.
static void collect_history(const SearchTree& t, int idx,
    std::vector<const Board*>& out, int want = 8) {
    out.clear();
    if (idx < 0 || idx >= (int)t.pool.size()) return;
    int cur = t.pool[idx]->parent;
    while (cur >= 0 && (int)out.size() < want) {
        out.push_back(&t.pool[cur]->board);
        cur = t.pool[cur]->parent;
    }
    if ((int)out.size() < want && t.game_history) {
        const std::deque<Board>& gh = *t.game_history;
        for (size_t i = 0; i < gh.size() && (int)out.size() < want; ++i) {
            // Deduplicate against what the ancestor chain already supplied.
            // The most recent entry of the caller's window IS the root's
            // parent (make_child stores the pre-move board), so appending
            // blindly would repeat it and push a real predecessor out of the
            // 8-move window -- a silently wrong history, no error.
            bool dup = false;
            for (size_t h = 0; h < out.size(); ++h) {
                if (out[h] && out[h]->hash == gh[i].hash) { dup = true; break; }
            }
            if (!dup) out.push_back(&gh[i]);
        }
    }
}

// Computes priors for every legal move of `idx` and evaluates the position.
// Network inference happens WITHOUT the lock, so other threads stay in the
// selection phase while this one waits on the GPU/CPU forward pass.
static void expand_node(int idx, const SearchParams& params) {
    SearchTree* t = &g_tree;
    TreeNode* n = t->pool[idx];
    if (n->expanded) return;

    std::vector<int> legal;
    const Board& b = n->board;
    if (b.passes >= 2) {
        n->terminal = true;
        bool black_ahead = (area_score(b, t->root_komi) > 0);
        n->nn_v = (b.to_move == BLACK) ? (black_ahead ? 1.0 : 0.0)
            : (black_ahead ? 0.0 : 1.0);
        legal.clear();
    }
    else {
        enumerate_legal_moves(b, legal);
        if (legal.size() <= 1) {
            // Only pass is legal: nothing left to search.
            n->terminal = true;
            n->nn_v = 0.5;
        }
    }

    double ptemp = (params.policy_temp > 0.05) ? params.policy_temp : 0.05;
    bool net_ok = params.use_nn && g_nn.loaded && !n->terminal;
    if (net_ok) {
        std::vector<int> syms;
        int ns = params.symmetries < 1 ? 1 : params.symmetries;
        if (ns > 8) ns = 8;
        for (int k = 0; k < ns; ++k) syms.push_back(k);
        std::vector<const Board*> history;
        collect_history(*t, idx, history);
        std::vector<float> policy;
        double wr = 0.5;
        nn_forward(b, syms, policy, wr, &history);
        n->nn_v = wr;
        nn_policy_to_targets(b, legal, policy, n->targets);
    }
    else if (!n->terminal) {
        // Heuristic fallback (no network loaded): the old hand-tuned prior,
        // renormalised over the legal set of *this* position.
        std::vector<double> raw(legal.size());
        double total = 0.0;
        for (size_t i = 0; i < legal.size(); ++i) {
            raw[i] = move_prior(b, legal[i]);
            total += raw[i];
        }
        n->targets.resize(legal.size());
        for (size_t i = 0; i < legal.size(); ++i) {
            double p = (total > 0) ? raw[i] / total : 1.0 / (double)legal.size();
            n->targets[i] = (float)p;
        }
        n->nn_v = 0.5;
    }

    if (!n->terminal) {
        damp_own_eye_priors(b, n->targets);
        // policy_temp sharpens or flattens the prior. It has to actually be
        // applied -- it used to be computed and then never used, so the option
        // was silently inert.
        if (ptemp != 1.0 && !n->targets.empty()) {
            std::vector<float> shaped(n->targets.size());
            double s = 0.0;
            for (size_t i = 0; i < n->targets.size(); ++i) {
                double p = (n->targets[i] > 0.0f) ? (double)n->targets[i] : 1e-9;
                shaped[i] = (float)std::pow(p, ptemp);
                s += shaped[i];
            }
            if (s > 0.0)
                for (size_t i = 0; i < shaped.size(); ++i)
                    n->targets[i] = (float)(shaped[i] / s);
        }
    }

    n->legal = legal;
    // Children are created lazily during selection, one per descent, because
    // materialising 362 children for every expanded node costs far more memory
    // and time than the tree can ever use.
    n->expanded = true;
}

// Picks the child with the best PUCT score.
//
//   score = -Q(child)             value from THIS node's perspective
//         + cpuct * P * sqrt(N) / (1 + n)
//         - virtual_loss * inflight / (1 + n)
//
// Unvisited children fall back to FPU (parent Q minus a reduction) instead of
// Q = 0. Treating an unvisited move as "even" is dangerously optimistic for
// the side to move and makes the search dig into lost variations; FPU is what
// KataGo and Leela use to keep the unvisited frontier pessimistic.
static void collect_child_order(int idx, const SearchParams& params,
    std::vector<int>& order) {
    SearchTree* t = &g_tree;
    TreeNode* n = t->pool[idx];
    const std::vector<int>& ch = n->children;
    order.clear();
    if (ch.empty()) return;
    double parent_q = t->q_of(idx);
    double fpu = parent_q - params.fpu_reduction;
    if (fpu < -1.0) fpu = -1.0;
    int parent_visits = n->visits.load();
    double sqrt_n = std::sqrt((double)(parent_visits > 0 ? parent_visits : 1));
    std::vector<std::pair<double, int> > scored(ch.size());
    for (size_t i = 0; i < ch.size(); ++i) {
        int ci = ch[i];
        TreeNode* c = t->pool[ci];
        int nv = c->visits.load();
        double q = (nv > 0) ? (c->value.load() / (double)nv) : fpu;
        double prior = (double)c->child_prior;
        double u = params.cpuct * prior * sqrt_n / (1.0 + (double)nv);
        double score = -q + u;
        int inflight = c->vloss.load();
        if (inflight > 0)
            score -= params.virtual_loss * (double)inflight / (1.0 + (double)nv);
        scored[i] = std::make_pair(score, ci);
    }
    std::sort(scored.begin(), scored.end(),
        [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
            return a.first > b.first;
        });
    order.resize(scored.size());
    for (size_t i = 0; i < scored.size(); ++i) order[i] = scored[i].second;
}

// Materialises the child reached by legal move `slot` under `node`.
// Returns -1 if it cannot be created.
static int make_child(int node, int slot) {
    SearchTree* t = &g_tree;
    TreeNode* n = t->pool[node];
    if (slot < 0 || slot >= (int)n->legal.size()) return -1;
    int mv = n->legal[slot];
    // Capture the prior BEFORE the erase below. `targets` shrinks together with
    // `legal`, so once every legal move has a child it becomes empty -- which
    // meant collect_child_order() read 0.0 for every child and the PUCT
    // exploration term was identically zero. The prior has to travel with the
    // child instead of being looked up through the parent's shrinking array.
    float child_prior = (slot < (int)n->targets.size()) ? n->targets[slot] : 0.0f;
    Board nb = n->board;
    if (!play_move(nb, mv)) {
        // Should be impossible for a move out of enumerate_legal_moves(); make
        // it impossible to loop on by dropping it from future consideration.
        n->legal.erase(n->legal.begin() + slot);
        if (slot < (int)n->targets.size())
            n->targets.erase(n->targets.begin() + slot);
        return -1;
    }
    int ci = t->create_node(nb, node, mv);
    t->pool[ci]->child_prior = child_prior;
    // Keep prior and legal aligned while `legal` still holds unexpanded moves.
    n->legal.erase(n->legal.begin() + slot);
    if (slot < (int)n->targets.size())
        n->targets.erase(n->targets.begin() + slot);
    n->children.push_back(ci);
    return ci;
}

// Descends from the root to a leaf, creating one child per level, then
// evaluates the leaf with the network and backs the value up. Returns the
// path of node indices that were visited (root first).
static void descend(const SearchParams& params, std::vector<int>& path) {
    SearchTree* t = &g_tree;
    path.clear();
    if (t->root < 0) return;
    int cur = t->root;
    path.push_back(cur);

    while (true) {
        TreeNode* n = t->pool[cur];
        if (n->terminal) break;

        if (!n->expanded) {
            // Expand either under the lock, or bail out if another thread is
            // already doing it or the pool is full.
            bool did = false;
            {
                std::lock_guard<std::mutex> lock(t->expand_mutex);
                if (!t->pool[cur]->expanded && !t->full(params.max_nodes)) {
                    expand_node(cur, params);
                    did = true;
                }
            }
            if (!did) {
                if (!t->pool[cur]->expanded) return;   // pool full / raced
            }
            n = t->pool[cur];
            if (n->terminal) break;
        }

        if (!n->legal.empty()) {
            // Un-expanded legal moves: take the highest-prior one. This is what
            // makes the search complete -- every legal move is probed before
            // the search starts refining by value. `legal` holds only the
            // moves that have no child yet, so the test is on it alone.
            int slot = -1;
            double bestp = -1.0;
            for (size_t i = 0; i < n->legal.size(); ++i) {
                double p = (i < n->targets.size()) ? (double)n->targets[i] : 0.0;
                if (p > bestp) { bestp = p; slot = (int)i; }
            }
            int ci = -1;
            {
                std::lock_guard<std::mutex> lock(t->expand_mutex);
                if (t->full(params.max_nodes)) return;
                if (!t->pool[cur]->legal.empty()) {
                    ci = make_child(cur, slot);
                    // Claim the new child for this simulation. The backup loop
                    // releases one in-flight visit per non-root path node, so
                    // without this every node created this way would end each
                    // backup one below zero and the `inflight > 0` test in
                    // collect_child_order would under-count live claims.
                    if (ci >= 0) t->pool[ci]->vloss.fetch_add(1);
                }
            }
            if (ci < 0) continue;      // move was illegal: try again
            // STOP here: the child just created is the leaf of this simulation.
            // Walking into it would expand and network-evaluate a whole chain of
            // nodes per simulation -- hundreds of forward passes instead of one
            // -- and would also keep the PUCT branch below unreachable until a
            // node's legal list ran dry. run_simulation() expands path.back()
            // and reads its value, which is exactly one evaluation per
            // simulation, as the benchmark's nn/sim figure checks.
            path.push_back(ci);
            break;
        }

        if (n->children.empty()) break;    // nothing playable

        std::vector<int> order;
        collect_child_order(cur, params, order);
        if (order.empty()) break;
        int pick = order[0];
        {
            // Claim the child: one in-flight visit, so sibling threads see the
            // virtual loss through collect_child_order and pick elsewhere.
            std::lock_guard<std::mutex> lock(t->expand_mutex);
            collect_child_order(cur, params, order);
            if (order.empty()) break;
            pick = order[0];
            t->pool[pick]->vloss.fetch_add(1);
        }
        path.push_back(pick);
        cur = pick;
    }
}

// One simulation: descend, evaluate, back up.
static void run_simulation(const SearchParams& params, std::mt19937_64& rng) {
    SearchTree* t = &g_tree;
    if (t->root < 0) return;
    (void)rng;

    std::vector<int> path;
    descend(params, path);
    if (path.empty()) return;
    int leaf = path.back();

    // Evaluation. expand_node() already ran the network (or scored a terminal
    // position) and stored the result in nn_v, so this must NOT evaluate again:
    // a second forward pass per simulation would halve the simulations per
    // second for no extra information.
    {
        std::lock_guard<std::mutex> lock(t->expand_mutex);
        if (!t->pool[leaf]->expanded && !t->full(params.max_nodes))
            expand_node(leaf, params);
    }
    double v = t->pool[leaf]->nn_v;
    if (!(v == v)) v = 0.5;                    // NaN guard
    if (params.use_nn && g_nn.loaded) {
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
    }
    else {
        // No network: there is no learned value, and a random playout would
        // reintroduce exactly the noise this design removes. Stay at the even
        // value so the tree is driven by visit counts and priors alone.
        // Terminal positions keep their scored value, which is exact.
        if (!t->pool[leaf]->terminal) v = 0.5;
    }

    // Backup: v is a winrate for the player to move at the leaf. Walking to the
    // root flipping at every edge turns it into each ancestor's own view.
    int8_t leaf_color = t->pool[leaf]->color;
    for (int i = (int)path.size() - 1; i >= 0; --i) {
        int ni = path[i];
        TreeNode* n = t->pool[ni];
        double w = (n->color == leaf_color) ? v : (1.0 - v);
        double desired = w * 2.0 - 1.0;        // -> [-1, 1]
        n->visits.fetch_add(1);
        // `value` is a SUM, and every reader divides it by visits to get the
        // mean (q_of, collect_child_order, the root average). Accumulate the
        // sum: writing a running mean here instead while the readers still
        // divided by visits made Q shrink like mean/visits, so the value term
        // of PUCT collapsed towards 0 as a node was visited and selection
        // degenerated into prior * sqrt(N) / (1+n).
        double old = n->value.load();
        while (!n->value.compare_exchange_weak(old, old + desired)) {
        }
        if (i > 0) n->vloss.fetch_sub(1);      // release the claim on the edge
    }
}

static void search_worker(const SearchParams& params, uint64_t seed) {
    SearchTree* t = &g_tree;
    std::mt19937_64 rng(seed);
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    long long done = 0;
    while (!t->stop_flag.load()) {
        if (params.max_sims > 0 && done >= params.max_sims) break;
        run_simulation(params, rng);
        ++done;
        if ((done & 7) == 0) {
            // Check the clock often. A simulation can cost one network forward
            // pass, and an early bug made it cost a whole chain of them, so
            // checking only every 64 simulations let the search overrun its time
            // budget by minutes -- the AI appeared to think forever.
            if (params.max_time > 0) {
                double el = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t0).count();
                if (el > params.max_time) {
                    t->stop_flag.store(true);
                    break;
                }
            }
            if (t->full(params.max_nodes)) break;
        }
    }
}

// Runs one search and reports everything the callers need. The move policy
// returned in `policy` is the root visit distribution over 362 slots (index
// NN is pass); it is what a trainer should use as the policy target.
struct RunResult {
    int move = PASS;
    SearchStats stats;
    std::vector<float> policy;
    std::vector<std::pair<int, int> > top_moves;
    RunResult() {}
};

static RunResult mcts_run(const Board& board, double komi,
    const SearchParams& params) {
    SearchTree* t = &g_tree;
    RunResult out;
    SearchParams p = params;
    if (p.threads < 1) p.threads = 1;
    if (p.use_nn && !g_nn.loaded) p.use_nn = false;

    ensure_root(board, komi);
    t->game_history = params.game_history;
    // Fix the node capacity before any worker starts: the data pointer of
    // `pool` must not move while threads index it without the mutex.
    t->reserve_pool(p.max_nodes);
    t->stop_flag.store(false);
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();

    // Root expansion plus optional Dirichlet root noise, both on the root only.
    {
        std::lock_guard<std::mutex> lock(t->expand_mutex);
        if (!t->pool[t->root]->expanded && !t->full(p.max_nodes))
            expand_node(t->root, p);
        TreeNode* rootn = t->pool[t->root];
        if (p.root_noise_alpha > 0.0 && !rootn->targets.empty()) {
            std::mt19937_64 nrng(p.seed ^ 0xD1B54A32D192ED03ULL);
            std::gamma_distribution<double> gamma(p.root_noise_alpha, 1.0);
            std::vector<double> noise(rootn->targets.size());
            double sum = 0.0;
            for (size_t i = 0; i < noise.size(); ++i) {
                noise[i] = gamma(nrng);
                sum += noise[i];
            }
            if (sum > 0) {
                double eps = p.root_noise_eps;
                for (size_t i = 0; i < noise.size(); ++i)
                    rootn->targets[i] = (float)((1.0 - eps) * rootn->targets[i]
                        + eps * (noise[i] / sum));
            }
        }
    }

    std::vector<std::thread> pool;
    for (int i = 0; i < p.threads; ++i) {
        uint64_t sd = p.seed + 0x9E3779B97F4A7C15ULL * (uint64_t)(i + 1);
        pool.push_back(std::thread(search_worker, std::cref(p), sd));
    }
    for (size_t i = 0; i < pool.size(); ++i) pool[i].join();
    t->stop_flag.store(true);

    // Statistics and the visit distribution are read *before* the root moves,
    // so they describe the position that was actually searched.
    TreeNode* old_root = t->pool[t->root];
    int total_visits = old_root->visits.load();
    double weighted = 0.0;
    long long weighted_n = 0;
    double root_value = 0.0;
    out.policy.assign((size_t)NN + 1, 0.0f);
    for (size_t i = 0; i < old_root->children.size(); ++i) {
        TreeNode* c = t->pool[old_root->children[i]];
        int v = c->visits.load();
        if (v <= 0) continue;
        int slot = (c->move == PASS) ? NN : c->move;
        if (slot >= 0 && slot < (int)out.policy.size())
            out.policy[slot] = (float)v;
        // Each node's `value` is stored from ITS OWN mover's point of view, and
        // a child of the root is the opponent of the root's mover. So the
        // child's mean must be negated to become the root's view; averaging it
        // as-is would report the OPPONENT's winrate.
        weighted += -c->value.load();
        weighted_n += v;
        out.top_moves.push_back(std::make_pair(c->move, v));
    }
    if (weighted_n > 0) {
        root_value = weighted / (double)weighted_n;
    }
    else {
        root_value = t->q_of(t->root);   // already the root's own view
    }
    if (total_visits > 0) {
        for (size_t i = 0; i < out.policy.size(); ++i)
            out.policy[i] = (float)(out.policy[i] / (double)total_visits);
    }
    std::sort(out.top_moves.begin(), out.top_moves.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.second > b.second;
        });
    if (out.top_moves.size() > 5) out.top_moves.resize(5);

    std::mt19937_64 rng(p.seed ^ 0xA24BAED4963EE407ULL);
    int mv = select_move_at_root(p, rng);      // also performs tree reuse
    t->trained = true;

    out.move = mv;
    out.stats.visits = total_visits;
    out.stats.nn_used = p.use_nn;
    out.stats.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    out.stats.winrate = (root_value + 1.0) * 0.5;
    if (out.stats.winrate < 0.0) out.stats.winrate = 0.0;
    if (out.stats.winrate > 1.0) out.stats.winrate = 1.0;
    out.stats.root_value = root_value;
    {
        // Principal variation depth, for the GUI/GTP "thinking" display.
        int depth = 0, node = t->root;
        while (node >= 0 && !t->pool[node]->children.empty() && depth < 200) {
            int best = -1, bv = -1;
            const std::vector<int>& ch = t->pool[node]->children;
            for (size_t i = 0; i < ch.size(); ++i) {
                int v = t->pool[ch[i]]->visits.load();
                if (v > bv) { bv = v; best = ch[i]; }
            }
            if (best < 0 || bv <= 0) break;
            node = best;
            depth++;
        }
        out.stats.depth = depth;
    }
    return out;
}

struct SearchResult {
    int move;                 // PASS for pass
    int visits;
    double winrate;           // root player's expected winrate
    double root_value;        // [-1, 1]
    double seconds;
    int depth;
    bool nn_used;
    std::vector<std::pair<int, int> > top_moves;   // (move, visits), best first
    std::vector<float> policy;                     // visit distribution, 362 slots
    SearchResult()
        : move(PASS), visits(0), winrate(0.5), root_value(0.0), seconds(0.0),
        depth(0), nn_used(false) {}
};

SearchResult mcts_search(const Board& board, double komi,
    const SearchParams& params) {
    RunResult r = mcts_run(board, komi, params);
    SearchResult res;
    res.move = r.move;
    res.visits = r.stats.visits;
    res.winrate = r.stats.winrate;
    res.root_value = r.stats.root_value;
    res.seconds = r.stats.seconds;
    res.depth = r.stats.depth;
    res.nn_used = r.stats.nn_used;
    res.top_moves = r.top_moves;
    res.policy = r.policy;
    return res;
}

// ============================================================
// Self test
//
// The search invariants below are the ones that silently degrade playing
// strength when they break (a policy that does not cover the legal set, a
// broken symmetry transform, a prior that does not normalise), so they are
// worth checking without needing a reference opponent.
// ============================================================
// Coord formatting lives next to the GTP loop further down; it is declared
// here so the self test can print moves in the usual board notation.
std::string coord_to_gtp(int idx);

static int run_selftest() {
    int fails = 0;
    std::printf("GoAI self test\n");

    // 1. Every symmetry must be a bijection of the board, and applying it
    //    twice with the same index must return a legal point.
    {
        int bad = 0;
        for (int s = 0; s < 8; ++s) {
            std::vector<int> seen(NN, 0);
            for (int i = 0; i < NN; ++i) {
                int j = transform_move(i, s);
                if (j < 0 || j >= NN) { bad++; continue; }
                if (seen[j]) bad++;
                seen[j] = 1;
            }
            for (int i = 0; i < NN; ++i) if (!seen[i]) bad++;
        }
        std::printf("  [%s] symmetry transforms are permutations\n", bad ? "FAIL" : "ok");
        if (bad) { fails++; std::printf("        %d bad mappings\n", bad); }
    }

    // 2. transform_board must be consistent with transform_move: a stone on i
    //    has to land on transform_move(i), and the hash must stay nonzero.
    {
        int bad = 0;
        Board b;
        std::mt19937_64 rng(12345);
        for (int k = 0; k < 60; ++k) {
            int i = (int)(rng() % NN);
            b.cells[i] = (k & 1) ? BLACK : WHITE;
        }
        for (int s = 0; s < 8; ++s) {
            Board t;
            transform_board(b, s, t);
            for (int i = 0; i < NN; ++i) {
                int j = transform_move(i, s);
                if (t.cells[j] != b.cells[i]) bad++;
            }
            if (t.hash == 0) bad++;
        }
        std::printf("  [%s] transform_board matches transform_move\n", bad ? "FAIL" : "ok");
        if (bad) { fails++; std::printf("        %d mismatches\n", bad); }
    }

    // 3. enumerate_legal_moves must return exactly the points is_legal accepts,
    //    plus PASS once.
    {
        int bad = 0, pass_seen = 0;
        Board b;
        std::mt19937_64 rng(999);
        for (int k = 0; k < 200; ++k) {
            int i = (int)(rng() % NN);
            b.cells[i] = (k & 1) ? BLACK : WHITE;
        }
        std::vector<int> legal;
        enumerate_legal_moves(b, legal);
        for (size_t i = 0; i < legal.size(); ++i) {
            if (legal[i] == PASS) { pass_seen++; continue; }
            if (!is_legal(b, legal[i])) bad++;
        }
        for (int i = 0; i < NN; ++i) {
            bool listed = false;
            for (size_t k = 0; k < legal.size(); ++k)
                if (legal[k] == i) { listed = true; break; }
            if (listed != is_legal(b, i)) bad++;
        }
        if (pass_seen != 1) bad++;
        std::printf("  [%s] legal move enumeration is exact (%d moves)\n",
            bad ? "FAIL" : "ok", (int)legal.size());
        if (bad) { fails++; std::printf("        %d discrepancies\n", bad); }
    }

    // 4. Priors: same size as the legal list, non-negative, finite, summing to
    //    1, and with PASS present. This mirrors expand_node's data path.
    {
        Board b;
        b.cells[3 * N + 3] = BLACK;
        b.cells[15 * N + 15] = WHITE;
        std::vector<int> legal;
        enumerate_legal_moves(b, legal);
        // A small synthetic history, so the history planes are exercised too.
        Board h1 = b, h2 = b;
        h1.cells[9 * N + 9] = BLACK;
        h2.cells[9 * N + 9] = WHITE;
        std::vector<const Board*> history;
        history.push_back(&h1);
        history.push_back(&h2);
        std::vector<float> policy;
        if (g_nn.loaded) {
            std::vector<int> syms;
            syms.push_back(0);
            double wr = 0.5;
            nn_forward(b, syms, policy, wr, &history);
        }
        std::vector<float> targets;
        nn_policy_to_targets(b, legal, policy, targets);

        // The engine indexes the pass prior at slot NN unconditionally, and
        // the search reads policy[361] before the sizes are compared. A
        // network whose policy head is 361 wide would make that an
        // out-of-bounds read, so it has to be a hard failure here.
        if (g_nn.loaded) {
            bool layout_ok = g_nn.weights_converted || g_nn.layout_converted_at_load;
            int bad_net = 0;
            if ((int)policy.size() != NN + 1) bad_net++;
            if (g_nn.in_ch != 17) bad_net++;
            if (!layout_ok) bad_net++;
            std::printf("  [%s] network I/O contract: policy=%d (want %d) "
                "in_ch=%d (want 17) layout=%s\n",
                bad_net ? "FAIL" : "ok", (int)policy.size(), NN + 1,
                g_nn.in_ch,
                g_nn.weights_converted ? "pre-converted"
                : (g_nn.layout_converted_at_load ? "converted-in-memory"
                    : "UNKNOWN"));
            if ((int)policy.size() != NN + 1) {
                std::printf("        the policy head is %d wide, not %d.\n",
                    (int)policy.size(), NN + 1);
            }
            if (bad_net) fails++;
        }
        damp_own_eye_priors(b, targets);
        bool pass_present = false;
        double sum = 0.0, mn = 1e30, mx = -1e30;
        for (size_t i = 0; i < legal.size(); ++i) {
            if (legal[i] == PASS) pass_present = true;
            double p = targets[i];
            if (!(p == p)) p = -1.0;                 // NaN -> fail
            if (p < mn) mn = p;
            if (p > mx) mx = p;
            sum += p;
        }
        int bad = 0;
        if (targets.size() != legal.size()) bad++;
        if (!pass_present) bad++;
        if (mn < 0.0) bad++;
        if (std::fabs(sum - 1.0) > 1e-3) bad++;
        std::printf("  [%s] priors normalise (n=%d sum=%.6f min=%.2e max=%.4f%s)\n",
            bad ? "FAIL" : "ok", (int)targets.size(), sum, mn, mx,
            g_nn.loaded ? "" : ", heuristic priors");
        if (bad) { fails++; }
    }

    // 5. A short search on the empty board must return a legal move, create a
    //    tree, and populate the visit distribution.
    {
        Board b;
        SearchParams sp;
        sp.max_time = 1.0;
        sp.threads = 2;
        sp.seed = 42;
        SearchResult r = mcts_search(b, 7.5, sp);
        int bad = 0;
        if (r.move != PASS && !is_legal(b, r.move)) bad++;
        if (r.visits <= 0) bad++;
        if (r.policy.size() != (size_t)NN + 1) bad++;
        double psum = 0.0;
        for (size_t i = 0; i < r.policy.size(); ++i) psum += r.policy[i];
        if (std::fabs(psum - 1.0) > 1e-3) bad++;
        if (r.winrate < 0.0 || r.winrate > 1.0) bad++;
        std::printf("  [%s] empty-board search: move=%s visits=%d wr=%.1f%% "
            "depth=%d %.2fs policy_sum=%.4f nn=%s\n",
            bad ? "FAIL" : "ok", coord_to_gtp(r.move).c_str(), r.visits,
            r.winrate * 100.0, r.depth, r.seconds, psum,
            r.nn_used ? "yes" : "no");
        if (bad) { fails++; }
        if (r.top_moves.size() > 1) {
            std::printf("        top:");
            for (size_t i = 0; i < r.top_moves.size(); ++i)
                std::printf(" %s/%d", coord_to_gtp(r.top_moves[i].first).c_str(),
                    r.top_moves[i].second);
            std::printf("\n");
        }
    }

    if (fails) std::printf("\nSELF TEST FAILED (%d)\n", fails);
    else std::printf("\nSELF TEST PASSED\n");
    return fails ? 1 : 0;
}

// ============================================================
// Benchmark
//
// Measures the things that decide actual playing strength and that a
// correctness test cannot see: simulations per second, tree nodes per
// simulation, how deep the search gets, and how many network evaluations each
// simulation costs. It runs on a few fixed positions at increasing time
// budgets so the scaling behaviour is visible (a healthy policy-guided MCTS
// spends its extra time on depth, not on re-evaluating the root).
// ============================================================
static void show_moves(const SearchResult& r, const Board& b) {
    std::printf("        best %s", coord_to_gtp(r.move).c_str());
    if (r.top_moves.size() > 1) {
        std::printf("   top:");
        for (size_t i = 0; i < r.top_moves.size(); ++i)
            std::printf(" %s/%d", coord_to_gtp(r.top_moves[i].first).c_str(),
                r.top_moves[i].second);
    }
    std::printf("   winrate %.1f%%\n", r.winrate * 100.0);
    (void)b;
}

static Board position_after(const std::vector<int>& moves) {
    Board b;
    for (size_t i = 0; i < moves.size(); ++i) {
        if (moves[i] < 0) play_move(b, PASS);
        else if (!play_move(b, moves[i])) break;
    }
    return b;
}

static int run_benchmark(SearchParams sp) {
    std::printf("GoAI benchmark (build %s)\n", GOAI_BUILD_TAG);
    std::printf("  network: %s", g_nn.loaded ? "loaded" : "NOT LOADED (heuristics)");
    if (g_nn.loaded) {
        std::printf("  ch=%d blocks=%d in_ch=%d layout=%s",
            g_nn.ch, g_nn.n_blocks, g_nn.in_ch,
            g_nn.weights_converted ? "pre-converted"
            : (g_nn.layout_converted_at_load ? "converted-in-memory" : "UNKNOWN"));
    }
    std::printf("\n");
    std::printf("  threads: %d   cpuct: %.2f   symmetries: %d\n",
        sp.threads, sp.cpuct, sp.symmetries);
    if (g_nn.loaded && !g_nn.weights_converted && !g_nn.layout_converted_at_load) {
        std::printf("  !! unknown weight layout: results below are not meaningful\n");
    } else if (g_nn.loaded && g_nn.layout_converted_at_load) {
        std::printf("  (weights have no conversion marker; the board layout was\n");
        std::printf("   converted in memory at load time, which is the correct\n");
        std::printf("   thing for a file written by train_alphazero.py)\n");
    }
    if (g_nn.loaded && g_nn.in_ch != 17) {
        std::printf("  !! in_ch != 17: history planes will not match training\n");
    }

    // A short opening line and a midgame position, both as board points.
    std::vector<int> opening;
    opening.push_back(3 * N + 3);    // D4
    opening.push_back(15 * N + 15);  // Q16
    opening.push_back(15 * N + 3);   // D16
    std::vector<int> midgame = opening;
    midgame.push_back(3 * N + 15);   // Q4
    midgame.push_back(9 * N + 9);    // tengen
    midgame.push_back(2 * N + 15);
    midgame.push_back(15 * N + 2);
    midgame.push_back(9 * N + 3);
    midgame.push_back(3 * N + 9);

    const char* names[2] = { "opening", "midgame" };
    std::vector<int>* lines[2] = { &opening, &midgame };
    const double budgets[4] = { 0.15, 0.5, 1.5, 3.0 };

    for (int p = 0; p < 2; ++p) {
        Board b = position_after(*lines[p]);
        std::printf("\n  --- %s (%d stones) ---\n", names[p],
            (int)lines[p]->size());
        for (int k = 0; k < 4; ++k) {
            sp.max_time = budgets[k];
            sp.seed = 0xBEEF0000ULL + (uint64_t)p * 100 + (uint64_t)k;
            long long nn_before = g_nn_calls.load();
            SearchResult r = mcts_search(b, 7.5, sp);
            long long nn_used = g_nn_calls.load() - nn_before;
            double nps = (r.seconds > 0) ? (r.visits / r.seconds) : 0.0;
            double nnps = (r.seconds > 0) ? (nn_used / r.seconds) : 0.0;
            double per_sim = (r.visits > 0) ? ((double)nn_used / r.visits) : 0.0;
            std::printf("    t=%5.2fs  visits=%7d  depth=%3d  "
                "%.0f visits/s  nn=%lld (%.2f/sim, %.0f/s)  %.2fs\n",
                budgets[k], r.visits, r.depth, nps, nn_used, per_sim, nnps,
                r.seconds);
            show_moves(r, b);
        }
    }

    std::printf("\n  interpretation\n");
    std::printf("    nn/sim should be about 1.0 * symmetries; noticeably higher\n");
    std::printf("    means positions are being evaluated twice.\n");
    std::printf("    depth should grow with the time budget, not stall.\n");
    return 0;
}

// ============================================================
// SGF export
// ============================================================
void export_sgf(const std::vector<std::pair<int8_t, int> >& history,
    int size, double komi, int handicap,
    const std::string& result, const std::string& filename) {
    std::ofstream f(filename.c_str());
    if (!f) return;
    f << "(;GM[1]FF[4]CA[UTF-8]SZ[" << size << "]KM[" << komi << "]";
    if (handicap > 0) f << "HA[" << handicap << "]";
    if (!result.empty()) f << "RE[" << result << "]";
    static const char* L = "abcdefghijklmnopqrs";
    for (size_t i = 0; i < history.size(); ++i) {
        int8_t c = history[i].first;
        int m = history[i].second;
        f << ";" << (c == BLACK ? "B" : "W") << "[";
        if (m != PASS) f << L[m % size] << L[m / size];
        f << "]";
    }
    f << ")\n";
}

// ============================================================
// Selfplay
// ============================================================
struct SelfplayStats {
    int games = 0;
    int black_wins = 0;
    int white_wins = 0;
    int total_moves = 0;
};

struct SelfplayOptions {
    int games = 1;
    std::string outdir = "selfplay_games";
    double komi = 7.5;
    int handicap = 0;
    bool random_color = false;
    bool learn_from_games = false;
    uint64_t seed = 0;
};

// Self-play with the same search the engine uses to play. Opening diversity
// comes from the temperature instead of from randomly placing the first few
// stones, so the moves played are always the engine's own choices.
void run_selfplay(const SelfplayOptions& opt, SearchParams sp) {
    const std::string& outdir = opt.outdir;
    double komi = opt.komi;
    int handicap = opt.handicap;
    int games = opt.games;
#ifdef _WIN32
    CreateDirectoryA(outdir.c_str(), NULL);
#else
    mkdir(outdir.c_str(), 0755);
#endif

    uint64_t base_seed = opt.seed ? opt.seed
        : (uint64_t)std::chrono::steady_clock::now()
        .time_since_epoch().count();

    SelfplayStats stats;
    std::chrono::steady_clock::time_point t_all = std::chrono::steady_clock::now();

    std::cerr << "Selfplay start" << std::endl;
    std::cerr << "  games=" << games << "  time/move=" << sp.max_time << "s"
        << "  threads=" << sp.threads << std::endl;
    std::cerr << "  komi=" << komi << "  handicap=" << handicap
        << "  random_color=" << (opt.random_color ? "yes" : "no")
        << "  temp=" << sp.temperature << std::endl;

    for (int g = 1; g <= games; ++g) {
        Board board;
        board.set_handicap(handicap);

        std::vector<std::pair<int8_t, int> > history;
        int move_count = 0;
        int max_moves = NN * 4;

        while (board.passes < 2 && move_count < max_moves) {
            SearchParams game_sp = sp;
            // Deterministic per-position seed. A fresh random seed on every
            // move would change the RNG stream, which does not change the
            // move but does throw away tree reuse: the subtree kept from the
            // previous move would be discarded and re-expanded from scratch,
            // paying for the same network evaluations twice.
            game_sp.seed = base_seed
                ^ ((uint64_t)g * 0x9E3779B97F4A7C15ULL)
                ^ (board.hash * 0xD1B54A32D192ED03ULL);
            SearchResult r = mcts_search(board, komi, game_sp);
            int mv = r.move;
            int8_t color = board.to_move;
            if (mv == PASS || !play_move(board, mv)) {
                history.push_back(std::make_pair(color, PASS));
                if (mv != PASS) play_move(board, PASS);
            }
            else {
                history.push_back(std::make_pair(color, mv));
            }
            move_count++;
        }

        int winner = area_winner(board, komi);
        double score = area_score(board, komi);
        if (winner == BLACK) stats.black_wins++;
        else stats.white_wins++;
        stats.games++;
        stats.total_moves += move_count;

        char result[32];
        // %.1f, not %d: komi is fractional and truncating it here would misreport
        // the margin of every game by up to a point.
        if (score > 0) std::snprintf(result, sizeof(result), "B+%.1f", score);
        else std::snprintf(result, sizeof(result), "W+%.1f", -score);

        char fname[512];
        std::snprintf(fname, sizeof(fname),
            "%s/game_%05d_%s_HA%d.sgf",
            outdir.c_str(), g, result, handicap);

        export_sgf(history, N, komi, handicap, result, fname);

        if (opt.learn_from_games && g_pattern.enabled) {
            Board tmp;
            tmp.set_handicap(handicap);
            for (size_t i = 0; i < history.size(); ++i) {
                int8_t c = history[i].first;
                int m = history[i].second;
                if (m == PASS) { play_move(tmp, PASS); continue; }
                if (c == winner) {
                    g_pattern.learn_move(tmp, m, c);
                }
                if (!play_move(tmp, m)) break;
            }
        }

        double el = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_all).count();
        std::cout << "Game " << g << "/" << games
            << "  " << (winner == BLACK ? "B" : "W")
            << "+" << std::abs(score)
            << "  moves=" << move_count
            << "  B_wins=" << stats.black_wins
            << "  W_wins=" << stats.white_wins
            << "  time=" << (int)el << "s" << std::endl;
    }

    std::cout << "\n===== Selfplay Done =====" << std::endl;
    std::cout << "Games: " << stats.games << std::endl;
    std::cout << "Black wins: " << stats.black_wins
        << "  White wins: " << stats.white_wins << std::endl;
    if (stats.games > 0) {
        double bwr = 100.0 * stats.black_wins / stats.games;
        std::cout << "Black winrate: " << bwr << "%" << std::endl;
        std::cout << "Avg moves/game: " << (stats.total_moves / stats.games)
            << std::endl;
    }

    if (opt.learn_from_games && g_pattern.enabled) {
        g_pattern.save("patterns_19.bin");
        std::cout << "Pattern DB updated and saved" << std::endl;
    }
}

// ============================================================
// GTP
// ============================================================
std::string coord_to_gtp(int idx) {
    if (idx == PASS) return "pass";
    static const char* L = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    std::string s; s += L[idx % N];
    std::ostringstream oss; oss << (N - idx / N);
    s += oss.str();
    return s;
}

int gtp_to_coord(const std::string& s) {
    if (s == "pass" || s == "PASS") return PASS;
    if (s.size() < 2) return -2;
    char c = (char)std::toupper((unsigned char)s[0]);
    static const char* L = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    int x = -1;
    for (int i = 0; i < N; ++i) if (L[i] == c) { x = i; break; }
    if (x < 0) return -2;
    int row;
    try { row = std::stoi(s.substr(1)); }
    catch (...) { return -2; }
    if (row < 1 || row > N) return -2;
    return (N - row) * N + x;
}

void run_gtp(double komi, SearchParams sp) {
    Board board; board.reset();
    // Previous positions, newest first, up to 7 (the current board is the 8th
    // history plane). Maintained here rather than inside the tree because tree
    // reuse cuts the in-tree ancestor chain at every move.
    std::deque<Board> hist;
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        std::string cmd, id;
        iss >> cmd;
        if (!cmd.empty() && std::isdigit((unsigned char)cmd[0])) { id = cmd; iss >> cmd; }
        std::string resp;
        if (cmd == "protocol_version") resp = "2";
        else if (cmd == "name") resp = "GoAI++ (KataGo-style MCTS)";
        else if (cmd == "version") resp = "3.0";
        else if (cmd == "list_commands")
            resp = "protocol_version\nname\nversion\nlist_commands\n"
            "boardsize\nclear_board\nkomi\nplay\ngenmove\ntime_settings\ntime_left\nquit";
        else if (cmd == "boardsize") { board.reset(); g_tree.clear(); hist.clear(); }
        else if (cmd == "clear_board") { board.reset(); g_tree.clear(); hist.clear(); }
        else if (cmd == "komi") { double k; iss >> k; komi = k; }
        else if (cmd == "time_settings") {
            // time_settings <main> <byoyomi> <stones>: no clock model yet, so
            // treat the main time as the per-move budget floor.
            double main_t = 0, byo = 0; int stones = 0;
            if (iss >> main_t >> byo >> stones) {
                double per = (stones > 0) ? (main_t / stones + byo) : main_t;
                if (per > 0.05 && per < 600.0) sp.max_time = per;
            }
        }
        else if (cmd == "time_left") { /* accepted, unused */ }
        else if (cmd == "play") {
            std::string color, coord; iss >> color >> coord;
            int idx = gtp_to_coord(coord);
            int want = BLACK;
            if (!color.empty() && (color[0] == 'w' || color[0] == 'W')) want = WHITE;
            if (idx < -1) resp = "invalid coordinate";
            else if (board.to_move != want) {
                // The colour is part of the command; silently playing for
                // whoever happens to be to move would put the wrong stone down
                // and hand the client a corrupt game.
                resp = "illegal move";
            }
            else if (idx == PASS) {
                push_snapshot(hist, board);
                play_move(board, PASS);
            }
            else {
                // Record the position BEFORE the move: that is what the network
                // needs as "the previous position" for the next search.
                push_snapshot(hist, board);
                if (!play_move(board, idx)) resp = "illegal move";
            }
        }
        else if (cmd == "genmove") {
            std::string color; iss >> color;
            SearchParams gp = sp;
            gp.game_history = &hist;
            gp.seed = (uint64_t)std::chrono::steady_clock::now()
                .time_since_epoch().count();
            SearchResult r = mcts_search(board, komi, gp);
            if (r.move == PASS) { push_snapshot(hist, board); play_move(board, PASS); }
            else {
                push_snapshot(hist, board);
                if (!play_move(board, r.move)) play_move(board, PASS);
            }
            resp = coord_to_gtp(r.move);
            // GTP comments start with a space and are conventionally ignored by
            // clients, but they make the engine's own assessment visible.
            char info[192];
            std::snprintf(info, sizeof(info),
                "  (visits=%d winrate=%.1f%% depth=%d %.2fs%s)",
                r.visits, r.winrate * 100.0, r.depth, r.seconds,
                r.nn_used ? "" : " no-nn");
            resp += info;
        }
        else if (cmd == "quit") { std::cout << "= \n\n" << std::flush; break; }
        else resp = "unknown command";
        if (!id.empty()) std::cout << "=" << id << " ";
        else std::cout << "= ";
        std::cout << resp << "\n\n" << std::flush;
    }
}

// ============================================================
// Win32 GUI
// ============================================================
#ifdef _WIN32
#define WM_USER_AI_DONE (WM_USER + 1)

struct GuiState {
    HWND hwnd;
    Board board;
    int cell, margin, last_move, hover_idx;
    bool human_black, ai_thinking, game_over;
    int ai_move;
    SearchParams sp;
    double komi;
    int handicap;               // remembered so New Game restores it
    int cap_b, cap_w;
    Gdiplus::Image* bg_image;
    bool draw_grid;
    // Last search's own assessment, shown in the title bar.
    double last_winrate;
    int last_visits;
    // Previous positions, newest first, up to 7. Combined with the current
    // board this gives the network its 8-move history planes. Without it, tree
    // reuse cuts the in-tree ancestor chain and every search root would report
    // "no history" -- an input the trainer only produces for the first eight
    // moves of a game.
    std::deque<Board> history;
    GuiState() : hwnd(NULL), cell(34), margin(42), last_move(-1),
        hover_idx(-1), human_black(true), ai_thinking(false),
        game_over(false), ai_move(PASS), sp(), komi(7.5), handicap(0),
        cap_b(0), cap_w(0),
        bg_image(NULL), draw_grid(true),
        last_winrate(0.5), last_visits(0) {
    }
};
GuiState g_gui;

int pixel_to_board(int px, int py) {
    int col = (px - g_gui.margin + g_gui.cell / 2) / g_gui.cell;
    int row = (py - g_gui.margin + g_gui.cell / 2) / g_gui.cell;
    if (col < 0 || col >= N || row < 0 || row >= N) return -1;
    int cx = g_gui.margin + col * g_gui.cell;
    int cy = g_gui.margin + row * g_gui.cell;
    int dx = px - cx, dy = py - cy;
    if (dx * dx + dy * dy > (g_gui.cell * g_gui.cell) / 4) return -1;
    return row * N + col;
}

void board_to_pixel(int idx, int* px, int* py) {
    *px = g_gui.margin + (idx % N) * g_gui.cell;
    *py = g_gui.margin + (idx / N) * g_gui.cell;
}

void draw_stone(HDC hdc, int cx, int cy, int r, int color) {
    HBRUSH brush = CreateSolidBrush(color == BLACK ? RGB(20, 20, 20) : RGB(248, 248, 248));
    HPEN pen = CreatePen(PS_SOLID, 1, color == BLACK ? RGB(0, 0, 0) : RGB(120, 120, 120));
    HGDIOBJ ob = SelectObject(hdc, brush);
    HGDIOBJ op = SelectObject(hdc, pen);
    Ellipse(hdc, cx - r, cy - r, cx + r, cy + r);
    SelectObject(hdc, ob); SelectObject(hdc, op);
    DeleteObject(brush); DeleteObject(pen);
    int hr = r / 3, hx = cx - r / 3, hy = cy - r / 3;
    HBRUSH hb = CreateSolidBrush(color == BLACK ? RGB(90, 90, 90) : RGB(255, 255, 255));
    ob = SelectObject(hdc, hb);
    op = SelectObject(hdc, GetStockObject(NULL_PEN));
    Ellipse(hdc, hx - hr, hy - hr, hx + hr, hy + hr);
    SelectObject(hdc, ob); SelectObject(hdc, op);
    DeleteObject(hb);
}

void draw_board(HDC hdc) {
    RECT rc;
    GetClientRect(g_gui.hwnd, &rc);
    int winW = rc.right - rc.left, winH = rc.bottom - rc.top;
    if (g_gui.bg_image) {
        Gdiplus::Graphics g(hdc);
        g.SetInterpolationMode(Gdiplus::InterpolationModeHighQualityBicubic);
        g.DrawImage(g_gui.bg_image, 0, 0, winW, winH);
    }
    else {
        HBRUSH wood = CreateSolidBrush(RGB(0xE3, 0xB9, 0x6B));
        FillRect(hdc, &rc, wood);
        DeleteObject(wood);
    }
    int cell = g_gui.cell, margin = g_gui.margin;
    int x1 = margin, y1 = margin;
    int x2 = margin + cell * (N - 1), y2 = margin + cell * (N - 1);
    if (g_gui.draw_grid) {
        HPEN lp = CreatePen(PS_SOLID, 1, RGB(0x4A, 0x34, 0x10));
        HGDIOBJ op = SelectObject(hdc, lp);
        for (int i = 0; i < N; ++i) {
            int p = margin + i * cell;
            MoveToEx(hdc, x1, p, NULL); LineTo(hdc, x2, p);
            MoveToEx(hdc, p, y1, NULL); LineTo(hdc, p, y2);
        }
        SelectObject(hdc, op); DeleteObject(lp);
        HPEN bp = CreatePen(PS_SOLID, 2, RGB(0x4A, 0x34, 0x10));
        op = SelectObject(hdc, bp);
        MoveToEx(hdc, x1, y1, NULL); LineTo(hdc, x2, y1);
        LineTo(hdc, x2, y2); LineTo(hdc, x1, y2); LineTo(hdc, x1, y1);
        SelectObject(hdc, op); DeleteObject(bp);
    }
    int r = 3;
    HBRUSH sb = CreateSolidBrush(RGB(0x3A, 0x2A, 0x10));
    HGDIOBJ ob = SelectObject(hdc, sb);
    for (int k = 0; k < 9; ++k) {
        int px, py;
        board_to_pixel(STAR_POINTS[k], &px, &py);
        Ellipse(hdc, px - r, py - r, px + r, py + r);
    }
    SelectObject(hdc, ob); DeleteObject(sb);
    int stone_r = (int)(cell * 0.46);
    for (int i = 0; i < NN; ++i) {
        int v = g_gui.board.cells[i];
        if (v == EMPTY) continue;
        int px, py;
        board_to_pixel(i, &px, &py);
        draw_stone(hdc, px, py, stone_r, v);
    }
    if (g_gui.last_move >= 0 && g_gui.board.cells[g_gui.last_move] != EMPTY) {
        int px, py;
        board_to_pixel(g_gui.last_move, &px, &py);
        int m = stone_r / 3;
        HPEN mp = CreatePen(PS_SOLID, 2, RGB(255, 60, 60));
        HGDIOBJ op = SelectObject(hdc, mp);
        HGDIOBJ ob2 = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Ellipse(hdc, px - m, py - m, px + m, py + m);
        SelectObject(hdc, op); SelectObject(hdc, ob2);
        DeleteObject(mp);
    }
    if (g_gui.hover_idx >= 0 &&
        g_gui.board.cells[g_gui.hover_idx] == EMPTY &&
        !g_gui.ai_thinking && !g_gui.game_over &&
        ((g_gui.human_black && g_gui.board.to_move == BLACK) ||
            (!g_gui.human_black && g_gui.board.to_move == WHITE))) {
        int px, py;
        board_to_pixel(g_gui.hover_idx, &px, &py);
        int m = stone_r / 2;
        HPEN hp = CreatePen(PS_SOLID, 1, RGB(100, 100, 100));
        HGDIOBJ op = SelectObject(hdc, hp);
        HGDIOBJ ob2 = SelectObject(hdc, GetStockObject(NULL_BRUSH));
        Ellipse(hdc, px - m, py - m, px + m, py + m);
        SelectObject(hdc, op); SelectObject(hdc, ob2);
        DeleteObject(hp);
    }
    SetBkMode(hdc, TRANSPARENT);
    SetTextColor(hdc, RGB(0x30, 0x20, 0x08));
    HFONT font = CreateFontA(14, 0, 0, 0, FW_BOLD, 0, 0, 0,
        DEFAULT_CHARSET, 0, 0, 0, 0, "Arial");
    HGDIOBJ of = SelectObject(hdc, font);
    const char* L = "ABCDEFGHJKLMNOPQRSTUVWXYZ";
    for (int i = 0; i < N; ++i) {
        char buf[4];
        int p = margin + i * cell;
        buf[0] = L[i]; buf[1] = 0;
        TextOutA(hdc, p - 4, margin - 22, buf, 1);
        TextOutA(hdc, p - 4, margin + cell * (N - 1) + 8, buf, 1);
        std::snprintf(buf, sizeof(buf), "%d", N - i);
        TextOutA(hdc, margin - 26, p - 7, buf, (int)std::strlen(buf));
        TextOutA(hdc, margin + cell * (N - 1) + 10, p - 7, buf, (int)std::strlen(buf));
    }
    SelectObject(hdc, of); DeleteObject(font);
}

void update_status() {
    char buf[512];
    if (g_gui.game_over) {
        double s = area_score(g_gui.board, g_gui.komi);
        std::snprintf(buf, sizeof(buf),
            "Game over. %s by %.1f.  B=%d W=%d",
            s > 0 ? "Black wins" : "White wins",
            (s > 0 ? s : -s), g_gui.cap_b, g_gui.cap_w);
    }
    else {
        const char* turn = (g_gui.board.to_move == BLACK) ? "Black" : "White";
        const char* who = ((g_gui.human_black && g_gui.board.to_move == BLACK) ||
            (!g_gui.human_black && g_gui.board.to_move == WHITE))
            ? " (You)" : " (AI)";
        std::snprintf(buf, sizeof(buf),
            "%s%s to play.  B=%d W=%d   AI winrate %.1f%% (v=%d)  %s",
            turn, who, g_gui.cap_b, g_gui.cap_w,
            g_gui.last_winrate * 100.0, g_gui.last_visits,
            g_gui.ai_thinking ? "[AI thinking...]" : "");
    }
    SetWindowTextA(g_gui.hwnd, buf);
}

void start_ai_think();

// The worker that searches for the AI move. Kept joinable (not detached) so the
// window can wait for it: the search mutates the global tree and the GUI state,
// and letting it outlive run_gui_win32() would mean it keeps running while
// static destruction tears the tree down.
std::thread g_ai_thread;

void join_ai_thread() {
    if (g_ai_thread.joinable()) {
        g_tree.stop_flag.store(true);   // ask it to finish quickly
        g_ai_thread.join();
    }
}

void refresh_board() {
    InvalidateRect(g_gui.hwnd, NULL, FALSE);
    UpdateWindow(g_gui.hwnd);
    update_status();
}

// Records the position that is about to be played from, for the network's
// history planes. Keeps the most recent seven; the current board is the eighth.
void push_history(const Board& b) {
    if (!g_gui.history.empty() && g_gui.history.front().hash == b.hash) return;
    g_gui.history.push_front(b);
    while (g_gui.history.size() > 7) g_gui.history.pop_back();
}

void do_human_move(int idx) {
    if (g_gui.ai_thinking || g_gui.game_over) return;
    int hc = g_gui.human_black ? BLACK : WHITE;
    if (g_gui.board.to_move != hc) return;
    Board trial = g_gui.board;
    if (!play_move(trial, idx)) return;
    push_history(g_gui.board);
    if (g_gui.board.to_move == BLACK) g_gui.cap_b += trial.cap_b - g_gui.board.cap_b;
    else g_gui.cap_w += trial.cap_w - g_gui.board.cap_w;
    g_gui.board = trial;
    g_gui.last_move = idx;
    refresh_board();
    if (g_gui.board.passes >= 2) { g_gui.game_over = true; refresh_board(); return; }
    start_ai_think();
}

void start_ai_think() {
    if (g_gui.game_over) return;
    int ac = g_gui.human_black ? WHITE : BLACK;
    if (g_gui.board.to_move != ac) return;
    g_gui.ai_thinking = true;
    update_status();
    HWND hwnd = g_gui.hwnd;
    Board bc = g_gui.board;
    double komi = g_gui.komi;
    SearchParams sp = g_gui.sp;     // by value: the worker owns its copy
    sp.game_history = &g_gui.history;
    sp.seed = (uint64_t)std::chrono::steady_clock::now()
        .time_since_epoch().count();
    // The worker hands the result back through the message, so shared GUI
    // state is only ever touched by the window thread. The winrate rides in
    // the message's WPARAM, which is 64-bit here. Any previous worker has
    // already finished, because ai_thinking gated entry to this function.
    join_ai_thread();
    g_ai_thread = std::thread([hwnd, bc, komi, sp]() {
        SearchResult r = mcts_search(bc, komi, sp);
        g_gui.ai_move = r.move;
        union { double d; WPARAM w; } pack;
        pack.d = r.winrate;
        PostMessageA(hwnd, WM_USER_AI_DONE, pack.w, (LPARAM)r.visits);
    });
}

void on_ai_done(WPARAM wParam, LPARAM lParam) {
    g_gui.ai_thinking = false;
    union { double d; WPARAM w; } pack;
    pack.w = wParam;
    g_gui.last_winrate = pack.d;
    g_gui.last_visits = (int)lParam;
    if (g_gui.game_over) return;
    int mv = g_gui.ai_move;
    if (mv == PASS) {
        push_history(g_gui.board);
        play_move(g_gui.board, PASS);
        g_gui.last_move = -1;
    }
    else {
        Board trial = g_gui.board;
        if (!play_move(trial, mv)) {
            push_history(g_gui.board);
            play_move(g_gui.board, PASS);
            g_gui.last_move = -1;
        }
        else {
            push_history(g_gui.board);
            if (g_gui.board.to_move == BLACK) g_gui.cap_b += trial.cap_b - g_gui.board.cap_b;
            else g_gui.cap_w += trial.cap_w - g_gui.board.cap_w;
            g_gui.board = trial;
            g_gui.last_move = mv;
        }
    }
    refresh_board();
    if (g_gui.board.passes >= 2) { g_gui.game_over = true; refresh_board(); }
}

void new_game() {
    if (g_gui.ai_thinking) return;
    // set_handicap() resets the board itself, so the handicap stones come back.
    // A plain reset() would silently drop them, because Board::reset clears the
    // handicap field.
    g_gui.board.set_handicap(g_gui.handicap);
    // Drop the accumulated search tree: the new game starts from an empty
    // board, and keeping thousands of nodes from the previous game would only
    // make the first searches waste their budget revisiting dead positions.
    g_tree.clear();
    g_gui.history.clear();
    g_gui.last_move = -1;
    g_gui.game_over = false;
    g_gui.cap_b = g_gui.cap_w = 0;
    g_gui.last_winrate = 0.5;
    g_gui.last_visits = 0;
    refresh_board();
    int hc = g_gui.human_black ? BLACK : WHITE;
    if (g_gui.board.to_move != hc) start_ai_think();
}

LRESULT CALLBACK WndProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
    case WM_CREATE: g_gui.hwnd = hwnd; return 0;
    case WM_PAINT: {
        PAINTSTRUCT ps;
        HDC hdc = BeginPaint(hwnd, &ps);
        draw_board(hdc);
        EndPaint(hwnd, &ps);
        return 0;
    }
    case WM_LBUTTONDOWN: {
        int x = LOWORD(lParam), y = HIWORD(lParam);
        int idx = pixel_to_board(x, y);
        if (idx >= 0) do_human_move(idx);
        return 0;
    }
    case WM_MOUSEMOVE: {
        int x = LOWORD(lParam), y = HIWORD(lParam);
        int idx = pixel_to_board(x, y);
        if (idx != g_gui.hover_idx) {
            g_gui.hover_idx = idx;
            InvalidateRect(hwnd, NULL, FALSE);
        }
        return 0;
    }
    case WM_KEYDOWN:
        if (wParam == 'N') new_game();
        else if (wParam == 'P') {
            if (!g_gui.ai_thinking && !g_gui.game_over) {
                int hc = g_gui.human_black ? BLACK : WHITE;
                if (g_gui.board.to_move == hc) {
                    play_move(g_gui.board, PASS);
                    g_gui.last_move = -1;
                    refresh_board();
                    if (g_gui.board.passes >= 2) { g_gui.game_over = true; refresh_board(); }
                    else start_ai_think();
                }
            }
        }
        return 0;
    case WM_USER_AI_DONE: on_ai_done(wParam, lParam); return 0;
    case WM_RBUTTONDOWN: new_game(); return 0;
    case WM_DESTROY:
        // Ask any in-flight search to stop before the message loop unwinds, so
        // the thread we join below finishes promptly.
        g_tree.stop_flag.store(true);
        PostQuitMessage(0);
        return 0;
    }
    return DefWindowProcA(hwnd, msg, wParam, lParam);
}

void load_background_image(const std::string& path) {
    if (path.empty()) return;
    int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, NULL, 0);
    if (wlen <= 0) wlen = MultiByteToWideChar(CP_ACP, 0, path.c_str(), -1, NULL, 0);
    if (wlen <= 0) return;
    std::vector<WCHAR> wpath(wlen);
    MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
    Gdiplus::Image* img = Gdiplus::Image::FromFile(&wpath[0]);
    if (img && img->GetLastStatus() == Gdiplus::Ok) {
        g_gui.bg_image = img;
        g_gui.draw_grid = false;
        std::cerr << "Loaded board image: " << path << std::endl;
    }
    else {
        delete img;
        std::cerr << "Failed to load image: " << path << std::endl;
    }
}

void run_gui_win32(double komi, const SearchParams& sp,
    bool human_black, const std::string& board_image,
    int handicap) {
    g_gui.komi = komi; g_gui.sp = sp;
    g_gui.human_black = human_black;
    g_gui.handicap = handicap;
    g_gui.board.set_handicap(handicap);

    Gdiplus::GdiplusStartupInput gsi;
    ULONG_PTR token;
    Gdiplus::GdiplusStartup(&token, &gsi, NULL);
    load_background_image(board_image);

    HINSTANCE hInst = GetModuleHandleA(NULL);
    const char* cls = "GoAIWindowClass";
    WNDCLASSEXA wc;
    std::memset(&wc, 0, sizeof(wc));
    wc.cbSize = sizeof(wc);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInst;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = cls;
    RegisterClassExA(&wc);

    g_gui.cell = 34; g_gui.margin = 42;
    int w = g_gui.margin * 2 + g_gui.cell * (N - 1);
    int h = g_gui.margin * 2 + g_gui.cell * (N - 1);
    int sw = GetSystemMetrics(SM_CXSCREEN), sh = GetSystemMetrics(SM_CYSCREEN);
    int wx = (sw - w) / 2, wy = (sh - h) / 2 - 20;
    HWND hwnd = CreateWindowExA(0, cls, "GoAI",
        WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX,
        wx, wy, w, h, NULL, NULL, hInst, NULL);
    if (!hwnd) { MessageBoxA(NULL, "Failed", "Error", MB_OK); return; }
    RECT rc = { 0, 0, w, h };
    AdjustWindowRect(&rc, WS_OVERLAPPEDWINDOW & ~WS_THICKFRAME & ~WS_MAXIMIZEBOX, FALSE);
    SetWindowPos(hwnd, NULL, wx, wy, rc.right - rc.left, rc.bottom - rc.top, SWP_NOZORDER);
    ShowWindow(hwnd, SW_SHOW);
    UpdateWindow(hwnd);
    update_status();
    int hc = human_black ? BLACK : WHITE;
    if (g_gui.board.to_move != hc) start_ai_think();
    MSG msg;
    while (GetMessageA(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessageA(&msg);
    }
    if (g_gui.bg_image) { delete g_gui.bg_image; g_gui.bg_image = NULL; }
    // Must happen before returning: the worker touches the global tree and
    // g_gui, both of which outlive this function only until main() returns.
    join_ai_thread();
    Gdiplus::GdiplusShutdown(token);
}
#endif

// Loads nn_weights.bin and reports exactly what it got. Every entry point
// (GUI, GTP, self test, benchmark) goes through here so the diagnosis is
// identical everywhere, and so a mismatch can never pass unnoticed.
static bool load_network_or_report() {
    if (!g_nn.load("nn_weights.bin")) {
        const std::string why = g_nn.load_error.empty()
            ? std::string("unknown reason") : g_nn.load_error;
        std::ifstream probe("nn_weights.bin", std::ios::binary | std::ios::ate);
        if (!probe) {
            std::cerr << "Not loading a neural network: nn_weights.bin was not found in the "
                "current directory (" << why << ").\n"
                "  The engine looks for it relative to the WORKING DIRECTORY, not the "
                "executable. Run from the directory that holds the file, or copy "
                "nn_weights.bin next to the exe.\n"
                "  The search will use heuristic priors and a constant leaf value; a trained "
                "network is what makes this engine play like KataGo rather than plain "
                "heuristic MCTS." << std::endl;
        } else {
            std::cerr << "nn_weights.bin exists but could not be used, so no network is "
                "loaded. Reported reason:\n"
                "  " << why << "\n"
                "  The search will use heuristic priors and a constant leaf value." << std::endl;
        }
        return false;
    }
    std::cerr << "Neural net loaded: ch=" << g_nn.ch
        << " blocks=" << g_nn.n_blocks
        << " in_ch=" << g_nn.in_ch
        << (g_nn.weights_converted ? " pre-converted=yes"
            : (g_nn.layout_converted_at_load
                ? " layout=converted-in-memory" : " layout=UNKNOWN"))
        << std::endl;
    if (!g_nn.weights_converted && !g_nn.layout_converted_at_load) {
        std::cerr << "  WARNING: the layout of this weight file is unknown, so "
            "the network may be evaluating a transposed board." << std::endl;
    }
    if (g_nn.in_ch != 17) {
        std::cerr << "  WARNING: input planes are " << g_nn.in_ch
            << " but the trainer uses 17 (8 moves of history x2 + colour)."
            << std::endl;
    }
    return true;
}

// ============================================================
// main
// ============================================================
int main(int argc, char** argv) {
    init_tables();
    // Printed first, before anything can fail, so a glance at the console says
    // whether the running exe is the one that was just built.
    std::cerr << "GoAI++ build " << GOAI_BUILD_TAG << std::endl;
    double komi = 7.5;
    SearchParams sp;                 // MCTS parameters and their defaults
    sp.max_time = 5.0;
    sp.threads = 2;
    std::string pattern_path = "patterns_19.bin";
    std::string board_image;
    bool gui_mode = true;
    bool human_black = true;
    int handicap = 0;

    {
        std::ifstream test("board.png");
        if (test.good()) board_image = "board.png";
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--selftest") {
            load_network_or_report();
            return run_selftest();
        }
        if (a == "--benchmark") {
            if (!load_network_or_report()) {
                std::cout << "Run 转换权重.bat first: it turns the trainer's "
                    "output into a file this engine can load, then run this "
                    "benchmark again." << std::endl;
                return 2;
            }
            for (int j = i + 1; j < argc; ++j) {
                std::string b = argv[j];
                if (b == "--time" && j + 1 < argc) sp.max_time = std::stod(argv[++j]);
                else if (b == "--threads" && j + 1 < argc) sp.threads = std::stoi(argv[++j]);
                else if (b == "--symmetries" && j + 1 < argc) sp.symmetries = std::stoi(argv[++j]);
                else if (b == "--cpuct" && j + 1 < argc) sp.cpuct = std::stod(argv[++j]);
            }
            return run_benchmark(sp);
        }
        if (a == "--gui") gui_mode = true;
        else if (a == "--cli") gui_mode = false;
        else if (a == "--board" && i + 1 < argc) board_image = argv[++i];
        else if (a == "--color" && i + 1 < argc) {
            std::string c = argv[++i];
            human_black = (c == "black" || c == "b" || c == "B");
        }
        else if (a == "--handicap" && i + 1 < argc) {
            handicap = std::stoi(argv[++i]);
            if (handicap < 0) handicap = 0;
            if (handicap > 9) handicap = 9;
        }
        else if (a == "--random-color") {
            std::mt19937_64 rng(
                (uint32_t)std::chrono::steady_clock::now().time_since_epoch().count());
            human_black = (rng() & 1) == 0;
            std::cerr << "Random color: player is "
                << (human_black ? "BLACK" : "WHITE") << std::endl;
        }
        else if (a == "--selfplay" && i + 1 < argc) {
            SelfplayOptions opt;
            opt.games = std::stoi(argv[++i]);
            opt.komi = komi;
            SearchParams game_sp = sp;
            game_sp.max_time = 1.0;
            game_sp.temperature = 1.0;      // opening diversity in self-play

            for (int j = i + 1; j < argc; ++j) {
                std::string b = argv[j];
                if (b == "--time" && j + 1 < argc) game_sp.max_time = std::stod(argv[++j]);
                else if (b == "--threads" && j + 1 < argc) game_sp.threads = std::stoi(argv[++j]);
                else if (b == "--output" && j + 1 < argc) opt.outdir = argv[++j];
                else if (b == "--komi" && j + 1 < argc) opt.komi = std::stod(argv[++j]);
                else if (b == "--handicap" && j + 1 < argc) opt.handicap = std::stoi(argv[++j]);
                else if (b == "--random-color") opt.random_color = true;
                else if (b == "--learn-from-games") opt.learn_from_games = true;
                else if (b == "--temp" && j + 1 < argc) game_sp.temperature = std::stod(argv[++j]);
                else if (b == "--cpuct" && j + 1 < argc) game_sp.cpuct = std::stod(argv[++j]);
                else if (b == "--symmetries" && j + 1 < argc) game_sp.symmetries = std::stoi(argv[++j]);
                else if (b == "--max-nodes" && j + 1 < argc) game_sp.max_nodes = (size_t)std::stoll(argv[++j]);
                else if (b == "--seed" && j + 1 < argc) game_sp.seed = (uint64_t)std::stoll(argv[++j]);
            }
            opt.seed = game_sp.seed;

            if (g_pattern.load(pattern_path))
                std::cerr << "Pattern DB: " << g_pattern.size() << " patterns" << std::endl;
            load_network_or_report();

            run_selfplay(opt, game_sp);
            return 0;
        }
        else if (a == "--learn" && i + 1 < argc) {
            std::string learn_dir = argv[++i];
            int max_files = -1;
            for (int j = i + 1; j < argc; ++j) {
                std::string b = argv[j];
                if (b == "--max-files" && j + 1 < argc) max_files = std::stoi(argv[++j]);
            }
            std::cerr << "Learning from: " << learn_dir << std::endl;
            int moves = g_pattern.learn_directory(learn_dir, max_files);
            g_pattern.save(pattern_path);
            std::cerr << "Done: " << moves << " moves, "
                << g_pattern.size() << " patterns" << std::endl;
            return 0;
        }
        else if (a == "--time" && i + 1 < argc) sp.max_time = std::stod(argv[++i]);
        else if (a == "--komi" && i + 1 < argc) komi = std::stod(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) sp.threads = std::stoi(argv[++i]);
        else if (a == "--pattern" && i + 1 < argc) pattern_path = argv[++i];
        else if (a == "--cpuct" && i + 1 < argc) sp.cpuct = std::stod(argv[++i]);
        else if (a == "--fpu" && i + 1 < argc) sp.fpu_reduction = std::stod(argv[++i]);
        else if (a == "--virtual-loss" && i + 1 < argc) sp.virtual_loss = std::stod(argv[++i]);
        else if (a == "--root-noise" && i + 1 < argc) sp.root_noise_alpha = std::stod(argv[++i]);
        else if (a == "--temp" && i + 1 < argc) sp.temperature = std::stod(argv[++i]);
        else if (a == "--policy-temp" && i + 1 < argc) sp.policy_temp = std::stod(argv[++i]);
        else if (a == "--symmetries" && i + 1 < argc) sp.symmetries = std::stoi(argv[++i]);
        else if (a == "--max-nodes" && i + 1 < argc) sp.max_nodes = (size_t)std::stoll(argv[++i]);
        else if (a == "--max-sims" && i + 1 < argc) sp.max_sims = std::stoll(argv[++i]);
        else if (a == "--no-nn") sp.use_nn = false;
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage:\n"
                "  go_ai --gui [--color white] [--time 5] [--handicap 2]\n"
                "              [--random-color] [--threads N] [--symmetries 1..8]\n"
                "  go_ai --cli --time 5 [--threads N]      (GTP on stdio)\n"
                "  go_ai --selfplay N [--output DIR] [--time T] [--threads TH]\n"
                "              [--komi K] [--handicap H] [--random-color]\n"
                "              [--temp T] [--learn-from-games] [--seed S]\n"
                "  go_ai --learn DIR [--max-files N]\n"
                "  go_ai --selftest                        (check search invariants)\n"
                "  go_ai --benchmark [--threads N]         (measure search speed)\n"
                "\n"
                "Search options (KataGo-style MCTS):\n"
                "  --time S          seconds per move (default 5, selfplay 1)\n"
                "  --threads N       search threads sharing one tree (default 2)\n"
                "  --cpuct X         PUCT exploration constant (default 1.10)\n"
                "  --fpu X           FPU reduction for unvisited moves (default 0.20)\n"
                "  --virtual-loss X  virtual loss per in-flight visit (default 1.0)\n"
                "  --root-noise X    Dirichlet root noise alpha, 0 disables\n"
                "  --temp T          move selection temperature, 0 = strongest\n"
                "  --policy-temp X   prior sharpening exponent (default 1.10)\n"
                "  --symmetries N    network evaluations per position, 1..8 (default 1)\n"
                "  --max-nodes N     tree size cap (default 2000000)\n"
                "  --max-sims N      fixed simulation budget instead of time\n"
                "  --no-nn           ignore nn_weights.bin\n";
            return 0;
        }
    }

    if (g_pattern.load(pattern_path))
        std::cerr << "Pattern DB: " << g_pattern.size() << " patterns" << std::endl;

    load_network_or_report();

#ifdef _WIN32
    if (gui_mode) {
        std::cerr << "GUI mode  threads=" << sp.threads
            << "  komi=" << komi << "  time=" << sp.max_time << "s"
            << "  handicap=" << handicap << std::endl;
        run_gui_win32(komi, sp, human_black, board_image, handicap);
        return 0;
    }
#endif

    std::cerr << "GTP mode" << std::endl;
    run_gtp(komi, sp);
    return 0;
}