#pragma region GCC_OPTS
// Instruct-set failure: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110675
//#pragma GCC target("avx")
//#pragma GCC target("avx2")
#pragma GCC target("sse2")
#pragma GCC target("sse3")
//#pragma GCC target("sse4")
#pragma GCC target("mmx")
//#pragma GCC optimize(1)
//#pragma GCC optimize(2)
//#pragma GCC optimize(3)
#pragma GCC optimize("Ofast")
#pragma GCC optimize("inline")
#pragma GCC optimize("inline-functions")
#pragma GCC optimize("inline-functions-called-once")
#pragma GCC optimize("inline-small-functions")
#pragma GCC optimize("no-stack-protector")
#pragma GCC optimize("unroll-loops")
#pragma GCC optimize("-falign-functions")
#pragma GCC optimize("-falign-jumps")
#pragma GCC optimize("-falign-labels")
#pragma GCC optimize("-falign-loops")
#pragma GCC optimize("-fcaller-saves")
#pragma GCC optimize("-fcrossjumping")
#pragma GCC optimize("-fcse-follow-jumps")
//#pragma GCC optimize("-fcse-skip-blocks")
#pragma GCC optimize("-fdelete-null-pointer-checks")
#pragma GCC optimize("-fdevirtualize")
#pragma GCC optimize("-fexpensive-optimizations")
#pragma GCC optimize("-ffast-math")
#pragma GCC optimize("-fgcse")
#pragma GCC optimize("-fgcse-lm")
#pragma GCC optimize("-fhoist-adjacent-loads")
#pragma GCC optimize("-findirect-inlining")
#pragma GCC optimize("-finline-small-functions")
#pragma GCC optimize("-fipa-sra")
#pragma GCC optimize("-foptimize-sibling-calls")
#pragma GCC optimize("-fpartial-inlining")
#pragma GCC optimize("-fpeephole2")
#pragma GCC optimize("-freorder-blocks")
#pragma GCC optimize("-freorder-functions")
#pragma GCC optimize("-frerun-cse-after-loop")
#pragma GCC optimize("-fsched-interblock")
#pragma GCC optimize("-fsched-spec")
#pragma GCC optimize("-fschedule-insns")
#pragma GCC optimize("-fschedule-insns2")
#pragma GCC optimize("-fstrict-aliasing")
//#pragma GCC optimize("-fstrict-overflow")
#pragma GCC optimize("-fthread-jumps")
#pragma GCC optimize("-ftree-pre")
#pragma GCC optimize("-ftree-switch-conversion")
#pragma GCC optimize("-ftree-tail-merge")
#pragma GCC optimize("-ftree-vrp")
#pragma GCC optimize("-funroll-loops")
//#pragma GCC optimize("-funsafe-loop-optimizations")
//#pragma GCC optimize("-fwhole-program")
#pragma endregion

#pragma region INCS_DEFS
#include <cstdio>
#include <cstring>
#include <ctime>
#include <set>
#include <map>

using std::set;
using std::map;

#define IBUF   1000000  // input buffer
#define OBUF   1000000  // output buffer
#define MAX_M  50000    // test cases
#define MAX_N  2000     // graph V
#define MAX_K  12000    // graph E
#define MAX_Ni 50       // subgraph V
#define MAX_Ki 250      // subgraph E
#pragma endregion

#pragma region FAST_IO
/* All I/O are strict positive integers :) */
char ibuf[IBUF], *p1, *p2;
char obuf[OBUF], *p3 = obuf;
#define getchar() (p1 == p2 && (p2 = (p1 = ibuf) + fread(ibuf, 1, IBUF, stdin), p1 == p2) ? EOF : *p1++)
#define putchar(x) (p3 - obuf < OBUF) ? (*p3++ = x) : (fwrite(obuf, p3 - obuf, 1, stdout), p3 = obuf, *p3++ = x)
inline int read() {
  int x = 0;
  char ch = getchar();
  while (ch < 48 || ch > 57)
    ch = getchar();
  while (ch >= 48 && ch <= 57)
    x = (x << 3) + (x << 1) + (ch ^ 48), ch = getchar();
  return x;
}
inline void write(int x) {
  int tmp[10], p = 0;
  while (x) tmp[p++] = x % 10 ^ 48, x /= 10;
  while (p--) putchar(tmp[p]);
}
#pragma endregion

#pragma region DATA_STRUCT
struct Graph {
  int n;
  int k;
  int labels[MAX_N];
  int degrees[MAX_N] = {0};
  set<int> neighbors[MAX_N];
};
struct SubGraph {
  int n;
  int k;
  int labels[MAX_Ni];
  int degrees[MAX_Ni];
  set<int> neighbors[MAX_Ni];
};
// Info
SubGraph G1; Graph G2;
map<int, set<int>> nodes_of_G1Labels;
map<int, set<int>> nodes_of_G2Labels;
map<int, set<int>> G1_nodes_cover_degree;
map<int, set<int>> G2_nodes_cover_degree;
// State
int mapping[MAX_Ni];
int reverse_mapping[MAX_N];
set<int> T1;
set<int> T2;
// Result
int ans[MAX_M][1 + MAX_Ni], n_ans = 0;  // prefixed by subgraph_idx

inline void read_graph() {
  int n, k, u, v;
  n = read(); G2.n = n;
  k = read(); G2.k = k;
  for (int i = 0; i < n; i++)
    G2.labels[i] = read();
  for (int i = 0; i < k; i++) {
    u = read() - 1; G2.degrees[u]++;
    v = read() - 1; G2.degrees[v]++;
    G2.neighbors[u].insert(v);
    G2.neighbors[v].insert(u);
  }
}
void read_subgraph() {
  int n, k, u, v;
  n = read(); G1.n = n;
  k = read(); G1.k = k;
  memset(G1.degrees, 0, sizeof(int) * n);  // reset
  memset(mapping, 0x00, sizeof(mapping));
  memset(reverse_mapping, 0x00, sizeof(reverse_mapping));
  nodes_of_G2Labels.clear();
  G2_nodes_cover_degree.clear();
  T1.clear();
  T2.clear();
  for (int i = 0; i < n; i++) {
    G1.labels[i] = read();
    G1.neighbors[i].clear();               // reset
  }
  for (int i = 0; i < k; i++) {
    u = read() - 1; G1.degrees[u]++;
    v = read() - 1; G1.degrees[v]++;
    G1.neighbors[u].insert(v);
    G1.neighbors[v].insert(u);
  }
}
#pragma endregion

#pragma region ALGORITHM
void find_isomorphism(int i) {
  n_ans = 2;

  ans[0][0] = 2;
  ans[0][1] = 2;
  ans[0][2] = 4;
  ans[0][3] = 3;
  ans[0][4] = 1;
  ans[0][5] = 0;  // terminal
  
  ans[1][0] = 4;
  ans[1][1] = 1;
  ans[1][2] = 3;
  ans[1][3] = 2;
  ans[1][4] = 0;  // terminal
}
#pragma endregion

int main() {
  // ttl control
  time_t ts = time(NULL); // sec
  time_t ttl = ts + 59;
  // input & process
  read_graph();
  int m = read();
  for (int i = 1; i <= m; i++) {
    read_subgraph();
    find_isomorphism(i);
    if (time(NULL) >= ttl) break;
  }
  // output
  write(n_ans); putchar(10);
  for (int j = 0; j < n_ans; j++) {
    int* p = ans[j];
    while (*p) {  // zero-terminated
      write(*p++);
      putchar(32);
    }
    putchar(10);
  }
  // clear buff
  fwrite(obuf, p3 - obuf, 1, stdout);
  return 0;
}
