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
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <set>
#include <map>
#include <deque>

using std::vector;
using std::set;
using std::map;
using std::deque;

#define IBUF   1000000  // input buffer
#define OBUF   1000000  // output buffer
#define MAX_M  50000    // test cases
#define MAX_N  2000     // graph V (1~2000)
#define MAX_K  12000    // graph E
#define MAX_Ni 50       // subgraph V (1~50)
#define MAX_Ki 250      // subgraph E
#define MAX_L  10       // label id (1~10)

#define max(a, b) (a) > (b) ? (a) : (b)
#define min(a, b) (a) < (b) ? (a) : (b)
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
  int degree[MAX_N] = {0};
  set<int> neighbors[MAX_N];
};
struct SubGraph {
  int n;
  int k;
  int labels[MAX_Ni];
  int degree[MAX_Ni];
  set<int> neighbors[MAX_Ni];   // TODO: use pure 2d-array?
};
struct Frame {
  int u;           // cur_node
  deque<int> nxt;  // candidates
};
// Info
SubGraph G1; Graph G2;
set<int> nodes_of_G1Labels[1 + MAX_L];
set<int> nodes_of_G2Labels[1 + MAX_L];
int label_rarity_G2[1 + MAX_L];
map<int, set<int>> nodes_of_G1Degrees;
map<int, set<int>> nodes_of_G2Degrees;
// State
Frame S[MAX_Ni]; int Sp;
int node_order[MAX_Ni];
int mapping[MAX_Ni];
int reverse_mapping[MAX_N];
bool T1[MAX_Ni];
bool T2[MAX_N];
set<int> u_labels_successors[1 + MAX_L];
set<int> v_labels_successors[1 + MAX_L];
// Result
int ans[MAX_M][1 + MAX_Ni], n_ans = 0;  // prefixed by subgraph_idx

bool DEBUG = false;

void debug_node_order() {
  printf("node_order: ");
  for (int k = 0; k < G1.n; k++)
    printf("%d ", node_order[k]);
  printf("\n");
}
void debug_array(int x[], int len, char* name) {
  printf("array %s: [", name);
  for (int i = 0; i < len; i++)
    printf("%d ", x[i]);
  printf("]\n");
}
void debug_set(set<int> &s, char* name) {
  printf("set %s: {", name);
  for (auto it : s)
    printf("%d ", it);
  printf("}\n");
}
void debug_map_set(map<int, set<int>> &ms, char* name) {
  printf("map-set %s: {", name);
  for (auto it : ms) {
    printf("%d {", it.first);
    for (auto e : it.second)
      printf("%d ", e);
    printf("}");
  }
  printf("}\n");
}
void debug_frame(Frame &frm) {
  printf("frm: u=%d, nxt=[", frm.u);
  for (auto it : frm.nxt)
    printf("%d ", it);
  printf("]\n");
}

inline void read_graph() {
  int n, k, u, v, l;
  n = read(); G2.n = n;
  k = read(); G2.k = k;
  for (int i = 0; i < n; i++) {
    l = read();
    G2.labels[i] = l;
    nodes_of_G2Labels[l].insert(i);
  }
  for (int i = 0; i < k; i++) {
    u = read() - 1; G2.degree[u]++; // offset
    v = read() - 1; G2.degree[v]++;
    G2.neighbors[u].insert(v);
    G2.neighbors[v].insert(u);
  }
  if (DEBUG) debug_array(G2.degree, G2.n, "G2.degree");
  // setup
  for (int l = 1; l <= MAX_L; l++)
    label_rarity_G2[l] = nodes_of_G2Labels[l].size();
  if (DEBUG) debug_array(label_rarity_G2, 1+MAX_L, "label_rarity_G2");
  for (int i = 0; i < n; i++) {
    int D = G2.degree[i];
    if (nodes_of_G2Degrees.find(D) == nodes_of_G2Degrees.end()) // not found
      nodes_of_G2Degrees[D] = {i};
    else
      nodes_of_G2Degrees[D].insert(i);
  }
  if (DEBUG) debug_map_set(nodes_of_G2Degrees, "nodes_of_G2Degrees");
}
inline void read_subgraph() {
  int n, k, u, v, l;
  n = read(); G1.n = n;
  k = read(); G1.k = k;
  for (int i = 1; i <= MAX_L; i++)
    nodes_of_G1Labels[i].clear();
  for (int i = 0; i < n; i++) {
    l = read();
    G1.labels[i] = l;
    nodes_of_G1Labels[l].insert(i);
    G1.neighbors[i].clear();
  }
  memset(G1.degree, 0x00, sizeof(int) * n);
  for (int i = 0; i < k; i++) {
    u = read() - 1; G1.degree[u]++; // offset
    v = read() - 1; G1.degree[v]++;
    G1.neighbors[u].insert(v);
    G1.neighbors[v].insert(u);
  }
  if (DEBUG) debug_array(G1.degree, G1.n, "G1.degree");
  // reset
  memset(mapping, 0xFF, sizeof(int) * n); // INVALID = -1
  memset(reverse_mapping, 0xFF, sizeof(reverse_mapping)); // INVALID = -1
  memset(T1, 0x00, sizeof(T1));
  memset(T2, 0x00, sizeof(T2));
  nodes_of_G1Degrees.clear();
  for (int i = 0; i < n; i++) {
    int D = G1.degree[i];
    if (nodes_of_G1Degrees.find(D) == nodes_of_G1Degrees.end()) // not found
      nodes_of_G1Degrees[D] = {i};
    else
      nodes_of_G1Degrees[D].insert(i);
  }
  if (DEBUG) debug_map_set(nodes_of_G1Degrees, "nodes_of_G1Degrees");
}
inline void write_ans() {
  if (!n_ans) putchar(48);
  else write(n_ans);
  putchar(10);
  for (int j = 0; j < n_ans; j++) {
    int* p = ans[j];
    while (*p >= 0) {  // INVALID = -1
      write(1 + *p++);  // inv-offset
      putchar(32);
    }
    putchar(10);
  }
}
#pragma endregion

#pragma region ALGORITHM
inline void intersection_update(set<int> &a, set<int> &b) {
  for (auto it = a.begin(); it != a.end();)
    if (b.find(*it) == b.end()) { // not found
      it = a.erase(it);
    } else it++;
}
inline void difference_update(set<int> &s, int x[]) {
  for (auto it = s.begin(); it != s.end();)
    if (x[*it] >= 0) { // mapped
      it = s.erase(it);
    } else it++;
}
inline int count_exist(set<int> &s, bool flag[]) {
  int cnt = 0;
  for (auto e : s)
    if (flag[e]) // found
      cnt++;
  return cnt;
}

inline void _make_order() {
  int label_rarity[1 + MAX_L];
  memcpy(label_rarity, label_rarity_G2, sizeof(label_rarity_G2)); // copy!
  int used_degree[MAX_Ni]; memset(used_degree, 0x00, sizeof(used_degree));
  set<int> V1_unordered;
  for (int i = 0; i < G1.n; i++)
    V1_unordered.insert(i);
  int p = 0;  // ptr(node_order)
  int tmp;

  while (!V1_unordered.empty()) {
    int max_rarity = G2.n;
    for (auto n : V1_unordered)
      max_rarity = min(max_rarity, label_rarity[G1.labels[n]]);
    if (DEBUG) printf("max_rarity: %d\n", max_rarity);
    int sel_node = -1, max_deg = -1;
    for (auto n : V1_unordered) {
      if (label_rarity[G1.labels[n]] == max_rarity) { // max rarity
        tmp = G1.degree[n];
        if (tmp > max_deg) {  // max deg
          max_deg = tmp;
          sel_node = n;
        }
      }
    }
    if (DEBUG) printf("max_node: %d\n", sel_node);

    bool visited[MAX_Ni]; memset(visited, 0x00, sizeof(visited));
    visited[sel_node] = true;
    set<int> current_layer = {sel_node};
    set<int> next_layer;
    set<int> nodes_to_add;
    vector<int> max_used_degree_nodes;
    vector<int> max_degree_nodes;
    while (!current_layer.empty()) {
      nodes_to_add.clear();
      nodes_to_add = current_layer;   // copy!
      while (!nodes_to_add.empty()) {
        // max_used_degree
        int max_used_degree = -1;
        for (auto n : nodes_to_add)
          max_used_degree = max(max_used_degree, used_degree[n]);
        if (DEBUG) printf("max_used_degree: %d\n", max_used_degree);
        max_used_degree_nodes.clear();
        for (auto n : nodes_to_add)
          if (used_degree[n] == max_used_degree)
            max_used_degree_nodes.push_back(n);
        // max_degree
        int max_degree = -1;
        for (auto n : max_used_degree_nodes)
          max_degree = max(max_degree, G1.degree[n]);
        if (DEBUG) printf("max_degree: %d\n", max_degree);
        max_degree_nodes.clear();
        for (auto n : max_used_degree_nodes)
          if (G1.degree[n] == max_degree)
            max_degree_nodes.push_back(n);
        // max_rarity
        int next_node = -1, min_cnt = G2.n;
        for (auto n : max_degree_nodes) {
          tmp = label_rarity[G1.labels[n]];
          if (tmp < min_cnt) {
            min_cnt = tmp;
            next_node = n;
          }
        }
        if (DEBUG) printf("next_node: %d\n", next_node);
        // go!
        nodes_to_add.erase(next_node);
        V1_unordered.erase(next_node);
        node_order[p++] = next_node;
        label_rarity[G1.labels[next_node]]--;
        for (auto n : G1.neighbors[next_node])
          used_degree[n]++;
      }

      next_layer.clear();
      for (auto n : current_layer)
        for (auto e : G1.neighbors[n])
          if (!visited[e]) {
            visited[e] = true;
            next_layer.insert(e);
          }
      current_layer = next_layer;
    }
  }
}
inline void _make_frame(int u) {
  set<int> covered_neighbors;
  for (auto nbr : G1.neighbors[u])
    if (mapping[nbr] >= 0)   // mapped
      covered_neighbors.insert(nbr);

  set<int> &valid_label_nodes = nodes_of_G2Labels[G1.labels[u]];
  set<int> valid_degree_nodes;
  int D = G1.degree[u];
  for (int i = 0; i <= G2.n; i++)
    if (G2.degree[i] >= D)
      valid_degree_nodes.insert(i);

  if (DEBUG) {
    debug_set(covered_neighbors, "covered_neighbors");
    debug_set(valid_label_nodes, "valid_label_nodes");
    debug_set(valid_degree_nodes, "valid_degree_nodes");
  }

  set<int> nxt;
  if (covered_neighbors.empty()) {
    if (DEBUG) printf("case-init\n");
    nxt = valid_label_nodes;  // copy!
    if (DEBUG) debug_set(nxt, "nxt");
    intersection_update(nxt, valid_degree_nodes);
    if (DEBUG) debug_set(nxt, "nxt");
    difference_update(nxt, reverse_mapping);
    if (DEBUG) debug_set(nxt, "nxt");
  } else {
    if (DEBUG) printf("case-succ\n");
    bool flag = true;
    for (auto nbr : covered_neighbors) {
      if (flag) {
        nxt = G2.neighbors[mapping[nbr]];  // copy!
        flag = false;
      } else {
        intersection_update(nxt, G2.neighbors[mapping[nbr]]);
      }
      if (DEBUG) {
        printf("nbr: %d\n", nbr);
        printf("mapping[nbr]: %d\n", mapping[nbr]);
        debug_set(G2.neighbors[mapping[nbr]], "G2.neighbors[mapping[nbr]]");
        debug_set(nxt, "nxt-i");
      }
    }
    if (DEBUG) debug_set(nxt, "nxt");
    difference_update(nxt, reverse_mapping);
    if (DEBUG) debug_set(nxt, "nxt");
    intersection_update(nxt, valid_degree_nodes);
    if (DEBUG) debug_set(nxt, "nxt");
    intersection_update(nxt, valid_label_nodes);
    if (DEBUG) debug_set(nxt, "nxt");
  }
  Frame &frm = S[Sp - 1];
  frm.u = u;
  frm.nxt.clear();
  for (auto e : nxt)
    frm.nxt.push_back(e);
}
inline bool _cut_PT(int u, int v) {
  for (int l = 1; l <= MAX_L; l++) {
    u_labels_successors[l].clear();
    v_labels_successors[l].clear();
  }
  for (auto n1 : G1.neighbors[u])
    u_labels_successors[G1.labels[n1]].insert(n1);
  for (auto n2 : G2.neighbors[v])
    v_labels_successors[G2.labels[n2]].insert(n2);

  for (int l = 1; l <= MAX_L; l++) {
    if (u_labels_successors[l].size() && v_labels_successors[l].empty()) {
      if (DEBUG) printf("cut by size()\n");
      return true;
    }
    if (count_exist(u_labels_successors[l], T1) > count_exist(v_labels_successors[l], T2)) {
      if (DEBUG) printf("cut by count_exist()\n");
      return true;
    }
  }
  return false;
}
inline void _update_Tinout(int u, int v) {
  for (auto succ : G1.neighbors[u])
    if (mapping[succ] < 0) // not mapped
      T1[succ] = true;
  for (auto succ : G2.neighbors[v])
    if (reverse_mapping[succ] < 0) // not mapped
      T2[succ] = true;
  T1[u] = false;
  T2[v] = false;
}
inline void _restore_Tinout(int u, int v) {
  bool found;
  for (auto neighbor : G1.neighbors[u]) {
    if (mapping[neighbor] >= 0) T1[u] = true;  // mapped
    else {
      found = false;
      for (auto nbr : G1.neighbors[neighbor])
        if (mapping[nbr] >= 0) { // mapped
          found = true;
          break;
        }
      if (!found) T1[neighbor] = false;
    }
  }
  for (auto neighbor : G2.neighbors[v]) {
    if (reverse_mapping[neighbor] >= 0) T2[v] = true;  // mapped
    else {
      found = false;
      for (auto nbr : G2.neighbors[neighbor])
        if (reverse_mapping[nbr] >= 0) { // mapped
          found = true;
          break;
        }
      if (!found) T2[neighbor] = false;
    }
  }
}

void check_isomorphism() {
  for (int u = 0; u < G1.n; u++) {
    int u_lbl = G1.labels[u];
    int v = mapping[u];
    if (u_lbl != G2.labels[v]) {
      printf("Wrong answer!");
      exit(-1);
    }
    auto X = G2.neighbors[v];
    for (auto e : G1.neighbors[u]) {
      if (X.find(mapping[e]) == X.end()) {  // not found
        printf("Wrong answer!");
        exit(-1);
      }
    }
  }
}
void find_isomorphism(int sgid) {
  for (int l = 1; l <= MAX_L; l++)
    if (nodes_of_G1Labels[l].size() > nodes_of_G2Labels[l].size())
      return;

  //for (auto it : nodes_of_G1Degrees) {
  //  int D = it.first;
  //  if (nodes_of_G2Degrees.find(D) == nodes_of_G2Degrees.end()) // not found
  //    return;
  //  //if (it.second.size() > nodes_of_G2Degrees[D].size())
  //  //  return;
  //}

  Sp = 0;  // ptr(node_order/stack), aka. dfs depth 
  _make_order();
  if (DEBUG) debug_node_order();

  _make_frame(node_order[Sp++]);
  while (Sp) {
    Frame &frm = S[Sp - 1];
    if (DEBUG) debug_frame(frm);

    if (frm.nxt.empty()) { // no more candidates
      Sp--;
      if (Sp) {    // backtrace to last frame
        Frame &frm = S[Sp - 1];
        int u = frm.u;
        int v = mapping[u];
        mapping[u] = -1;
        reverse_mapping[v] = -1;
        _restore_Tinout(u, v);
      }
      continue;
    }

    int u = frm.u;
    int v = frm.nxt.front(); frm.nxt.pop_front(); // try this new candidate
    bool cut = _cut_PT(u, v);
    if (DEBUG) printf("u->v: %d -> %d; cut=%d\n", u, v, cut);

    if (!cut) {
      if (Sp == G1.n) {
        mapping[u] = v;
        int *q = ans[n_ans++];
        *q++ = sgid;  // prefix with subgraph_id
        for (int i = 0; i < G1.n; i++)
          *q++ = mapping[i];
        *q = -1;      // mark terminal
        return;       // well done, exit!
      } else {
        mapping[u] = v;
        reverse_mapping[v] = u;
        _update_Tinout(u, v);
        _make_frame(node_order[Sp++]);
      }
    }
  }
}
#pragma endregion

int main() {
  // debug
  if(getenv("LOCAL_DEBUG")) DEBUG = true;
  // ttl control
  clock_t ts = clock();
  clock_t ttl = ts + CLOCKS_PER_SEC * 59;
  // process
  read_graph();
  int m = read();
  for (int i = 0; i < m; i++) { // offset at output :)
    if (clock() >= ttl) break;
    read_subgraph();
    find_isomorphism(i);
  }
  write_ans();
  // clear buff
  fwrite(obuf, p3 - obuf, 1, stdout);
  // TIMEIT
  printf("\nCPP TIME: %.2f\n", float(clock() - ts) / CLOCKS_PER_SEC);
  return 0;
}
