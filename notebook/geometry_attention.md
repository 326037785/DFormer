# Geometry Self-Attention — Detailed Notes

## 0. Notation & Shapes
| Symbol | Meaning | Shape |
| --- | --- | --- |
| \(h,w\) | Input image height & width | scalars |
| \(p\) | Patch size | scalar |
| \(H = \lfloor h/p \rfloor,\; W = \lfloor w/p \rfloor\) | Token grid resolution | scalars |
| \(N = HW\) | Tokens per stage | scalar |
| \(B\) | Batch size | scalar |
| \(C\) | Channels per token | scalar |
| \(N_h\) | Number of attention heads | scalar |
| \(d_k = d_v = C/N_h\) | Per-head dimension | scalar |
| \(X\) | Flattened token features | \(\mathbb{R}^{B \times N \times C}\) |
| \(d_{\text{raw}}\) | Raw depth map | \(\mathbb{R}^{B \times 1 \times h \times w}\) |
| \(z\) | Depth aligned to the token grid | \(\mathbb{R}^{B \times 1 \times H \times W}\) |
| \(Q,K,V\) | Projected features | \(\mathbb{R}^{B \times N_h \times N \times d_k}\) |
| \(S,D\) | Spatial & depth distances | \(S\in\mathbb{R}^{N\times N},\; D\in\mathbb{R}^{B\times N\times N}\) |
| \(M_s,M_d\) | Learnable non-negative weights | \(\mathbb{R}^{N\times N}\) |
| \(G\) | Geometry prior (log-bias) | \(\mathbb{R}^{B \times N \times N}\) |
| \(\lambda_c\) | Per-head slope (\(<0\)) | scalar |
| \(\beta_c = e^{\lambda_c}\) | Per-head decay base | scalar |

Tokens \(p\) and \(q\) correspond to grid positions \((i,j)\) and \((i',j')\).

---

## 1. Depth & Spatial Priors

**Depth pooling**
\[
z_{b,1,i,j} = \frac{1}{p^2} \sum_{u = pi}^{p(i+1)-1} \sum_{v = pj}^{p(j+1)-1} d_{\text{raw},\,b,1,u,v},\quad z \in \mathbb{R}^{B \times 1 \times H \times W}.
\]

**Depth distance**
\[
D_{b,p,q} = \left| z_{b,1,i,j} - z_{b,1,i',j'} \right|,\quad D \in \mathbb{R}^{B \times N \times N}.
\]

**Spatial L1 distance**
\[
S_{p,q} = |i - i'| + |j - j'|,\quad S \in \mathbb{R}^{N \times N}.
\]

Both matrices are symmetric with zero diagonals.

---

## 2. Geometry Prior Construction

**Elementwise fusion**
\[
G_{b,p,q} = M_d[p,q] \cdot D_{b,p,q} + M_s[p,q] \cdot S_{p,q},\quad M_s,M_d \ge 0.
\]

**Broadcast shapes**
- Full attention: \(G \in \mathbb{R}^{B \times 1 \times N \times N}\) and broadcast along the head axis.
- Decomposed attention:
  - Row prior: \(G^{(w)} \in \mathbb{R}^{B \times 1 \times H \times W \times W}\).
  - Column prior: \(G^{(h)} \in \mathbb{R}^{B \times 1 \times W \times H \times H}\).

**Decay interpretation**
Adding \(\lambda_c G_{b,p,q}\) to logits is equivalent to multiplying the attention weights by \(\beta_c^{G_{b,p,q}}\), where \(\beta_c = e^{\lambda_c} \in (0,1)\).

---

## 3. Geometry Self-Attention (Full 2D)

**Projections**
\[
Q,K,V = \text{Proj}(X),\quad Q,K,V \in \mathbb{R}^{B \times N_h \times N \times d_k}.
\]

**Logits with geometry bias**
\[
L_{b,c,p,q} = \frac{1}{\sqrt{d_k}} Q_{b,c,p,:} K_{b,c,q,:}^{\top} + \lambda_c G_{b,p,q}.
\]

**Softmax & equivalence**
\[
A_{b,c,p,q} = \frac{e^{L_{b,c,p,q}}}{\sum_{q'} e^{L_{b,c,p,q'}}} = \frac{e^{QK^{\top}/\sqrt{d_k}}\, \beta_c^{G_{b,p,q}}}{\sum_{q'} e^{QK^{\top}/\sqrt{d_k}}\, \beta_c^{G_{b,p,q'}}}.
\]

**Output**
\[
O_{b,c,p,:} = \sum_q A_{b,c,p,q} V_{b,c,q,:},\quad Y = \text{Concat}_c(O) \in \mathbb{R}^{B \times N \times C}.
\]
Optionally apply local enhancement (e.g., 5×5 depthwise conv/LEPE) before the final projection \(W_o\).

---

## 4. Decomposed Geometry Attention (Row → Column)

### Row (width) pass
For each batch \(b\), head \(c\), row \(i\):
\[
U_{b,c,i} = \operatorname{Softmax}\big( Q^{(w)}_{b,c,i} K^{(w)\top}_{b,c,i} + \lambda_c G^{(w)}_{b,1,i} \big) V^{(w)}_{b,c,i},\quad U_{b,c,i} \in \mathbb{R}^{W \times d_k}.
\]

### Column (height) pass
For each batch \(b\), head \(c\), column \(j\):
\[
O_{b,c,j} = \operatorname{Softmax}\big( Q^{(h)}_{b,c,j} K^{(h)\top}_{b,c,j} + \lambda_c G^{(h)}_{b,1,j} \big) V^{(h)}_{b,c,j},\quad O_{b,c,j} \in \mathbb{R}^{H \times d_k}.
\]

Stitch \(O\) back to \(\mathbb{R}^{B \times H \times W \times C}\), optionally apply LEPE, then project with \(W_o\).

**Complexity**
\[
\text{Full 2D: } \mathcal{O}(B N_h N^2 d_k),\qquad \text{Decomposed: } \mathcal{O}(B N_h HW (H+W) d_k).
\]

---

## 5. Stage-by-Stage Overview
| Stage | Resolution | Tokens \(N_s\) | Channels \(C_s\) | Attention |
| --- | --- | --- | --- | --- |
| 1 | \((H, W)\) | \(N\) | \(C\) | Decomposed |
| 2 | \((H/2, W/2)\) | \(N/4\) | \(2C\) | Decomposed |
| 3 | \((H/4, W/4)\) | \(N/16\) | \(4C\) | Decomposed |
| 4 | \((H/8, W/8)\) | \(N/64\) | \(8C\) | Full |

Depth pooling mirrors spatial downsampling: \(d_{\text{raw}} \rightarrow z^{(s)} \rightarrow G^{(s)}\).

---

## 6. Checks & Invariants
- \(G_{pp} = 0\) ⇒ self-attention remains unchanged.
- \(G_{pq} = G_{qp}\) when \(M_s, M_d\) are symmetric.
- \(M_s = M_d = 0\) reduces to vanilla attention.
- \(\lambda_c < 0\) ensures \(\beta_c^{G_{pq}} \le 1\) (only decay).
- Larger \(M_d\): depth dominates (foreground vs. background separation).
- Larger \(M_s\): spatial locality dominates.
- Vary \(\lambda_c\) across heads to span different decay rates.
