# Geometry Prior & Geometry Self-Attention — Guided Notes (Optimized)

## 0. Notation & Shapes
| Symbol | Meaning | Shape |
| --- | --- | --- |
| \(h,w\) | Input image height & width | scalars |
| \(p\) | Patch size | scalar |
| \(H=h/p,\;W=w/p\) | Token grid | scalars |
| \(N=HW\) | Tokens per stage | scalar |
| \(B\) | Batch size | scalar |
| \(C\) | Channels per token | scalar |
| \(N_h\) | Number of heads | scalar |
| \(d_k=d_v=C/N_h\) | Per-head dimension | scalar |
| \(X\) | Input features | \(\mathbb{R}^{B\times H\times W\times C}\) |
| \(d_{\text{raw}}\) | Raw depth map | \(\mathbb{R}^{B\times 1\times h\times w}\) |
| \(z\) | Downsampled depth | \(\mathbb{R}^{B\times 1\times H\times W}\) |
| \(Q,K,V\) | Projected features | \(\mathbb{R}^{B\times N_h\times N\times d_k}\) |
| \(S,D\) | Spatial & depth distances | \(S\in\mathbb{R}^{N\times N},\;D\in\mathbb{R}^{B\times N\times N}\) |
| \(w_s,w_d\) | Fusion weights | scalars |
| \(\lambda_c\) | Per-head slope (<0) | scalar |
| \(G\) | Geometry prior (log-bias) | \(\mathbb{R}^{B\times N\times N}\) |
| \(\beta_c=e^{\lambda_c}\) | Per-head decay base | scalar |

---

## 1. Depth & Spatial Priors

**Depth pooling**  
\[
z_{b,1,i,j}=\frac{1}{p^2}\sum_{u=pi}^{p(i+1)-1}\sum_{v=pj}^{p(j+1)-1}d_{\text{raw},\,b,1,u,v}.
\]

**Depth distance**  
\[
D_{b,p,q}=|z_{b,1,i,j}-z_{b,1,i',j'}|,\quad D\in\mathbb{R}^{B\times N\times N}.
\]

**Spatial distance**  
\[
S_{p,q}=|i-i'|+|j-j'|,\quad S\in\mathbb{R}^{N\times N}.
\]

Both are symmetric with zeros on the diagonal.

---

## 2. Fusion → Geometry Prior \(G\)

**Linear fusion**  
\[
G_{b,p,q}=w_sS_{p,q}+w_dD_{b,p,q},\quad w_s,w_d\ge0.
\]

**Decay interpretation**  
Adding \(\lambda_cG_{b,p,q}\) to logits equals multiplying weights by  
\(\beta_c^{G_{b,p,q}}\), where \(\beta_c=e^{\lambda_c}\in(0,1)\).

**Shapes**
- Full attention: \(G\in\mathbb{R}^{B\times1\times N\times N}\) (broadcast to \(N_h\)).
- Decomposed: \(G^w\in\mathbb{R}^{B\times1\times H\times W\times W}\),
  \(G^h\in\mathbb{R}^{B\times1\times W\times H\times H}\).

---

## 3. Geometry Self-Attention (Full)

**Projections**
\[
Q,K,V=\text{Proj}(X),\quad Q,K,V\in\mathbb{R}^{B\times N_h\times N\times d_k}.
\]

**Logits**
\[
L_{b,c,p,q}=\frac{1}{\sqrt{d_k}}Q_{b,c,p,:}K_{b,c,q,:}^\top+\lambda_cG_{b,p,q}.
\]

**Softmax**
\[
A_{b,c,p,q}=\frac{e^{L_{b,c,p,q}}}{\sum_{q'}e^{L_{b,c,p,q'}}}
=\frac{e^{QK^\top/\sqrt{d_k}}\beta_c^{G_{b,p,q}}}{\sum_{q'}e^{QK^\top/\sqrt{d_k}}\beta_c^{G_{b,p,q'}}}.
\]

**Output**
\[
O_{b,c,p,:}=\sum_qA_{b,c,p,q}V_{b,c,q,:},\quad
Y=\text{Concat}_c(O)\in\mathbb{R}^{B\times N\times C}.
\]
Add local enhancement (LEPE, 5×5 depthwise conv) and output projection.

---

## 4. Decomposed Geometry Attention

**Row pass (width)**  
\[
U_{b,c,i}=\text{Softmax}(Q^wK^{w\top}+\lambda_cG^w_{b,1,i})V^w,
\quad U_{b,c,i}\in\mathbb{R}^{W\times d_k}.
\]

**Column pass (height)**  
\[
O_{b,c,j}=\text{Softmax}(Q^hK^{h\top}+\lambda_cG^h_{b,1,j})V^h,
\quad O_{b,c,j}\in\mathbb{R}^{H\times d_k}.
\]

**Combine**
\[
O\to\mathbb{R}^{B\times H\times W\times C},\quad
Y=\text{LEPE}(O)W_o.
\]

**Complexity**
\[
\text{Full: } \mathcal{O}(BN_hN^2d_k),
\quad
\text{Decomposed: } \mathcal{O}(BN_hHW(H+W)d_k).
\]

---

## 5. Stage-by-Stage Summary
| Stage | Resolution | Tokens \(N_s\) | Channels \(C_s\) | Attention |
| --- | --- | --- | --- | --- |
| 1 | \((H,W)\) (¼) | \(N\) | \(C\) | Decomposed |
| 2 | \((H/2,W/2)\) | \(N/4\) | \(2C\) | Decomposed |
| 3 | \((H/4,W/4)\) | \(N/16\) | \(4C\) | Decomposed |
| 4 | \((H/8,W/8)\) | \(N/64\) | \(8C\) | Full |

Depth pooling mirrors stage downsampling: \(d_{\text{raw}}\to z^{(s)}\to G^{(s)}\).

---

## 6. Invariants & Checks
- \(G_{pp}=0\): self-attention unchanged.  
- \(G_{pq}=G_{qp}\): symmetry preserved.  
- \(w_s=w_d=0\): reduces to vanilla attention.  
- \(\lambda_c<0\): ensures \(\beta_c^{G_{pq}}\le1\).  
- Larger \(w_d\): depth dominates (foreground–background).  
- Larger \(w_s\): planar locality dominates.  
- Multi-head variety: initialize \(\lambda_c\) with different magnitudes to span decay ranges.
