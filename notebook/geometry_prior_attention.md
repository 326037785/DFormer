# Geometry Prior & Geometry Self-Attention — Guided Notes

> These notes repackage the math used in the DFormer encoder into an easy-to-skim walkthrough. Each section mirrors the order of computations in the code so you can jump between equations and implementation without deciphering dense prose.

## 0. Notation & shapes
| Symbol | Meaning | Shape |
| --- | --- | --- |
| $h, w$ | Input image height & width (pixels) | Scalars |
| $p$ | Patch size used for the stem | Scalar |
| $H = h / p$, $W = w / p$ | Token grid height & width | Scalars |
| $N = HW$ | Number of tokens at a stage | Scalar |
| $C$ | Channel dimension per token | Scalar |
| $x \in \mathbb{R}^{N \times C}$ | Token features before attention | $N \times C$ |
| $d \in \mathbb{R}^{h \times w}$ | Original depth map | $h \times w$ |
| $z \in \mathbb{R}^{H \times W}$ | Depth map downsampled to the token grid | $H \times W$ |
| $Q, K, V \in \mathbb{R}^{N \times d_k}$ | Per-head query/key/value matrices | $N \times d_k$ |
| $S, D \in \mathbb{R}^{N \times N}$ | Spatial & depth distance matrices | $N \times N$ |
| $w_s, w_d$ | Learnable scalars mixing spatial & depth cues | Scalars |
| $G \in \mathbb{R}^{N \times N}$ | Geometry prior (log-bias matrix) | $N \times N$ |
| $\beta$ | Per-head decay base in $(0,1)$ | Scalar |

## 1. Depth & spatial priors (definitions)
The goal is to create per-token distances that reflect both image-plane offsets and actual depth discontinuities.

1. **Depth pooling to token scale.** Average pooling with kernel/stride $p$ aligns the depth map with the token grid:
   $$
   z_{ij} = \frac{1}{p^2} \sum_{u=pi}^{p(i+1)-1} \sum_{v=pj}^{p(j+1)-1} d_{uv} \in \mathbb{R}, \qquad z \in \mathbb{R}^{H \times W}.
   $$
2. **Depth distance matrix.** For two token indices $p=(i,j)$ and $q=(i',j')$:
   $$
   D_{pq} = \lvert z_{ij} - z_{i'j'} \rvert, \qquad D \in \mathbb{R}^{N \times N}.
   $$
3. **Spatial Manhattan distance.** Encode 2D offsets with the $L_1$ metric:
   $$
   S_{pq} = \lvert i - i' \rvert + \lvert j - j' \rvert, \qquad S \in \mathbb{R}^{N \times N}.
   $$
Both matrices are symmetric with zeros on the diagonal, so self-relations stay unpenalized.

## 2. Prior fusion → geometry prior $G$
Two learnable scalars (shared by all heads in a block) mix the distance cues into a single log-bias matrix:

1. **Weighted blend.**
   $$
   G_{pq} = w_s S_{pq} + w_d D_{pq}, \qquad w_s, w_d \ge 0.
   $$
2. **Head-specific decay.** Each attention head $c$ carries a learnable slope $\lambda_c < 0$. Adding $\lambda_c G$ to the logits is equivalent to multiplying attention weights by $\beta_c^{G_{pq}}$ with $\beta_c = e^{\lambda_c} \in (0,1)$. The network therefore learns how aggressively to penalize long-range or cross-depth interactions.
3. **Caching shapes.**
   - Full attention: $G \in \mathbb{R}^{B \times N_h \times N \times N}$ after broadcasting batch/head axes.
   - Decomposed attention: reshape $G$ into $G^w \in \mathbb{R}^{B \times N_h \times H \times W \times W}$ (row pass) and $G^h \in \mathbb{R}^{B \times N_h \times W \times H \times H}$ (column pass).

## 3. Geometry self-attention (full head)
With per-head dimensionality $d_k = C / N_h$ and $X \in \mathbb{R}^{B \times H \times W \times C}$:

1. **Linear projections.** Compute $Q$, $K$, $V$ via learned matrices $W_q$, $W_k$, $W_v$.
2. **Reshape & scale.** Rearrange to $Q, K, V \in \mathbb{R}^{B \times N_h \times N \times d_k}$ and scale $K$ by $1/\sqrt{d_k}$.
3. **Rotary embedding (RoPE).** Apply cached sine/cosine pairs so that spatial indices affect $Q$ and $K$ without extra tensors.
4. **Logits with geometry bias.**
   $$
   L_{b,c,p,q} = Q_{b,c,p,:} K_{b,c,q,:}^{\top} + \lambda_c G_{b,p,q}.
   $$
5. **Softmax + decay.**
   $$
   A_{b,c,p,q} = \frac{\exp(L_{b,c,p,q})}{\sum_{q'} \exp(L_{b,c,p,q'})} = \text{Softmax}(L_{b,c,p,:})_q.
   $$
6. **Weighted sum & output.** Multiply $A$ with $V$, add the local enhancement (depthwise $5\times5$ convolution, i.e., LEPE), and project back with $W_o$.

## 4. Decomposed geometry attention (row/column factorization)
Early stages split attention into horizontal and vertical 1D passes to avoid $N^2$ cost.

1. **Shared projections.** Use the same $Q$, $K$, $V$ as in the full case.
2. **Row (width) pass.** For each batch $b$, head $c$, and row index $i$:
   - Extract $Q^w_{b,c,i} \in \mathbb{R}^{W \times d_k}$, $K^w_{b,c,i}$, $V^w_{b,c,i}$.
   - Add row-wise mask $G^w_{b,c,i} \in \mathbb{R}^{W \times W}$.
   - Softmax along the width dimension to obtain $U_{b,c,i} \in \mathbb{R}^{W \times d_k}$.
3. **Column (height) pass.** Using the intermediate $U$:
   - Form $Q^h_{b,c,j} \in \mathbb{R}^{H \times d_k}$ and $K^h_{b,c,j}$ for each column $j$.
   - Inject column mask $G^h_{b,c,j} \in \mathbb{R}^{H \times H}$.
   - Softmax along height to produce $O_{b,c,j} \in \mathbb{R}^{H \times d_k}$.
4. **Stitch & project.** Rearrange $O$ back to $B \times H \times W \times C$, add LEPE, and apply the final linear layer.

## 5. Stage-by-stage cheat sheet
| Stage | Resolution | Token count $N_s$ | Channels $C_s$ | Attention type |
| --- | --- | --- | --- | --- |
| 1 | $H \times W$ (1/4 of image) | $N$ | $C$ | Decomposed |
| 2 | $(H/2) \times (W/2)$ | $N/4$ | $2C$ | Decomposed |
| 3 | $(H/4) \times (W/4)$ | $N/16$ | $4C$ | Decomposed |
| 4 | $(H/8) \times (W/8)$ | $N/64$ | $8C$ | Full |
Depth pooling mirrors the downsampling path so each stage receives a matching $z^{(s)}$ and $G^{(s)}$.

## 6. Practical invariants & sanity checks
- **Zero diagonal:** $G_{pp}=0$ so each token retains its self-attention weight.
- **Symmetry:** $G$ remains symmetric; cross-token penalties are reciprocal.
- **Switch-off capability:** If $w_s = w_d = 0$, geometry terms vanish and the block collapses to vanilla self-attention.
- **Numerical stability:** Negative slopes $\lambda_c$ ensure $\beta_c^{G_{pq}} \le 1$, preventing amplification of logits.
- **Interpretability:** Larger $w_d$ emphasises depth alignment (foreground/background separation); larger $w_s$ keeps attention localized in the image plane.
