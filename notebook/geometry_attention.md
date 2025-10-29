# Geometry Self-Attention Derivations

## Notation & shapes (table)
| Symbol | Definition | Shape |
| --- | --- | --- |
| $h, w$ | Input image height and width in pixels | Scalars |
| $p$ | Patch size in pixels (square patches) | Scalar |
| $H = h / p$ | Number of patch rows | Scalar |
| $W = w / p$ | Number of patch columns | Scalar |
| $N = HW$ | Number of patches (tokens) | Scalar |
| $C$ | Channel dimension of token features | Scalar |
| $x \in \mathbb{R}^{N \times C}$ | Token feature matrix before attention | $N \times C$ |
| $d \in \mathbb{R}^{h \times w}$ | Original depth map aligned with the image | $h \times w$ |
| $z \in \mathbb{R}^{H \times W}$ | Average-pooled depth map aligned with tokens | $H \times W$ |
| $W_q, W_k, W_v \in \mathbb{R}^{C \times d_k}$ | Linear projections for queries, keys, and values | $C \times d_k$ |
| $Q, K, V \in \mathbb{R}^{N \times d_k}$ | Query, key, value projections per head | $N \times d_k$ |
| $D \in \mathbb{R}^{N \times N}$ | Depth-distance matrix | $N \times N$ |
| $S \in \mathbb{R}^{N \times N}$ | Spatial Manhattan-distance matrix | $N \times N$ |
| $M_d, M_s \in \mathbb{R}^{N \times N}$ | Learnable memory weight matrices for depth and spatial priors | $N \times N$ |
| $G \in \mathbb{R}^{N \times N}$ | Geometry prior matrix | $N \times N$ |
| $\beta$ | Scalar decay base sampled in $(0,1]$ | Scalar |
| $\Gamma = \beta^{G} \in (0,1]^{N \times N}$ | Decay-modulated geometry matrix | $N \times N$ |
| $Q_x, K_x, Q_y, K_y \in \mathbb{R}^{N \times d_k}$ | Row/column queries and keys | $N \times d_k$ |
| $G_x \in \mathbb{R}^{N \times W}$ | Row-wise geometry prior | $N \times W$ |
| $G_y \in \mathbb{R}^{N \times H}$ | Column-wise geometry prior | $N \times H$ |
| $f^{(s)} \in \mathbb{R}^{N_s \times C_s}$ | Token features at stage $s$ | $N_s \times C_s$ |
| $z^{(s)} \in \mathbb{R}^{H_s \times W_s}$ | Depth map pooled to stage $s$ | $H_s \times W_s$ |
| $G^{(s)} \in \mathbb{R}^{N_s \times N_s}$ | Stage-$s$ geometry prior | $N_s \times N_s$ |

## Depth & Spatial Priors (definitions)
Average pooling with kernel and stride equal to $p$ converts the depth map $d \in \mathbb{R}^{h \times w}$ into patch-level depths
\[
z_{ij} = \frac{1}{p^2} \sum_{u=pi}^{p(i+1)-1} \sum_{v=pj}^{p(j+1)-1} d_{uv} \in \mathbb{R}, \quad z \in \mathbb{R}^{H \times W}.
\]
The depth-distance matrix is defined element-wise by
\[
D_{(i,j),(i',j')} = \lvert z_{ij} - z_{i'j'} \rvert \in \mathbb{R}_{\ge 0}, \quad D \in \mathbb{R}^{N \times N}.
\]
Spatial priors capture patch offsets using Manhattan distance:
\[
S_{(i,j),(i',j')} = \lvert i - i' \rvert + \lvert j - j' \rvert \in \mathbb{Z}_{\ge 0}, \quad S \in \mathbb{R}^{N \times N}.
\]
Both $D$ and $S$ are symmetric with zeros on the diagonal, ensuring self-relations carry no penalty.

## Prior Fusion → Geometry Prior $\mathbf{G}$
To balance depth and spatial cues, two learnable memory weight matrices $M_d, M_s \in \mathbb{R}^{N \times N}$ scale each prior before summation:
\[
G = M_d \odot D + M_s \odot S \in \mathbb{R}^{N \times N},
\]
where $\odot$ denotes element-wise multiplication. Each entry is
\[
G_{(i,j),(i',j')} = m^{d}_{(i,j),(i',j')} \cdot \lvert z_{ij} - z_{i'j'} \rvert + m^{s}_{(i,j),(i',j')} \cdot (\lvert i - i' \rvert + \lvert j - j' \rvert) \in \mathbb{R}_{\ge 0}.
\]
Learnable non-negative memories constrain $G$ to remain non-negative, preserving the interpretation as a geometry cost.

## Geometry Self-Attention (full)
Standard self-attention over a single head uses projections $Q = x W_q$, $K = x W_k$, $V = x W_v$ with head dimension $d_k$; $W_q, W_k, W_v \in \mathbb{R}^{C \times d_k}$. The vanilla score matrix is $A = Q K^\top \in \mathbb{R}^{N \times N}$. Softmax normalization yields $\text{Softmax}(A)$ row-wise. Geometry self-attention injects the prior through exponential decay:
\[
\Gamma = \beta^{G} = [\beta^{G_{uv}}]_{u,v=1}^{N} \in (0,1]^{N \times N}, \quad \beta \in (0,1].
\]
Row-wise multiplication attenuates distant relations:
\[
\text{GeoAttn}(x) = \left( \text{Softmax}(A) \odot \Gamma \right) V \in \mathbb{R}^{N \times d_k}.
\]
Element-wise powers satisfy $\Gamma_{uu} = 1$ because $G_{uu}=0$. Off-diagonal entries shrink according to geometry costs, preserving the baseline weighting when $G$ vanishes.

Derivation from vanilla attention: starting from $\text{Softmax}(A) = \exp(A) / \sum_{v} \exp(A_{uv})$, inserting $\Gamma$ scales each term to $\exp(A_{uv}) \beta^{G_{uv}}$. Taking logarithms shows the modified logits equal $A_{uv} + G_{uv} \log \beta$, i.e.,
\[
\text{GeoAttn}(x) = \text{Softmax}\big(A + (\log \beta) G\big) V \in \mathbb{R}^{N \times d_k},
\]
which exhibits how geometry prior shifts attention logits before normalization.

## Decomposed Geometry Attention (row/col)
High-resolution features incur quadratic cost. Decomposition applies attention sequentially along columns then rows. Split projections into horizontal and vertical forms by reshaping tokens back to grids while sharing weights:
\[
Q_x = x W_q, \quad K_x = x W_k, \quad Q_y = x W_q, \quad K_y = x W_k \in \mathbb{R}^{N \times d_k}.
\]
Construct per-axis priors by summing over shared indices:
\[
(G_x)_{(i,j),j'} = G_{(i,j),(i,j')} \in \mathbb{R}_{\ge 0}, \quad G_x \in \mathbb{R}^{N \times W},
\]
\[
(G_y)_{(i,j),i'} = G_{(i,j),(i',j)} \in \mathbb{R}_{\ge 0}, \quad G_y \in \mathbb{R}^{N \times H}.
\]
Row-wise decay matrices become $\Gamma_x = \beta^{G_x} \in (0,1]^{N \times W}$ and $\Gamma_y = \beta^{G_y} \in (0,1]^{N \times H}$. The decomposed updates are
\[
Y = \left( \text{Softmax}\big(Q_x K_x^{\top}\big) \odot \Gamma_x^{\uparrow} \right) V \in \mathbb{R}^{N \times d_k},
\]
\[
\text{GeoAttn}_{\text{decomp}}(x) = \left( \text{Softmax}\big(Q_y K_y^{\top}\big) \odot \Gamma_y^{\uparrow} \right) Y \in \mathbb{R}^{N \times d_k},
\]
where $\Gamma_x^{\uparrow}$ and $\Gamma_y^{\uparrow}$ broadcast the per-row and per-column decays back to $N \times N$ during the sequential multiplications. This two-step process equals applying geometry penalties separately along axes, mirroring separable kernels.

## Stage-wise Shape Tracking (1/4, 1/8, 1/16, 1/32)
Stage resolutions follow the encoder hierarchy. Define $N_s = H_s W_s$ and $C_s$ as the token count and channel dimension at stage $s$.

- **Stage 1 (1/4 scale):** Tokens operate on the stem grid, so $H_1 = H$, $W_1 = W$, $N_1 = N$, $C_1 = C$. Depth pooling stride equals the patch embedding stride, producing $z^{(1)} \in \mathbb{R}^{H_1 \times W_1}$ and $G^{(1)} \in \mathbb{R}^{N_1 \times N_1}$.
- **Stage 2 (1/8 scale):** Patch merging halves spatial size, so $H_2 = H_1/2$, $W_2 = W_1/2$, $N_2 = N_1/4$, $C_2 = 2 C_1$. Depth pooling uses kernel and stride $2$ on $z^{(1)}$ to obtain $z^{(2)} \in \mathbb{R}^{H_2 \times W_2}$, then $G^{(2)} \in \mathbb{R}^{N_2 \times N_2}$.
- **Stage 3 (1/16 scale):** $H_3 = H_2/2$, $W_3 = W_2/2$, $N_3 = N_2/4$, $C_3 = 2 C_2$. Depth pooling stride $2$ gives $z^{(3)} \in \mathbb{R}^{H_3 \times W_3}$ and $G^{(3)} \in \mathbb{R}^{N_3 \times N_3}$.
- **Stage 4 (1/32 scale):** $H_4 = H_3/2$, $W_4 = W_3/2$, $N_4 = N_3/4$, $C_4 = 2 C_3$. Depth pooling stride $2$ outputs $z^{(4)} \in \mathbb{R}^{H_4 \times W_4}$, and $G^{(4)} \in \mathbb{R}^{N_4 \times N_4}$. Decomposition is optional at this coarsest level.

Each stage reuses equations above with stage-specific dimensions, ensuring priors align with the current token grid.

## Complexity & Memory (big-O with shapes)
- **Geometry prior construction:** Computing $D$ and $S$ requires pairwise differences over $N$ tokens: $\mathcal{O}(N^2)$ operations and storage. Learnable memories add no extra asymptotic cost beyond storing $2N^2$ parameters.
- **Full geometry self-attention:** Score matrix $A$ costs $\mathcal{O}(N^2 d_k)$ multiplications, identical to vanilla attention. Applying $\Gamma$ multiplies $N^2$ entries. Memory footprint stores $A$, $\Gamma$, and $G$, totaling $\mathcal{O}(N^2)$ elements per head.
- **Decomposed variant:** Column attention handles $H$ tokens per column across $W$ columns, costing $\mathcal{O}(W H^2 d_k)$; row attention similarly costs $\mathcal{O}(H W^2 d_k)$. Combined complexity is $\mathcal{O}(H^2 W d_k + H W^2 d_k)$ with memory $\mathcal{O}(H W (H+W))$ for $G_y$ and $G_x$, reducing quadratic scaling when $H \approx W$.
- **Stage scaling:** Since $N_s = H_s W_s$, each coarser stage reduces $N_s$ by a factor of four, yielding computational savings proportional to $1/4^s$ compared to the first stage.

## Edge Cases & Invariants
- When depth values are uniform, $D$ becomes zero, so $G$ reduces to spatial costs and $\Gamma$ depends solely on Manhattan distance.
- When both depth and spatial memories learn zeros, $G$ vanishes, and geometry self-attention collapses to vanilla self-attention.
- Diagonal entries of $G$ and $\Gamma$ remain zero and one, respectively, ensuring self-token weights are unaffected.
- Symmetry of $D$ and $S$ keeps $G$ symmetric when $M_d$ and $M_s$ are symmetric, stabilizing attention by maintaining reciprocal penalties.
- Non-negative $G$ ensures $\Gamma$ never exceeds one, preventing amplification beyond vanilla attention and keeping probabilities normalized.

## Minimal Takeaways
- Geometry priors arise from explicit depth and spatial distances pooled at token resolution.
- Learnable memories blend depth and spatial costs into a unified non-negative matrix $G$.
- Applying $\beta^{G}$ shifts attention logits by $(\log \beta) G$, shrinking weights for distant tokens.
- Decomposed attention replaces the $N^2$ cost with separable row and column passes guided by $G_x$ and $G_y$.
- Stage-wise pooling keeps geometry priors aligned with multiscale token grids.
- Complexity drops from quadratic to linear-quadratic along axes when using the decomposed form.
- Symmetry and non-negativity of priors preserve invariants such as unit diagonal decay.
- Setting memories to zero recovers vanilla self-attention, revealing geometry guidance as a controllable augmentation.
