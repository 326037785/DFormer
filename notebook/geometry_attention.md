# Geometry Self-Attention Derivations (Markdown Friendly)

## Notation & Shapes
Use bracket notation for shapes. All tensors are row-major with tokens flattened so that `N = H * W`.

- `h`, `w`: input height and width in pixels (scalars)
- `p`: patch size in pixels (scalar)
- `H = floor(h / p)`, `W = floor(w / p)`: patch-grid height and width
- `N = H * W`: number of tokens
- `B`: batch size
- `C`: channels per token
- `N_h`: number of attention heads
- `d_k = d_v = C / N_h`: per-head dimensionality
- `x`: token features before attention, shape `[B, N, C]` (flattened from `[B, H, W, C]`)
- `d_raw`: original depth map, shape `[B, 1, h, w]`
- `z`: depth downsampled to the token grid, shape `[B, 1, H, W]`
- `W_q`, `W_k`, `W_v`: projection matrices, each `[C, d_k]`
- `Q`, `K`, `V`: per-head projections, shape `[B, N_h, N, d_k]`
- `S`: spatial L1 (Manhattan) distance matrix, shape `[N, N]` (cacheable, batch-invariant)
- `D`: depth distance matrix, shape `[B, N, N]` (batch-dependent)
- `M_s`, `M_d`: learnable non-negative "memory" weight matrices (elementwise scalers), each `[N, N]`
- `G`: geometry prior (log-bias base), shape `[B, N, N]`
- `lambda_c`: per-head slope (`< 0`), one scalar per head
- `beta_c = exp(lambda_c)`: decay factor in `(0, 1)` per head
- `Gamma_c = beta_c ** G`: per-head decay matrix, shape `[B, N, N]`

Index note: token `p` corresponds to coordinates `(i, j)` and token `q` to `(i2, j2)` with `i ∈ [0, H-1]`, `j ∈ [0, W-1]`.

---

## Depth & Spatial Priors
Depth pooling (kernel = stride = `p`) aligns depth to the token grid:

- `z[b, 1, i, j]` equals the average of `d_raw[b, 1, u, v]` over `u ∈ [p*i, p*(i+1)-1]`, `v ∈ [p*j, p*(j+1)-1]`
- `z` has shape `[B, 1, H, W]`

Depth distance (batch-dependent):

- `D[b, p, q] = |z[b, 1, i, j] - z[b, 1, i2, j2]|`
- `D` has shape `[B, N, N]`

Spatial L1 distance (batch-invariant):

- `S[p, q] = |i - i2| + |j - j2|`
- `S` has shape `[N, N]`

Both `S` and `D` are symmetric with zero diagonals, so self-relations are unpenalised.

---

## Prior Fusion → Geometry Prior `G`
Elementwise "memory" weighting and fusion:

- `G[b, p, q] = (M_d[p, q] * D[b, p, q]) + (M_s[p, q] * S[p, q])`
- Shapes: `M_d`, `M_s` are `[N, N]`; `D` is `[B, N, N]`; `S` is `[N, N]`; result `G` is `[B, N, N]`

Recommended constraints and interpretation:

- Enforce `M_d ≥ 0`, `M_s ≥ 0` (elementwise) to keep `G ≥ 0`
- If `M_d` and `M_s` are symmetric, `G` remains symmetric (stability)
- Optional scale matching: normalise `S` and `D` by their off-diagonal means to avoid scale mismatch; otherwise the memories absorb scale

Per-head log-bias injection:

- Add `lambda_c * G` to the attention logits per head
- Equivalent view: multiply the attention weights by `Gamma_c = beta_c ** G`, where `beta_c = exp(lambda_c)` and lies in `(0, 1)`

Broadcast rules for adding `G` to logits:

- Full 2D attention: expand `G` to `[B, 1, N, N]`, then broadcast over the head axis
- Decomposed attention: slice `G` into
  - `G_row`: `[B, 1, H, W, W]` for the row/width pass (per row `i`, restrict to tokens `(i, *)`)
  - `G_col`: `[B, 1, W, H, H]` for the column/height pass (per column `j`, restrict to tokens `(*, j)`)

---

## Geometry Self-Attention (Full 2D)
1. **Projections and reshape**
   - From `x` obtain `Q`, `K`, `V` with shapes `[B, N_h, N, d_k]`

2. **Logits with geometry bias**
   - `L[b, c, p, q] = (Q[b, c, p, :] · K[b, c, q, :]) / sqrt(d_k) + lambda_c * G[b, p, q]`

3. **Softmax and decay (equivalence)**
   - `A[b, c, p, q] = softmax_q(L[b, c, p, q])`
   - This is identical to multiplying `exp(Q Kᵀ / sqrt(d_k))` by `Gamma_c = beta_c ** G`, then row-normalising
   - Therefore adding `lambda_c * G` to logits is exactly the same as multiplying the weights by `beta_c ** G`

4. **Output**
   - `O[b, c, p, :] = Σ_q A[b, c, p, q] * V[b, c, q, :]`
   - Concatenate heads: `Y_flat = Concat_c(O)` with shape `[B, N, C]`
   - Reshape back to `[B, H, W, C]`
   - Optional local enhancement (e.g. depthwise 5×5 conv, often called LEPE) before the final projection `W_o`

**Invariants**

- `G[p, p] = 0` ⇒ `Gamma_c[p, p] = 1` ⇒ self-attention is unchanged
- If `M_d = 0` and `M_s = 0` ⇒ `G = 0` ⇒ vanilla attention
- `lambda_c < 0` ⇒ `beta_c ∈ (0, 1)` ⇒ `Gamma_c` entries `≤ 1` (no amplification)

---

## Decomposed Geometry Attention (Row Then Column)
Goal: replace `N²` cost with two sequential 1D attentions along width (rows) and height (columns).

**Row (width) pass** for each batch `b`, head `c`, row `i`:

- Row sequences `Q_row[b, c, i]`, `K_row[b, c, i]`, `V_row[b, c, i]`, each `[W, d_k]`
- Row prior `G_row[b, 1, i]` of shape `[W, W]` by restricting `G` to tokens of row `i`
- `U[b, c, i] = softmax(Q_row K_rowᵀ + lambda_c * G_row[b, 1, i]) · V_row`
- `U[b, c, i]` has shape `[W, d_k]`

**Column (height) pass** for each batch `b`, head `c`, column `j`:

- Column sequences `Q_col[b, c, j]`, `K_col[b, c, j]`, `V_col[b, c, j]`, each `[H, d_k]`
- Column prior `G_col[b, 1, j]` with shape `[H, H]`, restricting `G` to tokens of column `j`
- `O[b, c, j] = softmax(Q_col K_colᵀ + lambda_c * G_col[b, 1, j]) · V_col`
- `O[b, c, j]` has shape `[H, d_k]`

Stitch `O` back to `[B, H, W, C]`, optionally apply LEPE, then project with `W_o`.

**Complexity**

- Full 2D: `O(B * N_h * N² * d_k)`
- Decomposed: `O(B * N_h * H * W * (H + W) * d_k)` (i.e. `HW(H + W)` vs `(HW)²`)

---

## Stage-wise Shape Tracking (1/4, 1/8, 1/16, 1/32)
Let stage `s` have `(H_s, W_s, N_s = H_s * W_s, C_s)`:

- Stage 1: `(H1, W1) = (H, W)`, `N1 = N`, `C1 = C`, use decomposed attention
- Stage 2: `(H2, W2) = (H/2, W/2)`, `N2 = N/4`, `C2 = 2 * C`, decomposed
- Stage 3: `(H3, W3) = (H/4, W/4)`, `N3 = N/16`, `C3 = 4 * C`, decomposed
- Stage 4: `(H4, W4) = (H/8, W/8)`, `N4 = N/64`, `C4 = 8 * C`, full 2D attention

Depth pooling mirrors spatial downsampling so `z^(s)` and `G^(s)` align with `(H_s, W_s)`.

---

## Complexity & Memory (Big-O with Shapes)
- **Prior construction**
  - Build `D` per batch and `S` once: `O(N²)` time and memory
  - Store `M_d`, `M_s`: `2 * N²` parameters
- **Full Geo-Attention**
  - `Q Kᵀ`: `O(N² * d_k)`
  - Add `lambda_c * G`: `O(N²)`
  - Memory ≈ `O(N²)` per head for logits and priors
- **Decomposed**
  - Row pass: `O(H * W * W * d_k)`
  - Column pass: `O(H * H * W * d_k)`
  - Total: `O(H * W * (H + W) * d_k)`
  - Memory ≈ `O(H * W * (H + W))` for sliced priors

---

## Edge Cases & Invariants
- Uniform depth ⇒ `D = 0` ⇒ `G` depends only on `S`
- If `M_d` and `M_s` learn near-zero values ⇒ `G ≈ 0` ⇒ vanilla attention
- Symmetric `M_d`, `M_s` ⇒ symmetric `G` ⇒ reciprocal penalties
- Non-negativity of `G` ⇒ `Gamma_c ≤ 1` elementwise ⇒ no over-amplification
- Missing depth: pool with masks; when a token’s depth is fully invalid, set its `D` row/column to `0` so `G` falls back to `S`

---

## Minimal Takeaways
- `G` comes from explicit depth and spatial distances at token resolution, optionally scaled by learnable `M_d`, `M_s`
- Adding `lambda_c * G` to logits is identical to multiplying attention weights by `beta_c ** G` (with `lambda_c < 0` so `beta_c ∈ (0, 1)`)
- Decomposed attention uses the row/column slices of `G` to reach `H W (H + W)` scaling instead of `N²`
- Stage-wise pooling keeps `z^(s)` and `G^(s)` aligned with multi-scale grids
- Setting the memories to zero recovers vanilla attention
