# Geometry Self-Attention Derivations (ASCII-only, text-safe)

## Notation & Shapes
(Use brackets for shapes; all tensors are row-major, tokens flattened as N = H*W.)
- h, w: input height, width (pixels), scalars
- p: patch size (pixels), scalar
- H = floor(h / p), W = floor(w / p): patch-grid height, width
- N = H*W: number of tokens
- B: batch size
- C: channels per token
- N_h: number of attention heads
- d_k = d_v = C / N_h: per-head dimension
- x: token features before attention, shape [B, N, C]  (flattened from [B, H, W, C])
- d_raw: original depth map, shape [B, 1, h, w]
- z: depth downsampled to token grid, shape [B, 1, H, W]
- W_q, W_k, W_v: projection matrices, each [C, d_k]
- Q, K, V: per-head projections, shape [B, N_h, N, d_k]
- S: spatial L1 (Manhattan) distance matrix, shape [N, N] (cacheable, batch-invariant)
- D: depth distance matrix, shape [B, N, N] (batch-dependent)
- M_s, M_d: learnable nonnegative "memory" weight matrices (elementwise scalers), each [N, N]
- G: geometry prior (log-bias base), shape [B, N, N]
- lambda_c: per-head slope (< 0), one scalar per head
- beta_c = exp(lambda_c), in (0, 1), per head
- Gamma_c = beta_c ^ G: per-head decay matrix, shape [B, N, N]

Index note: token p <-> (i, j), token q <-> (i2, j2), with i in [0..H-1], j in [0..W-1].

---

## Depth & Spatial Priors (definitions)

Depth pooling (kernel = stride = p) aligns depth to the token grid:
- z[b, 1, i, j] = average of d_raw[b, 1, u, v] over u in [p*i .. p*(i+1)-1], v in [p*j .. p*(j+1)-1]
- z has shape [B, 1, H, W]

Depth distance (batch-dependent):
- D[b, p, q] = abs( z[b, 1, i, j] - z[b, 1, i2, j2] )
- D has shape [B, N, N]

Spatial L1 distance (batch-invariant):
- S[p, q] = abs(i - i2) + abs(j - j2)
- S has shape [N, N]

Both S and D are symmetric with zero diagonals -> self-relations are unpenalized.

---

## Prior Fusion -> Geometry Prior G

Elementwise "memory" weighting and fusion:
- G[b, p, q] = (M_d[p, q] * D[b, p, q]) + (M_s[p, q] * S[p, q])
- Shapes: M_d, M_s are [N, N]; D is [B, N, N]; S is [N, N]; result G is [B, N, N]

Recommended constraints and interpretation:
- M_d >= 0, M_s >= 0 (elementwise) to keep G >= 0 (a cost)
- If M_d and M_s are symmetric, G remains symmetric (stability)
- Optional scale matching: divide S and D by their off-diagonal means to avoid scale mismatch; otherwise memories absorb scale

Per-head log-bias injection:
- We add lambda_c * G to the attention logits per head
- Equivalent view: multiply the attention weights by Gamma_c = beta_c ^ G, where beta_c = exp(lambda_c) in (0, 1)

Broadcast rules for adding G to logits:
- Full 2D attention: expand G to [B, 1, N, N], then broadcast over head axis
- Decomposed attention: slice G into
  - G_row: [B, 1, H, W, W] for row/width pass (per row i, restrict to tokens (i, *))
  - G_col: [B, 1, W, H, H] for column/height pass (per column j, restrict to tokens (*, j))

---

## Geometry Self-Attention (full 2D)

1) Projections and reshape
- From x -> Q, K, V with shapes [B, N_h, N, d_k]

2) Logits with geometry bias
- L[b, c, p, q] = ( Q[b, c, p, :] dot K[b, c, q, :] ) / sqrt(d_k)  +  lambda_c * G[b, p, q]

3) Softmax and decay (equivalence)
- A[b, c, p, q] = softmax over q of L[b, c, p, q]
- This is identical to: multiply exp(QK^T / sqrt(d_k)) by Gamma_c = beta_c ^ G, then row-normalize
- Therefore adding lambda_c * G to logits == multiplying weights by beta_c ^ G

4) Output
- O[b, c, p, :] = sum over q of A[b, c, p, q] * V[b, c, q, :]
- Concatenate heads: Y_flat = Concat_c(O) with shape [B, N, C]
- Reshape back to [B, H, W, C]
- Optional local enhancement (e.g., depthwise 5x5 conv, often called LEPE) before final projection W_o

Invariants
- G[p, p] = 0 -> Gamma_c[p, p] = 1 -> self-attention unchanged
- If M_d = 0 and M_s = 0 -> G = 0 -> vanilla attention
- lambda_c < 0 -> beta_c in (0, 1) -> Gamma_c entries <= 1 (no amplification)

---

## Decomposed Geometry Attention (row then column)

Goal: replace N^2 with two sequential 1D attentions along width (rows) and height (columns).

Row (width) pass for each b, head c, row i:
- Take row sequences Q_row[b, c, i], K_row[b, c, i], V_row[b, c, i], each [W, d_k]
- Build row prior G_row[b, 1, i] of shape [W, W] by restricting G to tokens of row i
- U[b, c, i] = softmax( Q_row * K_row^T + lambda_c * G_row[b, 1, i] ) * V_row
- U[b, c, i] has shape [W, d_k]

Column (height) pass for each b, head c, col j:
- From U, form column sequences Q_col[b, c, j], K_col[b, c, j], V_col[b, c, j], each [H, d_k]
- Column prior G_col[b, 1, j] is [H, H], restricting G to tokens of column j
- O[b, c, j] = softmax( Q_col * K_col^T + lambda_c * G_col[b, 1, j] ) * V_col
- O[b, c, j] has shape [H, d_k]

Stitch back O -> [B, H, W, C], optional LEPE, then W_o.

Complexity
- Full 2D: O( B * N_h * N^2 * d_k )
- Decomposed: O( B * N_h * H * W * (H + W) * d_k )
  i.e., HW(H+W) vs (HW)^2

---

## Stage-wise Shape Tracking (1/4, 1/8, 1/16, 1/32)

Let stage s have (H_s, W_s, N_s = H_s*W_s, C_s):
- Stage 1: (H1, W1) = (H, W), N1 = N, C1 = C, use decomposed attention
- Stage 2: (H2, W2) = (H/2, W/2), N2 = N/4, C2 = 2*C, decomposed
- Stage 3: (H3, W3) = (H/4, W/4), N3 = N/16, C3 = 4*C, decomposed
- Stage 4: (H4, W4) = (H/8, W/8), N4 = N/64, C4 = 8*C, full 2D attention
Depth pooling mirrors spatial downsampling so z^(s) and G^(s) align with (H_s, W_s).

---

## Complexity & Memory (big-O with shapes)

- Prior construction:
  - Build D per batch and S once: O(N^2) time and memory
  - Store M_d, M_s: 2*N^2 parameters
- Full Geo-Attn:
  - QK^T: O(N^2 * d_k)
  - Add lambda_c*G: O(N^2)
  - Memory ~ O(N^2) per head for logits and priors
- Decomposed:
  - Row pass: O(H * W * W * d_k)
  - Column pass: O(H * H * W * d_k)
  - Total: O(H * W * (H + W) * d_k)
  - Memory ~ O(H * W * (H + W)) for sliced priors

---

## Edge Cases & Invariants

- Uniform depth -> D = 0 -> G depends only on S
- If M_d, M_s learn near zeros -> G ~ 0 -> vanilla attention
- Symmetry: symmetric M_d, M_s -> symmetric G -> reciprocal penalties
- Nonnegativity of G -> Gamma_c <= 1 elementwise -> no over-amplification
- Missing depth: pool with masks; when a token’s depth is fully invalid, set its D row/col to 0 so G falls back to S for that token

---

## Minimal Takeaways

- G comes from explicit depth and spatial distances at token resolution, optionally scaled by learnable M_d, M_s
- Adding lambda_c * G to logits is exactly the same as multiplying attention weights by beta_c ^ G (with lambda_c < 0 so beta_c in (0, 1))
- Decomposed attention uses the row/col slices of G to reach HW(H+W) scaling instead of N^2
- Stage-wise pooling keeps z^(s) and G^(s) aligned with multi-scale grids
- Setting memories to zero recovers
