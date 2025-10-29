# Geometry Prior and Geometry Self-Attention Breakdown

## 1. Plain-language summary of the mathematical tricks
- **Depth patches as geometry probes.** The raw depth image is resized to the token grid used in each encoder stage so that every visual token shares a co-located depth sample. The average/resize step condenses the pixels inside a patch into a single distance-from-camera reading that can act as the token's 3D anchor.【F:models/encoders/DFormerv2.py†L196-L206】【F:sgmixed.txt†L231-L260】
- **Manhattan and depth distances become per-head decay factors.** For each attention head, the model multiplies the Manhattan distance between token coordinates and the absolute difference between their depth values by a learnable log-decay slope. Adding this log-bias to the attention logits is equivalent to multiplying the attention weights by \(\beta^{\text{distance}}\) with head-specific \(0<\beta<1\), shrinking the influence of geometrically distant tokens.【F:models/encoders/DFormerv2.py†L129-L191】【F:sgmixed.txt†L261-L320】
- **Two learnable "memory" coefficients mix spatial and depth cues.** A pair of trainable scalars controls how much the attention mask trusts spatial layout versus depth similarity, allowing the network to favor whichever signal is more reliable in a scene.【F:models/encoders/DFormerv2.py†L138-L191】【F:sgmixed.txt†L304-L320】
- **Rotary positional encoding (RoPE) ties priors to the queries/keys.** Sine and cosine waves derived from token indices rotate every head's query and key vectors so that spatial order information is baked into the dot products without adding extra tensors.【F:models/encoders/DFormerv2.py†L109-L187】
- **Depth-guided 1D decomposed attention reduces cost.** In early stages, attention is applied along width and height separately with matching 1D geometry masks, cutting the quadratic cost while still nudging the focus toward depth-consistent neighbours.【F:models/encoders/DFormerv2.py†L206-L276】【F:sgmixed.txt†L321-L392】
- **Depthwise convolutional residual encodes local context.** After attention, a depthwise 5×5 convolution (LEPE) adds fine-grained spatial bias before the output projection, reinforcing local structure.【F:models/encoders/DFormerv2.py†L220-L258】

## 2. Geometry prior generation with explicit tensor shapes
For a stage whose token grid has height \(H\) and width \(W\):

1. **Depth alignment.** Given a batch of depth maps \(D_{\text{raw}} \in \mathbb{R}^{B\times1\times H_d\times W_d}\), bilinear interpolation produces \(D \in \mathbb{R}^{B\times1\times H\times W}\) so that each visual token has a matching depth sample.【F:models/encoders/DFormerv2.py†L166-L191】
2. **Depth vectorization.** Flatten \(D\) into \(z \in \mathbb{R}^{B\times HW}\) where \(z_{b,p}\) corresponds to patch index \(p = i\cdot W + j\).【F:models/encoders/DFormerv2.py†L151-L166】
3. **Depth distance per head.** For head \(c\), compute
   \[
   \Delta^{(c)}_{b,p,q} = |z_{b,p} - z_{b,q}|\,\lambda_c,
   \]
   where \(\lambda_c = \log\bigl(1 - 2^{-(\texttt{initial\_value} + \texttt{heads\_range} \cdot c / N_h)}\bigr) < 0\) and the resulting tensor has shape \(B\times N_h\times HW\times HW\).【F:models/encoders/DFormerv2.py†L146-L191】
4. **Spatial Manhattan distance per head.** Enumerate the token coordinates \((i_p,j_p)\). For head \(c\), form
   \[
   M^{(c)}_{p,q} = (|i_p-i_q| + |j_p-j_q|)\,\lambda_c,
   \]
   yielding \(N_h\times HW\times HW\).【F:models/encoders/DFormerv2.py†L155-L191】
5. **Learnable fusion.** Let the trainable scalars be \(w_s = \texttt{weight}[0]\) and \(w_d = \texttt{weight}[1]\). The full attention geometry mask becomes
   \[
   G_{b,c,p,q} = w_s M^{(c)}_{p,q} + w_d \Delta^{(c)}_{b,p,q} \in \mathbb{R}^{B\times N_h\times HW\times HW}.
   \]
   In decomposed blocks, two analogous masks are built: \(G^h \in \mathbb{R}^{B\times N_h\times W\times H\times H}\) for column-wise attention and \(G^w \in \mathbb{R}^{B\times N_h\times H\times W\times W}\) for row-wise attention.【F:models/encoders/DFormerv2.py†L166-L191】【F:models/encoders/DFormerv2.py†L191-L206】
6. **Rotary sinusoid cache.** For each head, create \(\sin,\cos \in \mathbb{R}^{H\times W\times d_k}\) with frequencies defined by the shared \(\texttt{angle}\) vector. These tensors enable RoPE without additional parameters.【F:models/encoders/DFormerv2.py†L172-L191】

The tuple \(((\sin,\cos), G)\) (or \(((\sin,\cos), (G^h, G^w))\) in the decomposed case) is the geometry prior consumed by the attention layers.

## 3. Geometry self-attention with expanded formulas
Let the input features be \(X \in \mathbb{R}^{B\times H\times W\times C}\) and \(N_h\) heads with key dimension \(d_k = C/N_h\).

### 3.1 Full geometry self-attention (fourth stage)
1. **Linear projections.**
   \[
   Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\quad W_Q,W_K \in \mathbb{R}^{C\times C},\ W_V \in \mathbb{R}^{C\times C}.
   \]
   Shapes remain \(B\times H\times W\times C\).【F:models/encoders/DFormerv2.py†L220-L240】
2. **Head reshaping and scaling.** Reshape to \(\tilde{Q},\tilde{K} \in \mathbb{R}^{B\times N_h\times H\times W\times d_k}\), multiply keys by \(d_k^{-1/2}\).【F:models/encoders/DFormerv2.py†L233-L242】
3. **Rotary transform.** Apply RoPE component-wise:
   \[
   \text{RoPE}(\tilde{Q})_{b,c,i,j,2t} = \tilde{Q}_{b,c,i,j,2t}\cos_{i,j,t} - \tilde{Q}_{b,c,i,j,2t+1}\sin_{i,j,t},
   \]
   \[
   \text{RoPE}(\tilde{Q})_{b,c,i,j,2t+1} = \tilde{Q}_{b,c,i,j,2t+1}\cos_{i,j,t} + \tilde{Q}_{b,c,i,j,2t}\sin_{i,j,t},
   \]
   and similarly for \(\tilde{K}\).【F:models/encoders/DFormerv2.py†L109-L187】
4. **Flatten tokens.** Flatten \(H\times W\) so that \(Q_r,K_r \in \mathbb{R}^{B\times N_h\times HW\times d_k}\) and \(V_r \in \mathbb{R}^{B\times N_h\times HW\times d_v}\) with \(d_v=d_k\).【F:models/encoders/DFormerv2.py†L240-L258】
5. **Geometry-biased logits.** Compute logits and inject the mask:
   \[
   L_{b,c,p,q} = Q_{r,b,c,p,:} K_{r,b,c,q,:}^{\top} + G_{b,c,p,q}.
   \]
6. **Softmax with implicit decay.**
   \[
   A_{b,c,p,q} = \frac{\exp(L_{b,c,p,q})}{\sum_{q'} \exp(L_{b,c,p,q'})} = \frac{\exp(Q_{r,b,c,p,:}K_{r,b,c,q,:}^{\top})\, \beta_c^{\text{dist}_{b,c,p,q}}}{\sum_{q'} \exp(Q_{r,b,c,p,:}K_{r,b,c,q',:}^{\top})\, \beta_c^{\text{dist}_{b,c,p,q'}}},
   \]
   where \(\beta_c = e^{\lambda_c}\) and \(\text{dist}\) denotes the fused geometry distance.【F:models/encoders/DFormerv2.py†L244-L258】
7. **Weighted sum and projection.**
   \[
   O_{b,c,p,:} = \sum_q A_{b,c,p,q} V_{r,b,c,q,:},\quad Y = \text{reshape}(O) + \text{LEPE}(V).
   \]
   Finally, \(Y\) passes through \(W_O \in \mathbb{R}^{C\times C}\) to return to \(\mathbb{R}^{B\times H\times W\times C}\).【F:models/encoders/DFormerv2.py†L250-L258】

### 3.2 Decomposed geometry self-attention (stages 1–3)
1. **Shared projections and RoPE.** Steps 1–3 above remain unchanged, delivering \(\tilde{Q}, \tilde{K}, V\).【F:models/encoders/DFormerv2.py†L233-L243】
2. **Width-wise attention.** For every row \(i\), head \(c\), and batch \(b\):
   - Queries/keys: \(Q^w_{b,i,c} \in \mathbb{R}^{W\times d_k}\), \(K^w_{b,i,c} \in \mathbb{R}^{W\times d_k}\).
   - Logits with mask \(G^w_{b,i,c} \in \mathbb{R}^{W\times W}\):
     \(L^w_{b,i,c} = Q^w_{b,i,c} (K^w_{b,i,c})^{\top} + G^w_{b,i,c}\).
   - Softmax over the width dimension and apply to values \(V^w_{b,i,c} \in \mathbb{R}^{W\times d_v}\), producing \(U_{b,i,c} \in \mathbb{R}^{W\times d_v}\).【F:models/encoders/DFormerv2.py†L243-L254】
3. **Height-wise attention.** Treat each column \(j\) with values from the previous step:
   - Queries/keys: \(Q^h_{b,j,c} \in \mathbb{R}^{H\times d_k}\), \(K^h_{b,j,c} \in \mathbb{R}^{H\times d_k}\).
   - Geometry mask \(G^h_{b,j,c} \in \mathbb{R}^{H\times H}\).
   - Apply softmax to \(L^h_{b,j,c} = Q^h_{b,j,c} (K^h_{b,j,c})^{\top} + G^h_{b,j,c}\) and weight the column-wise values from \(U\) to yield \(O_{b,j,c} \in \mathbb{R}^{H\times d_v}\).【F:models/encoders/DFormerv2.py†L254-L258】
4. **Reassembly and output projection.** Rearrange \(O\) back to \(B\times H\times W\times (N_h d_v)\), add the LEPE convolutional bias, and apply the final linear layer exactly as in the full-attention case.【F:models/encoders/DFormerv2.py†L256-L258】

These steps show how every intermediate tensor size and operation is grounded in explicit geometry-aware computations rather than abstract "priors".
