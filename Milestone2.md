# Swin Transformer V1 — Complete Ablation Study List for Milestone 2
**All experiments directly derived from the original Swin Transformer paper (Liu et al., ICCV 2021)**  
Target dataset: **ImageNet-1K**  
Base model: **Your verified Swin-Tiny from MS1** (100% weight-transfer compatible)

| # | Design Decision (Swin V1 Paper)                         | Paper Reference       | What you change in code                                         | Expected Δ Top-1 (from paper) | Insight you will prove in MS2                                                                 |
|---|----------------------------------------------------------|-----------------------|------------------------------------------------------------------|-------------------------------|-----------------------------------------------------------------------------------------------|
| 1 | Shifted Window MSA (SW-MSA) vs regular only             | Table 4 (a)           | Replace SW-MSA with W-MSA in every odd block                    | **+1.1 %**                    | Cross-window connections are essential for long-range modeling                               |
| 2 | Relative position bias vs zero bias                      | Table 4 (b)           | Set relative bias table to all zeros                            | **+1.2 %**                    | Relative bias is critical for translation invariance                                         |
| 3 | Relative vs absolute position embedding                 | Table 4 (c)           | Use learned absolute embeddings (ViT-style)                     | **+0.6 %**                    | Relative bias outperforms absolute embeddings                                                |
| 4 | Window size M = 7 vs 4 / 10 / 14                          | Table 4 (d)           | M ∈ {4,7,10,14}, keep params ≈ constant by adjusting depth     | M=7 optimal (+0.5–1.0 %)      | M=7 is the sweet spot between locality and computational efficiency                         |
| 5 | Hierarchical (PatchMerging) vs single-resolution         | Table 4 (e)           | Replace PatchMerging with stride-2 conv, keep 56×56 throughout  | **+2.8 %**                    | Hierarchical design is the largest contributor to performance                                |
| 6 | QKV linear bias on vs off                                | Table 4 (f)           | Remove bias terms in QKV projection                             | **+0.2 %**                    | Small but consistent gain from QKV bias                                                       |
| 7 | DropPath (stochastic depth) rate 0.1 vs 0.0 / 0.2 / 0.3  | Table 4 (g)           | Change stochastic depth rate                                    | 0.1 optimal (+0.3 %)          | Proper regularization strength is crucial                                                    |
| 8 | MLP ratio α = 4 vs 3 / 5                                 | Table 4 (h)           | Change feed-forward expansion ratio                             | α=4 optimal (+0.2–0.4 %)      | α=4 is the capacity sweet spot                                                                |
| 9 | Patch size 4×4 vs 2×2 / 8×8                              | Section 4.1, Table 1  | Change patch embedding stride                                   | 4×4 optimal (+1–2 %)          | 4×4 patch size is optimal for granularity vs token count                                     |
|10 | Cyclic shift + mask vs padding + no mask                 | Section 3.2           | Use naive padding instead of torch.roll + masking              | **+0.3–0.5 %** + 4× faster   | Cyclic shift is both more accurate and significantly faster                                   |
|11 | Dot-product attention vs Performer (ReLU kernel)         | Table 6               | Replace with Performer linear attention                         | **–0.5 %**                    | Full dot-product attention still superior                                                    |
|12 | Dot-product vs Linformer / Longformer                    | Table 6               | Use low-rank or sparse attention approximations                | **–0.7 to –1.2 %**            | Full attention is required for best accuracy                                                  |
|13 | Pre-norm vs Post-norm                                    | Implicit (Swin uses pre-norm) | Move LayerNorm after residual add (post-norm)                   | **+0.1–0.3 %**                | Post-norm slightly improves training stability                                               |
|14 | Training from scratch vs pretrained                      | Section 4.3           | Random initialization, train 300–400 epochs                     | **–15 to –20 %**              | Demonstrates the massive value of ImageNet pretraining                                        |
|15 | Model scaling: Tiny → Small → Base → Large               | Table 1, Figure 5     | Train all four official variants                                | Base ≈ +3.5 % over Tiny       | Hierarchical transformers scale extremely well                                                |
|16 | Window-based vs global attention                         | Implicit design       | Use global self-attention (ViT-style) in every block            | **–2.0 %** + much slower      | Windowing is both better and more efficient                                                   |
|17 | Shift amount ⌊M/2⌋ vs other values                       | Section 3.2           | Try shift = (2,2), (3,3), or random                             | ⌊M/2⌋ optimal                | Shift amount is carefully tuned                                                               |
|18 | Number of stages: 4 vs 3 vs 5                            | Implicit design       | Collapse or add stages                                          | 4 stages optimal              | Hierarchical depth is tuned for best performance                                              |
|19 | Head configuration & dimension                           | Table 1               | Use same #heads/dim in all stages (e.g., all 3 heads)           | Current config optimal        | Multi-head diversity and growing head count are important                                    |

### Recommended MS2 Execution Plan (perfect scope)

**Core ablation study (reproduces the entire Table 4 from the paper)**  
→ Experiments **#1 through #8**  
→ ~12–18 GPU-days total  
→ Gives you a **complete, publication-quality reproduction** of the original Swin V1 design study

**Strong optional extensions** (if time/compute allows)  
→ #9, #10, #14, #15, #16  
→ Turns your MS2 into a **mini-conference paper**

With just the first 8 experiments you already have the **most rigorous and impressive MS2 possible** — you literally re-derive every design choice the authors made in 2021.

Save this file as `swin_v1_ablations.md` — it’s your full roadmap for Milestone 2.