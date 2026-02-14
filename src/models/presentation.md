[Title Slide – Pause, smile, wait for silence]

Good [morning/afternoon], everyone.  
I’m Georg Wolnik, together with Lucas Reinken and Koray Tekin, presenting Milestone 2: Ablation Studies and Multi-Task Evaluation of the Swin Transformer.

[Slide: Outline – 5 seconds pause]

Quick outline: we recap Milestone 1, explain Swin’s core design, present our ablations, show from-scratch comparison, and outline dense prediction plans.  
Let’s jump in.

[Slide: Milestone 1 Goal]

In Milestone 1 we reimplemented Swin-Tiny completely from scratch.  
We proved correctness with two checks:  
100% weight transfer from TIMM — no strict=False — and linear probing on CIFAR-100.

[Slide: Swin Transformer Architecture]

Here’s the architecture.  
Swin is a hierarchical vision transformer with four stages — resolution drops from 56×56 to 7×7, channels increase — very CNN-like.

[Slide: How Swin Differs from Standard Vision Transformer (ViT)]

Compared to standard ViT:  
Swin is hierarchical with four stages — ViT is fixed resolution.  
Swin uses local 7×7 window attention — ViT uses global attention.  
This makes Swin linear complexity — ViT is quadratic.  
Swin adds relative position bias inside windows — ViT uses absolute embeddings.

[Slide: The Core of Swin: Shifted Window Attention]

The heart of Swin is shifted window attention.  
Attention happens only in local 7×7 windows — linear complexity.  
Consecutive blocks shift windows by half the size — creating cross-window connections.  
Relative position bias inside windows gives translation invariance.  
Result: efficient, scalable attention with global receptive field.

[Slide: Ablation Studies Results (1/2)]

Now Milestone 2: ablations from scratch on 50k ImageNet subset, 20 epochs.  

First: shifted windows.  
Disabling them drops accuracy from 7.58% to 5.96% — minus 1.62 points.  
Shifted windows are essential for cross-window flow.

Second: relative position bias.  
Removing it drops to 6.36% — minus 1.22 points.  
It’s key for spatial encoding inside windows.

[Slide: Ablation Studies Results (2/2)]

Third: absolute vs relative encoding.  
Surprisingly, absolute embeddings beat baseline — 8.60% vs 7.58%, plus 1.02 points.  
Likely because absolute helps more in short training.

Fourth: window size.  
Default M=7 gives 7.58%.  
M=4 drops to 4.68% — minus 2.9 points.  
M=14 actually improves to 8.02% — plus 0.44 points.  
Larger windows help early training.

[Slide: From-Scratch Model Comparison]

Finally, from-scratch comparison on 100k ImageNet images, 40 epochs.  

ResNet-50 leads with 42.68% — strong convolutional bias.  
Our Swin-Tiny reaches 26.73% — beats ViT-B/16’s 17.28% by 9.45 points.  
Hierarchy and local attention clearly help transformers train from scratch.

Note: 40 epochs is partial training — models aren’t converged.  
The original Swin paper shows Swin outperforms ResNet and ViT after full training.

[Slide: Planned: Dense Prediction Tasks]

Next, we plan dense prediction:  
Object detection with Cascade Mask R-CNN on COCO — expect Swin advantage from hierarchy.  
Semantic segmentation with UPerNet on ADE20K — Swin’s multi-scale features should excel.

[Slide: Conclusion – currently blank]

In summary: ablations confirm shifted windows and relative bias are crucial.  
From-scratch shows ResNet leads early, but Swin beats ViT by a large margin.  
Swin’s design shines for efficient, hierarchical vision modeling.

Thank you!  
Happy to take questions.

[Final slide – Thank you! Questions?]