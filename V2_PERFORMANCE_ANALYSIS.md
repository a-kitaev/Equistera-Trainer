# RTMPose V2 Performance Analysis

## Executive Summary

**CRITICAL FINDING**: RTMPose V2 with text fusion and diffusion refinement performs **significantly worse** than the base model:

- **Base Model Best**: 70.71% AP (epoch 240)
- **V2 Model Best**: 62.51% AP (epoch 280)
- **Performance Gap**: **-8.2 percentage points (-11.6% relative degradation)**

## Training Comparison

### Base Model (rtmpose_m)
- Training Date: October 3, 2025
- Total Epochs: 250+
- Best AP: **0.7071 (70.71%)** at epoch 240
- Final AP (epoch 250): 0.7085 (70.85%)
- AP@0.5: 0.891584 (89.16%)
- AP@0.75: 0.797030 (79.70%)

### V2 Model (rtmpose_m_ap10k_v2)
- Training Date: October 14, 2025
- Total Epochs: 300
- Best AP: **0.6251 (62.51%)** at epoch 280
- Final AP (epoch 300): 0.6148 (61.48%)
- AP@0.5: 0.826076 (82.61%)
- AP@0.75: 0.670721 (67.07%)

## V2 Learning Curve Analysis

### Early Phase (Epochs 10-100)
- Epoch 10: AP 0.0000 (initialization issue or delayed learning)
- Epoch 20: AP 0.0805
- Epoch 50: AP 0.3846
- Epoch 100: AP 0.5002

**Observation**: Slow initial learning, suggests text fusion may be hindering gradient flow.

### Middle Phase (Epochs 100-200)
- Epoch 120: AP 0.5042
- Epoch 150: AP 0.5416
- Epoch 160: AP 0.5743 (sudden jump)
- Epoch 190: AP 0.5836

**Observation**: Steady improvement but already below base model plateau (~70%).

### Late Phase (Epochs 200-300)
- Epoch 220: AP 0.5853
- Epoch 240: AP 0.5991
- Epoch 260: AP 0.6081
- Epoch 280: AP 0.6251 (**BEST**)
- Epoch 300: AP 0.6148 (slight decline)

**Observation**: Model reaches plateau around 62%, with slight overfitting after epoch 280.

## Detailed Performance Degradation

| Metric | Base | V2 | Degradation |
|--------|------|-----|-------------|
| AP | 70.71% | 62.51% | **-8.2 pp** |
| AP@0.5 | 89.16% | 82.61% | **-6.55 pp** |
| AP@0.75 | 79.70% | 67.07% | **-12.63 pp** |
| AR | 72.86% | 75.90% | +3.04 pp ✓ |

**Key Insight**: V2 shows **worse precision** (AP) but **better recall** (AR). This suggests the model is making more predictions but with lower confidence/accuracy.

## Root Cause Hypotheses

### 1. FiLM Modulation Corrupting Features ⚠️ HIGH LIKELIHOOD
**Evidence**:
- Epoch 10 had AP 0.0000 (complete failure initially)
- Learning curve is significantly slower than base
- AP@0.75 (high threshold) degraded by 12.63pp - suggests coordinate accuracy issues

**Hypothesis**: Random initialization of FiLM gamma/beta parameters is destroying visual features during early training. By the time the network learns to compensate, it's trapped in a suboptimal solution.

**Test**: 
```python
# Check FiLM parameter initialization in logs
grep "neck.global_mlp" v2_full_training.log
```

### 2. Text Embeddings Providing Misleading Information ⚠️ MEDIUM LIKELIHOOD
**Evidence**:
- MiniLM-L6-v2 is a general-purpose sentence encoder, not trained on anatomical/equine data
- Text descriptions may not align with visual features
- Model may be over-relying on text instead of learning from images

**Hypothesis**: Generic text embeddings (e.g., "The horse's nose is at the front of the face") don't provide useful spatial priors for pose estimation.

**Test**: Run ablation without text (neck pass-through) to isolate impact.

### 3. Learning Rate Mismatch ⚠️ MEDIUM LIKELIHOOD
**Evidence**:
- Neck starts from random initialization while backbone is pretrained
- No separate learning rate schedule for neck components
- Slow convergence suggests optimization issues

**Hypothesis**: Neck needs higher initial learning rate or warmup to catch up with pretrained backbone.

**Test**: Train with separate LR for neck (e.g., 10x backbone LR) or freeze backbone for first 50 epochs.

### 4. Diffusion Refiner Overhead (UNLIKELY)
**Evidence**:
- Refiner only active during inference, not training
- Training loss should be unaffected
- Validation uses inference mode with refiner

**Hypothesis**: UNLIKELY to be the cause since refiner should only improve, not degrade performance.

### 5. Batch Size Reduction Side Effect (UNLIKELY)
**Evidence**:
- Base: batch size 32
- V2: batch size 16 (due to memory overhead)
- Smaller batch = noisier gradients

**Hypothesis**: UNLIKELY - batch size 16 should still be sufficient for stable training. Base model could achieve similar performance even with smaller batches.

## Architectural Issues

### FiLM Implementation
```python
# Current implementation in text_fusion_neck.py
film_params = self.global_mlp(global_text)  # [B, C*2]
gamma = film_params[:, :C].unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
beta = film_params[:, C:].unsqueeze(-1).unsqueeze(-1)   # [B, C, 1, 1]
modulated_feats = gamma * visual_feats + beta
```

**Problems**:
1. No residual connection - if gamma ≈ 0, visual features are destroyed
2. No regularization on gamma/beta values
3. MLP initialized randomly, may output extreme values initially
4. Entire spatial feature map modulated uniformly (no spatial attention)

### Diffusion Refiner Integration
- Refiner bypassed during training (correct)
- Active during validation (may be causing issues?)
- Need to verify if refiner is actually helping or hurting at inference time

## Comparison with Prior Work

### Original RTMPose Paper
- Base RTMPose: ~70% AP on COCO-WholeBody
- SimCC head: 2-3% improvement over heatmap-based methods
- No text fusion or diffusion refinement

### FiLM in Pose Estimation
- FiLM widely used in VQA, not common in pose estimation
- Most multimodal pose work uses attention mechanisms, not FiLM
- Text-to-pose papers (PoseFix, etc.) use text as supervision, not conditioning

**Insight**: We may be applying FiLM incorrectly for pose estimation. The domain mismatch suggests we need a different fusion strategy.

## Next Steps: Diagnosis Experiments

### Experiment 1: Ablation - Remove Text Fusion
**Goal**: Isolate impact of FiLM modulation vs diffusion refiner

**Method**:
```python
# Modify text_fusion_neck.py forward() to pass through
def forward(self, visual_feats, data_samples=None):
    return visual_feats  # No FiLM, just pass through
```

**Expected**: If AP improves back to ~70%, FiLM is the problem.

### Experiment 2: FiLM Initialization Fix
**Goal**: Prevent FiLM from destroying features during early training

**Method**:
```python
# Initialize gamma to ones, beta to zeros (identity transform)
def init_weights(self):
    nn.init.zeros_(self.global_mlp[0].weight)
    nn.init.ones_(self.global_mlp[0].bias[:self.in_channels])  # gamma = 1
    nn.init.zeros_(self.global_mlp[0].bias[self.in_channels:])  # beta = 0
```

**Expected**: Learning curve should start higher (AP > 0 at epoch 10) and converge faster.

### Experiment 3: Residual FiLM
**Goal**: Allow visual features to bypass FiLM if text is unhelpful

**Method**:
```python
# Add residual connection
modulated_feats = gamma * visual_feats + beta
output_feats = alpha * modulated_feats + (1 - alpha) * visual_feats
# where alpha is a learnable gating parameter initialized to 0.1
```

**Expected**: Network can learn to ignore text if it's not useful.

### Experiment 4: Better Text Embeddings
**Goal**: Use anatomically-aware text encoder

**Method**:
- Try BioBERT or PubMedBERT (medical domain)
- Use CLIP vision-language embeddings (visual grounding)
- Generate more detailed keypoint descriptions with spatial relationships

**Expected**: If text is the issue, better embeddings should improve AP.

### Experiment 5: Separate Learning Rate for Neck
**Goal**: Accelerate neck convergence

**Method**:
```python
param_groups = [
    {'params': model.backbone.parameters(), 'lr': 0.004},
    {'params': model.neck.parameters(), 'lr': 0.04},  # 10x higher
    {'params': model.head.parameters(), 'lr': 0.004},
]
optimizer = Adam(param_groups)
```

**Expected**: Faster convergence in early epochs, potentially higher final AP.

### Experiment 6: Ablation - Remove Diffusion Refiner
**Goal**: Verify refiner isn't causing validation issues

**Method**: Set `use_refiner=False` in config during validation

**Expected**: Minimal impact since refiner should only improve results.

## Recommended Action Plan

### Phase 1: Quick Wins (1-2 hours)
1. **Fix FiLM initialization** (Experiment 2) - highest likelihood fix
2. **Add residual connection** (Experiment 3) - safety net for bad text
3. **Restart training with fixes** - should see immediate improvement

### Phase 2: Deeper Investigation (3-5 hours)
4. **Run text ablation** (Experiment 1) - quantify FiLM impact
5. **Test diffusion ablation** (Experiment 6) - rule out refiner issues
6. **Analyze FiLM parameters** - visualize learned gamma/beta values

### Phase 3: Alternative Approaches (if needed)
7. **Try better text embeddings** (Experiment 4)
8. **Implement attention-based fusion** instead of FiLM
9. **Use visual-language models** (CLIP) for joint embeddings

## Expected Outcomes

### Best Case (FiLM initialization fix works)
- Training starts with AP > 30% at epoch 10
- Converges faster than base (leveraging text priors)
- Final AP: **75-80%** (5-10pp improvement over base)

### Medium Case (Residual FiLM helps)
- Training stabilizes, no more epoch 10 = 0.0 AP
- Final AP: **70-72%** (matches base, text doesn't hurt)

### Worst Case (Text fundamentally incompatible)
- Remove FiLM entirely, keep diffusion refiner
- Final AP: **70-71%** (match base, refiner provides 1-2pp improvement)

## Conclusion

The V2 architecture with text fusion **failed to improve** over the base model, achieving only 62.51% AP vs 70.71% baseline. The primary issue is likely the **FiLM modulation destroying visual features** during early training due to poor initialization.

**Immediate action**: Implement FiLM initialization fix and residual connections, then restart training. This should recover baseline performance at minimum, with potential for improvement if text embeddings are useful.

**Long-term**: Consider switching from FiLM to attention-based fusion, or abandon text fusion entirely in favor of simpler architectural improvements (better backbone, data augmentation, ensemble methods).
