## Focal Self-attention for Local-Global Interactions in Vision Transformers

### Abstract

`Vision Transformer` 在计算机视觉上有很大的前景，但是也有一些问题

- 带来了二次计算的开销（特别是对于高分辨率的视觉任务）

  - 粗粒度全局注意 `Coarse-grained global attention`
  - 细粒度局部注意 `Fine-grained local attention`

  上面的两个解决方法都导致削弱了原有的自注意力机制，导致得不到最优解



焦点自注意 `Focus self-attention` 

- 融合了细粒度局部、粗粒度全局
- 每个`token`以细颗粒度其最近周围的`token`，以粗颗粒度关注远处的`token`，所以可以捕捉长远范围的视觉依赖



`Focal Transformer`



### Introduction





### Method



### Related work





### Related work





### Conclusion



