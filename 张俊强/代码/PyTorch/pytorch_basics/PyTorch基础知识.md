### 导入

```python
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as TF
```

### 自动求导

- 样例一

  ```python
  # Create Tensor
  x = torch.tensor(1., requires_grad=True)
  w = torch.tensor(2., requires_grad=True)
  b = torch.tensor(3., requires_grad=True)
  
  # Build a computational graph
  y = w * x + b # y = 2 * x + 3
  
  # Compute Gradients
  y.backward()
  
  # get value
  print(x.grad)
  print(w.grad)
  print(b.grad)
  ```

  

- 样例二

  ```python
  # Create tensor of shape(10, 3) as (10, 2)
  x = torch.randn(10, 3)
  y = torch.randn(10, 2)
  
  linear = nn.Linear(3, 2)
  print('w: ', linear.weight)
  print('b: ', linear.bias)
  
  # Build loss function and optimizer
  criterion = nn.MSELoss()
  optimizer = torch.optim.SGD(linear.parameters(), lr=0.01)
  
  ```

  