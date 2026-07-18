import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # there are up to seq_lenght positions and each position should have a d_model long vector embedding
    """
    For seq_length=3 and d_model=4:

pos_vec = np.arange(seq_length).reshape(-1, 1)
np.arange(3) creates [0, 1, 2]
.reshape(-1, 1) converts it into a column vector:
[[0],
 [1],
 [2]]
Shape: (3, 1). -1 means NumPy automatically calculates that dimension.

output = np.zeros((seq_length, d_model))
Creates a matrix filled with zeros:

[[0., 0., 0., 0.],
 [0., 0., 0., 0.],
 [0., 0., 0., 0.]]
Shape: (3, 4). Each row represents a sequence position, and each column represents an embedding dimension. The loop later fills this matrix with sine and cosine values.


    """
    pos_vec = np.arange(seq_length).reshape(-1, 1)
    output = np.zeros((seq_length, d_model)); 
    """
    Both use NumPy’s `array[rows, columns]` indexing.

- `pos_vec[:, 0]`
  - `:` selects **all rows/positions**
  - `0` selects the **first column**
  - For `seq_length=3`, this gives `[0, 1, 2]`

- `output[:, 2 * i]`
  - `:` selects **all rows/positions**
  - `2 * i` selects an **even embedding column**
  - The computed sine values are assigned to that entire column

With `d_model=4`:

```text
i = 0 → output[:, 0]  # sine
        output[:, 1]  # cosine

i = 1 → output[:, 2]  # sine
        output[:, 3]  # cosine
```

So line 11 calculates sine values for every token position and stores them in even-numbered embedding dimensions.
    """
    for i in range(d_model // 2): 
        output[:, 2 * i] = np.sin(pos_vec[:, 0] / (10000.0 ** (2 * i / d_model)))
        output[:, 2 * i + 1] = np.cos(pos_vec[:, 0] / (10000.0 ** (2 * i / d_model)))
    # Your code here
    return output


