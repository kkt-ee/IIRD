# IIRD: Fast trainable D-dimensional IIR filter layers in TensorFlow

[![PyPI Version](https://img.shields.io/pypi/v/IIRD?label=PyPI&color=gold)](https://pypi.org/project/IIRD/)
[![Python Versions](https://img.shields.io/pypi/pyversions/IIRD)](https://pypi.org/project/IIRD/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-required-darkorange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-deepgreen.svg?style=flat)](https://github.com/kkt-ee/IIRD/LICENSE)

`IIRD` provides trainable 1D, 2D, and 3D IIR filter layers for TensorFlow/Keras.

<!-- Copyright 2025 Kishore Kumar Tarafdar.
Licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE). -->

## Capabilities

- `IIR1D`, `IIR2D`, and `IIR3D` Keras layers.
- Trainable feedforward and feedback coefficients.
- Batched multichannel TensorFlow tensors.
- Frequency-response plotting utility.

## Minimal Example

```python
import tensorflow as tf
from IIRD import IIR1D, IIR2D, IIR3D

x1 = tf.random.normal([1, 64, 1])
y1 = IIR1D(Delays=2, filters=4)(x1)

x2 = tf.random.normal([1, 32, 32, 1])
y2 = IIR2D(Delays=2, filters=4)(x2)

x3 = tf.random.normal([1, 16, 16, 16, 1])
y3 = IIR3D(Delays=1, filters=2)(x3)

print(y1.shape, y2.shape, y3.shape)
```

## Module Commands

```bash
python -m IIRD.IIR1D
python -m IIRD.IIR2D
python -m IIRD.IIR3D
```

## Citation

This software is released for broad research, educational, and engineering use. If this package helps your work, please cite the following paper:

```bibtex
@misc{tarafdar2026interpretablefrugallearningsystems,
      title={Interpretable and Frugal Learning Systems Employing Multiresolution Pyramids and Volterra Kernels},
      author={Kishore Kumar Tarafdar},
      year={2026},
      eprint={2606.15011},
      archivePrefix={arXiv},
      primaryClass={eess.SP},
      url={https://arxiv.org/abs/2606.15011},
}
```

## License

Apache License 2.0. See [`LICENSE`](LICENSE).

* * *

***IIRD (C) 2026 Kishore Kumar Tarafdar, भारत*** 🇮🇳