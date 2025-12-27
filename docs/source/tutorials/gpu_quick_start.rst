GPU Quick Start Tutorial
========================

This tutorial shows how to verify your GPU setup and run a quick training
test in under 5 minutes.


Prerequisites
-------------

- NVIDIA GPU with CUDA support
- PyTorch with CUDA (verify: ``torch.cuda.is_available()``)
- Ternary VAE installed (``pip install -e .``)


Step 1: Verify GPU
------------------

.. code-block:: bash

    python scripts/quick_train.py

This runs a 5-epoch smoke test and displays:

- GPU name and memory
- Model creation time
- Training progress
- Final metrics

Expected output:

.. code-block:: text

    === Ternary VAE Quick Training ===
    GPU: NVIDIA GeForce RTX 2060 SUPER
    GPU Memory: 7.79 GB

    Training Progress:
    Epoch 1/5: loss=2.3456, recon=1.8765, kl=0.4691
    ...
    Epoch 5/5: loss=1.2345, recon=0.8765, kl=0.3580

    Training completed successfully!


Step 2: Understanding the Output
--------------------------------

The quick test validates:

1. **GPU Detection**: PyTorch sees your GPU
2. **Memory Allocation**: Model fits in VRAM
3. **Forward/Backward**: Gradients compute correctly
4. **Checkpoint Saving**: Model serializes properly


Step 3: Run Full Training
-------------------------

Once verified, use the unified training launcher:

.. code-block:: bash

    # Basic V5.11 training
    python src/train.py --mode v5.11 --epochs 100

    # V5.11.11 with homeostasis
    python src/train.py --mode v5.11.11 --epochs 100

    # With custom parameters
    python src/train.py --mode v5.11 \
        --epochs 200 \
        --batch-size 512 \
        --learning-rate 1e-3 \
        --latent-dim 16


Memory Optimization
-------------------

For GPUs with limited VRAM (4-8GB):

.. code-block:: python

    # Use mixed precision (FP16)
    from torch.cuda.amp import autocast, GradScaler

    scaler = GradScaler()

    with autocast():
        output = model(x)
        loss = criterion(output, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

Or via CLI:

.. code-block:: bash

    python src/train.py --mode v5.11 --mixed-precision


Troubleshooting
---------------

**CUDA out of memory**
    Reduce batch size: ``--batch-size 256``

**Slow training**
    Enable mixed precision: ``--mixed-precision``

**Model not converging**
    Adjust learning rate: ``--learning-rate 5e-4``


Next Steps
----------

- :doc:`full_training` - Complete training guide
- :doc:`hiv_resistance` - HIV-specific analysis
- :doc:`predictors` - Build resistance predictors
