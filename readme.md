# LuGre Friction Model Parameter Tuner

A simple tool in PyQt5 to interactively adjust LuGre friction model parameters and see the results in real-time.

## What it does

Lets you tune 6 friction parameters with sliders while watching how they affect the friction curve. Super useful for fitting the model to experimental data.

## Install

```bash
pip install numpy matplotlib PyQt5 scipy pandas
```

## Run it

```bash
python lugre_interactive.py
```

## Controls

- **Arrow keys**: ←→ to switch parameters, ↑↓ to finetune values
- **F11**: Fullscreen
- **Sliders**: Click and drag to adjust

## LuGre Friction Model

The LuGre model computes friction force using:

```
F = σ₀z + σ₁(dz/dt) + σ₂v
```

Where:
- `z` is the average bristle deflection (internal state)
- `v` is the relative velocity
- `dz/dt = v - z|v|/g(v)`
- `g(v) = (Fc + (Fs - Fc)exp(-(v/vs)²))/σ₀`

From:
> Canudas de Wit, C., Olsson, H., Astrom, K. J., & Lischinsky, P. (1995). A new model for control of systems with friction. IEEE Transactions on automatic control, 40(3), 419-425.

## Parameters

- **sigma_0**: Bristle stiffness
- **sigma_1**: Bristle damping  
- **sigma_2**: Viscous friction
- **F_c**: Coulomb friction
- **F_s**: Static friction
- **v_s**: Stribeck velocity

## Requirements

You need these data files in a `DATAM/` folder:
- `timeM4.npy`
- `stage_posM4.npy` 
- `mobile_speedM4.npy`

Plus the `function.py` module with the data loading function.
