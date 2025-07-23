# Erlang Calculator

This repository contains a basic Erlang calculator with advanced call center analysis tools implemented in `erlang_calculator.py`.

## Class hierarchy

The library exposes three small classes that build on each other:

1. **`X`** – static Erlang formulas (`erlang_b`, `erlang_c`).
2. **`CHAT`** – call-center metrics that depend on `X` (`service_level`, `asa`, `required_agents`).
3. **`BL`** – advanced analysis utilities that use `CHAT` (`sensitivity`, `monte_carlo`).

### Parameters and valid ranges

All functions share a common set of core parameters:

- `traffic` – offered traffic intensity in erlangs, `traffic >= 0`.
- `agents`/`trunks` – number of agents or phone lines, integer `> 0`.
- `aht` – average handle time in seconds, `aht > 0`.
- `target` – service level target time in seconds, `target >= 0`.
- `iters` – iterations for Monte Carlo (BL), integer `> 0`.

## Installation

Use `install.py` to install required dependencies and generate example scripts.

```bash
python install.py
```

The script installs the packages listed in `requirements.txt` (`scipy`, `numpy`, `pandas`, and `matplotlib`) and creates an `examples` directory with a simple usage script.

## Running the Streamlit App

After installing the requirements you can run the Streamlit interface:

```bash
streamlit run erlang_calculator.py
```

The app computes Erlang B blocking probability, Erlang C waiting probability, service level, and average speed of answer. It also provides sensitivity analysis, Monte Carlo simulation, and staffing optimisation tools.

## Examples

`install.py` creates an `examples` directory with scripts that show how the library is used. Execute the script below to generate it:

```bash
python install.py
```

`examples/basic_usage.py` demonstrates a **basic** workflow using functions from `X` and `CHAT`:

```bash
python examples/basic_usage.py
```

`erlang_examples.py` contains further **chat** and **blending** examples:

```bash
python erlang_examples.py
```

The first part of that script runs core metrics (chat use case), followed by sensitivity analysis and Monte Carlo simulation (blending use case with the `BL` class).

## Troubleshooting

- Ensure you are using a recent version of Python with `pip` available.
- If package installation fails, verify your network connection or use a local PyPI mirror.
- Delete the `examples` directory if you need a clean regeneration of the sample scripts and rerun `python install.py`.
