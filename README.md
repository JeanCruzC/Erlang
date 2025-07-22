# Erlang Calculator

This repository contains a basic Erlang calculator with advanced call center analysis tools implemented in `erlang_calculator.py`.

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

## Parameters

The main calculator functions accept the following arguments:

- `traffic` – offered traffic intensity in erlangs.
- `agents`/`trunks` – number of available agents or phone lines.
- `aht` – average handle time in seconds.
- `target` – service level target time in seconds.

See `erlang_examples.py` for scripted workflows demonstrating these parameters.

## Troubleshooting

- Ensure you are using a recent version of Python with `pip` available.
- If package installation fails, verify your network connection or use a local PyPI mirror.
- Delete the `examples` directory if you need a clean regeneration of the sample scripts and rerun `python install.py`.
