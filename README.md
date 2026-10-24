# IDEAS seizure recurrence survival analysis

Analysis code for the manuscript **“Temporal pattern and multivariable predictors of seizure recurrence after resective epilepsy surgery: a discrete-time survival analysis of the open IDEAS cohort.”**

The workflow reproduces the longitudinal outcome construction, Kaplan–Meier and log-rank analyses, discrete-time Cox models, imaging-augmented nested models, sensitivity analyses, bootstrap internal validation, and the two publication figures.

## Data

The input data are not redistributed in this repository. They are openly available from the IDEAS project:

- OpenNeuro dataset: [ds005602](https://doi.org/10.18112/openneuro.ds005602.v1.0.0)
- IDEAS data portal and derived tables: [cnnp-lab.com/ideas-data](https://cnnp-lab.com/ideas-data)

Place the downloaded tables in the following structure, relative to the repository root:

```text
data/
├── tables_metadata/
│   └── Metadata_Release_Anon.csv
├── zscores/
│   ├── thick_zscores_patients.csv
│   └── vol_zscores_patients.csv
└── resected_table/
    └── table_resected.csv
```

The scripts use only the released anonymous tabular data. No MRI images are required for this workflow.

## Environment

Python 3.11 or later is recommended. Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`; on macOS/Linux, use `source .venv/bin/activate`.

## Reproduction

Run the scripts from the repository root in this order:

```bash
python analysis/longitudinal_analysis.py
python analysis/enhanced_cox.py
python analysis/revise_analyses.py
python figures/optimize_figures.py
```

1. **`analysis/longitudinal_analysis.py`** constructs the person-period dataset, yearly hazards, Kaplan–Meier curves, log-rank tests, clinical discrete-time Cox model, and GEE sensitivity analysis.
2. **`analysis/enhanced_cox.py`** adds imaging aggregates and resection extent, fits nested M0–M3 models, evaluates the resection-by-time interaction, and runs the hippocampal-sclerosis subgroup analysis.
3. **`analysis/revise_analyses.py`** reproduces endpoint-threshold, missingness, proportional-hazards, FDR, robust/cloglog, temporal-only, regional collective, focal-index, and bootstrap analyses. The bootstrap uses 150 resamples and random seed 42.
4. **`figures/optimize_figures.py`** checks the final model against its saved result table and exports Figure 1 and Figure 2 as PNG, PDF, SVG, and 600-dpi TIFF files.

Default outputs are written to:

```text
outputs/
├── longitudinal/
│   ├── person_period_long.csv
│   ├── enhanced_cox_results.csv
│   ├── enhanced_model_comparison.csv
│   ├── resect_time_hr.csv
│   ├── hs_subgroup_comparison.csv
│   └── revision/revision_results.json
└── figures/
    ├── Figure_1.*
    └── Figure_2.*
```

## Custom paths

To use directories outside the repository, set these environment variables before running the workflow:

- `IDEAS_DATA_DIR`: parent directory containing `tables_metadata`, `zscores`, and `resected_table`.
- `IDEAS_OUTPUT_DIR`: directory for generated result tables and revision outputs.

Generated figures are always written to `outputs/figures` under the repository root.

## Repository contents

```text
analysis/
├── longitudinal_analysis.py
├── enhanced_cox.py
└── revise_analyses.py
figures/
└── optimize_figures.py
requirements.txt
```

Data, generated outputs, local virtual environments, and submission documents are intentionally excluded.
