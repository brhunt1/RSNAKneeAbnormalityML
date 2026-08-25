# Notebooks

Exploration and Kaggle submission notebooks.

Keep them thin. Anything reusable belongs in `src/` where it can be tested and
shared between notebooks. A notebook that grows its own copy of the preprocessing
pipeline is how a project drifts out of sync with itself.

Planned:

- `01-eda.ipynb` label prevalence, co-occurrence, series and slice distributions
- `02-dicom-audit.ipynb` decode every transfer syntax, count failures
- `03-folds.ipynb` build and inspect the CV split
- `04-baseline.ipynb` first 2.5D image model
- `05-report-mining.ipynb` pseudo-labels from radiology reports
- `99-submit.ipynb` the Kaggle inference notebook
