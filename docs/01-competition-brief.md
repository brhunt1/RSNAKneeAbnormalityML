# Competition brief

## The setup

RSNA and Kaggle are running a knee MRI challenge. Over 5,000 knee MRI exams from
roughly 16 to 19 international institutions, each study paired with its original
radiology report. Reports appear in several languages. Prize pool is $77,000,
including a separate efficiency track that rewards models that are fast and cheap
to run, not just accurate.

Recognition happens at the RSNA annual meeting, November 29 to December 3, Chicago.

## What you predict

For each study in the test set, twelve probabilities between 0 and 1:

| Label | Meaning |
|---|---|
| ACL | Anterior cruciate ligament injury |
| MCL | Medial collateral ligament injury |
| Medial Meniscus | Medial meniscus tear |
| Lateral Meniscus | Lateral meniscus tear |
| Medial OA | Osteoarthritis, medial tibiofemoral compartment |
| Lateral OA | Osteoarthritis, lateral tibiofemoral compartment |
| PF OA | Patellofemoral osteoarthritis |
| Effusion | Joint effusion, excess fluid |
| Synovitis | Inflammation of the joint lining |
| Baker's | Baker's cyst |
| Contusion | Bone contusion, bone bruise |
| Fracture | Fracture |

## The metric

Macro-averaged AUC ROC. Twelve separate AUCs, then a plain average.

Three consequences worth internalizing:

1. **Rare labels count as much as common ones.** Fracture and Synovitis are probably
   uncommon. Each is worth exactly 1/12 of your score, same as Effusion. A model
   that nails the four easy labels and guesses on the rest caps out low.
2. **AUC is rank based.** Only the ordering of your predictions matters, not their
   absolute values. Calibration is irrelevant to the score. Do not waste time on it.
3. **Threshold tuning is irrelevant.** No decision threshold enters the metric.

## The data, and the thing that makes this competition unusual

Read this paragraph from the description again:

> Only a small subset of training studies carry per-condition labels. We also
> provide the original text of the radiology report from which you may wish to
> derive the labels for the remaining studies.

That is the central problem of this competition. You have thousands of studies with
images but no labels, plus a free text report for each one. Whoever extracts the
best pseudo-labels from those reports gets a much larger effective training set
than everyone else.

The reports are multilingual, which raises the difficulty. A multilingual
transformer or an LLM based extraction pass is the natural approach.

**Important:** reports are not available at test time. The report is a training
signal only. Your final model must work from images alone. This is a common trap.
People build a strong report classifier, score well in local validation, and then
discover it cannot be used for inference.

## Data files

- `train.csv` one row per study: `StudyInstanceUID`, `Report`, 12 binary labels
- `train_series.csv` one row per series, with useful metadata:
  - `Fluid_Sensitive` (T2, PD, STIR and similar)
  - `Fat_Suppression`
  - `Anatomical_Plane` (Sagittal, Coronal, Axial)
- `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`
- `test.csv`, `test_series.csv`, `test_series/` same layout, roughly 1,300 test studies
- `sample_submission.csv` all values 0.5

Series typically hold 20 to 45 slices, median 30, with a long tail into the hundreds.

## DICOM gotchas

The description flags these explicitly, which means they will bite people:

- Mixed transfer syntaxes: uncompressed Explicit VR Little Endian, JPEG Lossless,
  JPEG 2000, Implicit VR Little Endian. You need `pylibjpeg` and `gdcm` installed
  or a chunk of the data silently fails to decode.
- Intensities, orientations and resolutions vary across series and studies.
  Per-series intensity normalization is not optional.
- Only 86 allowlisted metadata tags survive. Do not build logic around a tag
  without checking it is actually present.

## Distribution shift

> the prevalence of abnormalities is not guaranteed to be the same across the
> training, public leaderboard, and final evaluation datasets

Because the metric is AUC, prevalence shift hurts less than it would with a
threshold based metric. But it does mean the public leaderboard is a noisy guide.
Trust a well built local cross validation over the public LB.

## Which sequences matter for which finding

Rough clinical priors, useful for deciding what to feed the model:

| Finding | Best plane | Sequence |
|---|---|---|
| ACL | Sagittal | Fluid sensitive, often fat suppressed |
| MCL | Coronal | Fluid sensitive, fat suppressed |
| Medial / Lateral Meniscus | Sagittal, confirm coronal | Proton density |
| Medial / Lateral OA | Coronal | PD or T2 |
| PF OA | Axial | PD or T2 |
| Effusion | Any, axial reads well | Fluid sensitive |
| Synovitis | Axial | Fluid sensitive, fat suppressed |
| Baker's | Axial | Fluid sensitive |
| Contusion | Coronal and sagittal | Fat suppressed fluid sensitive (STIR, T2 FS) |
| Fracture | Coronal and sagittal | Fat suppressed fluid sensitive |

You do not need to hard code this, but it explains why `Anatomical_Plane`,
`Fluid_Sensitive` and `Fat_Suppression` in `train_series.csv` are worth using as
model inputs or as a series selection rule.

## Efficiency track

A separate award exists for models that are accurate and cheap. If a giant ensemble
is out of reach for you on time or compute, a single efficient model is still a
legitimate target. Keep an eye on runtime from the start rather than trying to
shrink a bloated model in the last week.
