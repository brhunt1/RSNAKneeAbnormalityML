# Roadmap

From August 25, 2026 to the October 22 final submission deadline. Roughly eight working
weeks. Each week ends with something submitted or something measured, never with
"still building".

**Hard dates**
- October 15: entry and team merger deadline. Accept the rules well before this.
- October 22: final submission deadline.
- November 5: winner requirements deadline, only relevant if you place.

---

## Week 1 (Aug 25 to Aug 31) Get on the board

Goal: a valid submission exists and you understand the data.

- [ ] Join the competition on Kaggle and accept the rules. Do this first, today.
- [ ] Read the rules page properly. Note the external data policy and any runtime cap.
- [ ] Download or attach the data. Check total size before downloading locally.
- [ ] Submit `sample_submission.csv` unchanged. Confirm the scoring works.
- [ ] EDA notebook: label prevalence, label co-occurrence, series counts per study,
      slice counts, plane and sequence distribution, how many studies have labels vs
      only a report.
- [ ] Confirm you can decode DICOMs from every transfer syntax present. Install
      `pylibjpeg`, `pylibjpeg-libjpeg`, `pylibjpeg-openjpeg`, `gdcm`. Loop over a
      random 500 files and count failures.
- [ ] Read the discussion forum end to end once.

Deliverable: `notebooks/01-eda.ipynb`, plus notes in `docs/experiments.md`.

## Week 2 (Sep 1 to Sep 7) Validation and preprocessing

Goal: the scaffolding that every later experiment depends on.

- [ ] Build the CV split. Multilabel stratified 5 fold on the labeled subset. Save
      `data/folds.csv`. Never change it after this week.
- [ ] Implement the metric locally and sanity check it against a known case.
- [ ] Preprocessing pipeline: per-series intensity normalization, resize, cache to
      compressed `.npy` or `.npz`. Measure how long a full pass takes.
- [ ] Series selection rule v1: pick one sagittal fluid sensitive, one coronal fat
      suppressed, one axial per study, using `train_series.csv` metadata.
- [ ] Submit a prevalence-only baseline. Note the LB score. This is your floor.

Deliverable: reproducible folds, cached preprocessed data, a metric you trust.

## Week 3 (Sep 8 to Sep 14) First real model

Goal: a trained image model with a CV score.

- [ ] 2.5D approach: stack adjacent slices as channels, feed a pretrained 2D backbone,
      pool across slices to a study level prediction. Start with a small backbone at
      low resolution, one sequence only.
- [ ] Train fold 0 only. Get the loop working before spending compute on five folds.
- [ ] Train all folds. Record per-label AUC.
- [ ] Submit. Compare CV to LB. This comparison is the most valuable number you will
      collect all competition.

Deliverable: first honest CV score, first per-label breakdown.

## Week 4 (Sep 15 to Sep 21) Report mining, the big lever

Goal: labels for the unlabeled majority of the training set.

- [ ] Quantify the opportunity: how many studies have a report but no labels.
- [ ] Language detection across reports. Understand the mix.
- [ ] Approach A, rules: build multilingual keyword and negation patterns per label.
      Validate against the labeled subset, where you know the truth. Measure precision
      and recall per label.
- [ ] Approach B, model: multilingual encoder fine tuned on the labeled subset to
      predict the 12 labels from report text. Validate the same way.
- [ ] Pick the better one, or combine. Generate soft pseudo-labels for the unlabeled
      studies. Keep them soft, do not threshold to 0 or 1.
- [ ] Save as `data/pseudo_labels.csv` with a confidence column.

Reminder: reports do not exist at test time. This is a training signal only.

Deliverable: pseudo-labels with a measured quality number per label.

## Week 5 (Sep 22 to Sep 28) Train on the expanded set

Goal: convert pseudo-labels into leaderboard points.

- [ ] Retrain the Week 3 model on labeled plus pseudo-labeled data. Weight pseudo-labeled
      samples lower, or pretrain on pseudo then fine tune on gold.
- [ ] Compare CV against Week 3. Same folds, same metric. If it does not improve, the
      pseudo-labels are the problem, not the model.
- [ ] Submit. Check the CV to LB relationship still holds.

Deliverable: a measured answer on whether report mining paid off, and by how much.

## Week 6 (Sep 29 to Oct 5) Scale what works

Goal: push the single model as far as it goes.

- [ ] Increase resolution. This usually matters most for small findings like meniscal
      tears and fractures.
- [ ] Add the remaining planes and sequences. Multi-view fusion at the study level.
- [ ] Stronger backbone, longer schedule, sensible augmentation.
- [ ] Attack the worst per-label AUCs specifically. Ask what a radiologist looks at for
      that finding and whether your model can see it.

Deliverable: best single model, with per-label AUC compared to Week 5.

## Week 7 (Oct 6 to Oct 12) Ensemble and efficiency

Goal: squeeze the last points, and decide on the efficiency track.

- [ ] Train diverse variants: different backbones, seeds, input configurations.
- [ ] Blend on out-of-fold predictions. Rank averaging usually works well for AUC.
      Consider per-label blend weights.
- [ ] Test time augmentation, horizontal flip at minimum.
- [ ] Measure inference runtime. Decide whether to target the efficiency award with a
      single lean model alongside the main entry.

**Confirm you have accepted the rules and are formally entered before October 15.**

Deliverable: best blend, and a runtime number.

## Week 8 (Oct 13 to Oct 19) Consolidate

Goal: a submission you would defend.

- [ ] Freeze modeling. No new ideas after this point unless something is broken.
- [ ] Full clean rerun of the winning pipeline from scratch. Confirm reproducibility.
- [ ] Verify the submission notebook runs inside the Kaggle runtime limit with margin.
      Test on the full test set size, not the three example studies.
- [ ] Write the model documentation you would need if you placed.

## Final days (Oct 20 to Oct 22)

- [ ] Select final submissions. Best CV and best blend. Not best public LB.
- [ ] Submit with a full day of margin. Deadlines are UTC and servers get busy.
- [ ] Write a short retrospective in `docs/retrospective.md` regardless of placement.
      This is where the actual learning gets consolidated.

---

## If you fall behind

Cut in this order: ensembling first, then resolution scaling, then extra sequences.
Never cut the CV split or the report mining. A single well trained model on
well pseudo-labeled data beats an ensemble trained on a tenth of the data.
