# How to be an effective Kaggle competitor

Written for someone entering their first serious competition. Most of this is not
about machine learning. It is about process.

## The one habit that separates top competitors from everyone else

**A trustworthy local validation score.**

Kaggle gives you a public leaderboard, but it is computed on a slice of the test set
and it is a trap. Chasing it is called leaderboard overfitting and it is the single
most common way beginners waste a competition. Here it matters even more, because the
organizers warned that prevalence differs between train, public LB, and final scoring.

What you build instead:

1. A fixed cross validation split, decided once, saved to disk, never changed.
2. Every experiment reports a CV score computed the same way.
3. You keep every out-of-fold prediction file.
4. You only accept a change if CV improves.

Track the relationship between your CV and the public LB. If they move together, you
can trust CV. If they diverge, trust CV.

For this competition, use **multilabel stratified K-fold**, grouped so that no study
appears in two folds. The `iterative-stratification` package does the multilabel part.
Rare labels like Fracture need stratification or some folds will have almost no
positives and your per-label AUC will be pure noise.

## Build the dumbest possible end-to-end thing first

Before any modeling, produce a valid submission file. All 0.5. Submit it. Confirm the
pipeline works, confirm you understand the format, confirm the notebook runs inside
whatever runtime limit Kaggle imposes.

Then make it slightly less dumb. Predict each label's training prevalence. Then train a
single small model on a single sequence at low resolution. Each step should be a working
end-to-end run. Never spend three weeks building something elaborate that has never
produced a submission.

## Read the discussion forum every day

This is not optional and it is not cheating. On a competition like this, public notebooks
and forum posts will surface:

- Working DICOM decoding for the awkward transfer syntaxes
- Which series selection heuristics work
- Report parsing approaches
- Data leaks or quirks in the dataset
- Whether the public LB is trustworthy

The gap between someone who reads the forum and someone who does not is usually larger
than the gap between two modeling approaches. Budget 20 minutes a day.

## Keep an experiment log

One row per run: config, CV score, per-label AUC, LB score if submitted, and a one
sentence note on what you changed and what you learned. `docs/experiments.md` in this
repo is for that. Without it you will re-run the same failed idea in week six.

Log per-label AUC, not just the mean. The mean hides that your Synovitis AUC is 0.52
while ACL is 0.92, and that Synovitis is where the remaining points live.

## Where the points actually are, in order

For this competition specifically:

1. **Pseudo-labels from reports.** Most studies lack labels. Extracting good labels from
   the free text multiplies your training set. This is the biggest single lever and it is
   the reason the competition exists in this form.
2. **Sensible series selection and 2.5D input.** Feeding the right sequences and planes
   beats feeding everything at random.
3. **A strong pretrained backbone at adequate resolution.** Knee findings are small.
   Resolution matters more than model size.
4. **Ensembling and test time augmentation.** Reliable but small gains, and they cost you
   in the efficiency track. Do this last.

Notice that hyperparameter tuning is not on the list. It almost never is.

## Time and compute reality

Kaggle gives you a weekly GPU quota. It runs out. Plan around it:

- Do data exploration and report parsing on CPU. Save the GPU for training.
- Cache preprocessed images as compressed arrays. Decoding DICOM every epoch will
  dominate your runtime and burn quota on I/O.
- Start at small resolution and short schedules. Scale up only once the pipeline is proven.

## Submission discipline

- Submit early and often. You get a limited number per day, so use them on things you
  actually want to learn from.
- Select your final submissions deliberately. Usually: your best CV model, and your best
  blend. Not your best public LB score.
- Watch the entry deadline. October 15 is when you must have accepted the rules and joined.
  Missing that means you cannot compete at all, no matter how good your model is.

## Reading the rules

Skim the competition rules page once, properly. Things that catch people out:

- External data policy and whether pretrained weights are allowed and which ones
- Whether submission is via notebook with a runtime cap
- Team size limits and merger deadline
- Winner obligations, including a code and documentation submission by November 5

## What to ignore

- Calibration. The metric is rank based.
- Class weights and focal loss tuned for imbalance. AUC does not care about thresholds.
  Plain BCE with logits is usually fine.
- Exotic architectures. A well trained standard backbone with good data beats a clever
  model with bad data, essentially always.
