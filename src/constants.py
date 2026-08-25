"""Shared constants. Import these rather than retyping label names anywhere."""

# Order matters. This is the exact column order the submission file requires.
TARGETS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

N_TARGETS = len(TARGETS)

ID_COL = "StudyInstanceUID"
SERIES_COL = "SeriesInstanceUID"
REPORT_COL = "Report"

PLANES = ["Sagittal", "Coronal", "Axial"]

# Clinical priors: which plane is most informative for each finding.
# Used for series selection, not as a hard rule.
PREFERRED_PLANE = {
    "ACL": "Sagittal",
    "MCL": "Coronal",
    "Medial Meniscus": "Sagittal",
    "Lateral Meniscus": "Sagittal",
    "Medial OA": "Coronal",
    "Lateral OA": "Coronal",
    "PF OA": "Axial",
    "Effusion": "Axial",
    "Synovitis": "Axial",
    "Baker's": "Axial",
    "Contusion": "Coronal",
    "Fracture": "Coronal",
}

N_FOLDS = 5
SEED = 42
