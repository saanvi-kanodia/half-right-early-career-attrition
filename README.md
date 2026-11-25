# Half Right: Managers and Early-Career Attrition Risk in India

![Manager and early-career employee seeing the same job differently](paper/cover.png)

Survey-based study of how managerial practices relate to early-career attrition risk among Indian professionals.

- **Paper:** [paper/half-right-early-career-attrition.pdf](paper/half-right-early-career-attrition.pdf)
- **Slides:** [paper/presentation.pdf](paper/presentation.pdf)

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Final paper (PDF + DOCX), presentation, and its HTML source (`paper/slides/`) |
| `data/survey_data_anonymized.csv` | Survey responses (N = 38), with timestamps and company names removed |
| `analysis.py` | Single reproducible analysis script |
| `results_clean/` | Output: `report.txt`, factor scores, figures |

## Methods (in `analysis.py`)

- Descriptives and Likert item distributions, split by career stage
- Mann–Whitney U group comparisons with Holm correction
- KMO, Bartlett's test, parallel analysis, PCA of predictor items
- Logistic regression of high attrition risk, evaluated with leave-one-out cross-validation

## Reproduce

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python analysis.py            # writes results_clean/
```

## Data note

Interview transcripts and the raw survey export are not included, to protect respondent privacy.
