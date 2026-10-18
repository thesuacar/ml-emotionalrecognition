# Assignment 2: Detecting Emotions with ML Algorithms — LaTeX report

This project is a compilable LaTeX reporting template following the provided assignment instructions and the supplied LaTeX project HOW-TO. Its text is a checklist/outline, **not experimental findings**.

## Structure and build

- `main/main.tex` is the primary source. Each report section lives in `main/sections/`.
- `main/formatting.tex` defines document style and placeholder/figure helpers.
- All actual image files belong in `main/figures/`.
- From this project root, run `make` to compile. Requires LaTeX with `latexmk`, `pdflatex`, `biber`, and `biblatex-apa`.
- The compiled PDF lands at the **project root**, named `Assignment_2_Detecting_Emotions_with_ML_Algorithms.pdf`.
- The temporary `.build/` directory is ignored by Git.

## Zotero bibliography — manually created, NOT supplied

1. Add the FER2013 and KDEF publications from the course assignment to Zotero and verify metadata.
2. Export the selected references as **Better BibTeX** or BibLaTeX/BibTeX compatible `.bib` (depending on your Zotero setup).
3. Save the export as **`main/references.bib`**. This file is intentionally **not generated** in this project.
4. The source already contains `\IfFileExists{references.bib}{\addbibresource{references.bib}}{}` and an APA bibliography command. Insert the proper `\parencite{your-zotero-key}` commands where appropriate.
5. Run `make` again, confirm references appear, and review APA formatting.

## Rubric-driven writing checklist

| Section | Assignment component | Weight | Essential evidence to add |
| --- | --- | ---: | --- |
| `03_fer2013_baseline.tex` | FER2013 baseline | 20% | Logistic Regression, SVM, Random Forest × HOG, landmarks, PCA; accuracy, balanced accuracy, macro-F1; sample errors; selected model confusion matrices |
| `04_kdef_study.tex` | KDEF study | 15% | Frontal faces only; subject-independent 80/20; comparison after retraining the FER-best features; emotion-wise errors |
| `05_cross_dataset_generalization.tex` | Generalization | 20% | **Both** FER→KDEF and KDEF→FER; FER-compatible preprocessing; domain shift, degradation and feature robustness |
| `06_explainable_fuzzy_classifier.tex` | Fuzzy classifier | 10% | 3–10 interpretable geometry features on KDEF; membership functions, rules, performance, plausibility |
| `07_creative_challenge.tex` | Creative challenge | 25% | At least **two** justified improvements of FER2013; fair ablation/comparison, evidence and limitations |
| `08_realtime_demo.tex` | Real-time webcam demo | 10% | How best model is integrated, observable performance, practical problems and fine-tuning |

Also complete `01_introduction.tex`, `02_data_and_protocol.tex`, `09_discussion_conclusion.tex` and the mandatory `10_ai_usage_statement.tex`.

### Data handling and validity

- FER2013: train = `Training`, validation = `PublicTest`, final test = `PrivateTest`. Never use PrivateTest for tuning or model selection.
- KDEF: **frontal (`S`) images only**, subject-level 80/20 split; the same subject must never appear in both subsets.
- Convert KDEF to the FER format (face crop with enough context, grayscale, **48×48**, FER-equivalent normalization) for all cross-dataset comparisons.
- Train or fit feature scalers, PCA components, preprocessing statistics, and model hyperparameters **only on the eligible training split**.
- Maintain a class mapping across KDEF `Afraid` and FER `fear` (and all remaining labels).
- Report accuracy, balanced accuracy, macro-F1 and confusion matrices; distinguish validation from untouched final-test results.
- The assignment expressly expects appendices with confusion matrices for each reported model. Replace every figure slot with actual output and adjust the appendix as needed.
- Explain *why* model rankings, performance and mistakes differ, not just which score is highest.
- **No scores, plots, findings, authors or dates are pre-filled.** Replace every `[Replace: ...]` marker.

### Page budget

The assignment says **maximum 8 pages** and additionally requests an appendix. It does not explicitly settle whether appendices count within the limit. The conservative choice is to stay at 8 pages total until teaching staff confirm the interpretation. This initial scaffold is deliberately concise; check the final compiled page count as you add results.

### Full course hand-in (separate from the LaTeX source structure)

The teaching team also requires **one submission ZIP** with: (1) report PDF; (2) Jupyter notebooks used for training and measurement; (3) `.pkl` trained model files; and (4) a command-line executable Python webcam demo. If needed, one separate feature-extraction Python module is permitted; otherwise modeling code belongs in notebooks. The ZIP provided here is the **report starter**, not the completed course submission.

### Required AI statement

Edit `main/sections/10_ai_usage_statement.tex` to describe actual AI usage and review, preserving the wording requested by the assignment. The template includes its original disclosure wording with replacement fields.

### References to collect in Zotero (assignment's R1 and R2)

- Goodfellow et al. (2013), *Challenges in representation learning: A report on three machine learning contests*.
- Goeleven et al. (2008), *The Karolinska directed emotional faces: A validation study*.

Do not treat these reminders as a pre-built bibliography; export the authoritative entries yourself via Zotero.
