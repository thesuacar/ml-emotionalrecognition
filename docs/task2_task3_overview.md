# Tasks 2 and 3: report questions and notebook coverage

This page lists what the report has to answer for Task 2 (KDEF dataset study) and Task 3 (cross-dataset generalization), what the notebooks do for each question, and the results so far. Use it to write the report sections `04_kdef_study.tex` and `05_cross_dataset_generalization.tex`, and check the open items at the end before submitting.

| Task | Notebook | Report section |
|---|---|---|
| Task 2: KDEF dataset study (15%) | `notebooks/02_kdef_dataset_study.ipynb` | `docs/report/main/sections/04_kdef_study.tex` |
| Task 3: Cross-dataset generalization (20%) | `notebooks/03_cross_dataset_generalization.ipynb` | `docs/report/main/sections/05_cross_dataset_generalization.tex` |

All numbers below come from the saved notebook outputs.

---

## Task 2: KDEF dataset study

### Questions to answer

From the report template and the workload plan (T08, T09):

1. Use the best FER2013 feature representation from Task 1 and retrain the classifiers on preprocessed frontal KDEF images.
2. Split KDEF 80/20 **by subject**, so every person is in exactly one split. Document the counts and the random seed.
3. Explain how the KDEF model was selected **without using the test set**.
4. Report accuracy, balanced accuracy and macro-F1 on the held-out test subjects for Logistic Regression, SVM and Random Forest.
5. Explain why controlled KDEF images differ from in-the-wild FER2013 images.
6. Does the ranking of the models change compared with FER?
7. Which emotions are easiest and hardest?
8. Which emotions might transfer well to FER?
9. Cite the KDEF confusion matrices (Appendix A).

### What notebook 02 does

1. **Lists the KDEF files.** It reads all KDEF image names and extracts session (A/B), subject, emotion and pose. 4,898 images are parsed; two files with malformed names (`AM31H`, `AF31V`) are skipped.
2. **Splits by subject.** It uses the provided split `auxFiles/kdef_subject_split.pkl` (56 training / 14 test subjects, made with `train_test_split(random_state=42)` in the provided loader) and keeps only the frontal pose. That gives 784 training and 196 test images, 112 and 28 per emotion. An assert checks that no subject is in both splits. Figure: `figures/task2_kdef_class_distribution.png`.
3. **Preprocesses KDEF into the FER format** with `src/preprocess.py`: Haar face detection, box enlarged by 20%, grayscale, resized to 48x48, scaled to [0, 1].
4. **Compares mean faces** per emotion for KDEF and FER train. Figure: `figures/task2_mean_faces_kdef_vs_fer.png`.
5. **Extracts HOG**, the best Task 1 feature, with the same parameters as Task 1 (9 orientations, 8x8 cells, 2x2 blocks, 900 values).
6. **Selects the models on the training subjects only.** A grid search with 5-fold `GroupKFold` grouped by subject, scored on macro-F1, for Logistic Regression, SVM and Random Forest (model `random_state=10`). The test subjects are not used. The best configurations are saved to `models/task2_kdef_best_model_configs.pkl`.
7. **Evaluates on the 14 test subjects** and compares with the Task 1 FER results. It prints classification reports, a ranking table, and confusion matrices (`figures/task2_kdef_confusion_test.png`). The comparison is saved to `figures/task2_kdef_vs_fer_results.csv`.

### Results

Best configurations (selected by cross-validation on the training subjects):

| Classifier | Best parameters |
|---|---|
| Logistic Regression | PowerTransformer, C = 0.01 |
| SVM | PowerTransformer, RBF kernel, C = 10, gamma = 1e-4 |
| Random Forest | max_depth = 10 (see notebook for the rest) |

Macro-F1 (accuracy and balanced accuracy are in the notebook):

| Classifier | FER validation (Task 1) | FER test (Task 1) | KDEF CV (train subjects) | KDEF test subjects |
|---|---|---|---|---|
| Logistic Regression | 0.396 | – | 0.669 | 0.581 |
| SVM | 0.505 | 0.568 | **0.679** | 0.604 |
| Random Forest | 0.440 | – | 0.611 | **0.614** |

KDEF test accuracy: LR 0.577, SVM 0.602, RF 0.622.

Per-emotion F1 on the KDEF test subjects:

| Emotion | LR | SVM | RF |
|---|---|---|---|
| Happy | 0.885 | 0.926 | 0.767 |
| Disgust | 0.533 | 0.653 | 0.694 |
| Neutral | 0.610 | 0.656 | 0.689 |
| Surprise | 0.625 | 0.615 | 0.630 |
| Sad | 0.509 | 0.545 | 0.528 |
| Angry | 0.500 | 0.500 | 0.653 |
| Fear | 0.406 | 0.333 | 0.340 |

### How each question is answered

| # | Question | Where | Answer so far |
|---|---|---|---|
| 1 | Best FER features retrained on KDEF | Steps 5–6 | HOG with the Task 1 parameters; LR, SVM and RF retrained |
| 2 | Subject split, counts, seed | Step 2 | 56 / 14 subjects, 784 / 196 images, split seed 42, no overlap |
| 3 | Selection without test leakage | Step 6 | Grid search with `GroupKFold` by subject on the training subjects only; test subjects only used once at the end |
| 4 | Results table | Step 7 | Table above |
| 5 | Why KDEF differs from FER | Step 4, and Task 3 section 2 | Posed studio photos, same lighting, frontal pose, aligned faces vs varied web images. Task 3 section 2 has the measurements |
| 6 | Does the model ranking change? | Step 7 | On FER: SVM > RF > LR. On KDEF CV: SVM > LR > RF. On KDEF test: RF > SVM > LR, but within about 3 points on only 196 images, so the test ranking is not reliable. LR moves from last on FER to second on KDEF CV |
| 7 | Easiest and hardest emotions | Step 7 | Happy is the easiest for all three models; Fear is the hardest for all three |
| 8 | Which emotions might transfer | Task 3 section 3 | Predicted: emotions with large, clear shapes (Happy, Surprise). Task 3 confirms that these two transfer best |
| 9 | Confusion matrices | Step 7 | `figures/task2_kdef_confusion_test.png` |

---

## Task 3: Cross-dataset generalization (KDEF -> FER)

### Questions to answer

From the assignment ("Cross-generalization experiments") and the report template:

1. What adaptations did you have to make to run the model?
2. What is domain shift?
3. Why does performance degrade? Quantify it against in-domain results.
4. Which features transfer best?

The template also asks to analyse the per-emotion confusion and to mention crop/alignment, resolution and grayscale/RGB as confounders.

**Scope decision:** this notebook covers only **KDEF -> FER**: the best Task 2 configuration, trained on KDEF and tested on FER. FER -> KDEF is not done here (see open items).

### What notebook 03 does

| Section | What it does |
|---|---|
| 1. Load both datasets | Loads all 980 frontal KDEF images and preprocesses them as in Task 2. Extracts HOG and retrains the best Task 2 configuration (SVM, PowerTransformer, C = 10, gamma = 1e-4) on **all 980 KDEF images**; saved as `models/task3_kdef_hog_svm.pkl`. Loads the **full FER2013 dataset** (`fer2013.csv`, 35,887 images) and extracts HOG with the same function |
| KDEF -> FER evaluation | Predicts all FER images, maps KDEF label names to FER numbers (afraid -> Fear) and compares against two references: KDEF -> KDEF (Task 2 CV) and FER -> FER (Task 1, PrivateTest). Shows a classification report, a confusion matrix and the true vs predicted class distribution |
| 2.1 Imbalanced dataset | Class shares: FER vs KDEF (label shift) |
| 2.2 How different are the images? | One example face per emotion, brightness and contrast per image, mean faces, and mean face per emotion |
| 2.3 Can a model tell FER and KDEF apart? | Domain classifier: a logistic regression predicts "FER or KDEF?" from HOG, with 980 images each (FER sampled with the same emotion mix as KDEF), 5-fold CV |
| 3. Which emotions transfer? | F1 per emotion on KDEF (cross-validation grouped by person, on all 70 subjects) vs on FER |
| 3.1 Which FER images are wrong? | 14 random misclassified FER images with true and predicted emotion |
| 4. Which HOG features transfer best? | For each of the 900 HOG features: the correlation between its 7 per-emotion averages in KDEF and in FER (+1 = same pattern). Best and worst 10 features, and a map of the correlation on the face |
| 5. Notes for the report | Draft answers, to be removed before submission |

### Results

**Main result**

| Setting | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---|
| KDEF -> FER (full FER, 35,887 images) | 0.225 | 0.215 | 0.195 |
| KDEF -> KDEF (Task 2 CV) | 0.686 | 0.686 | 0.679 |
| FER -> FER (Task 1 PrivateTest) | 0.574 | 0.549 | 0.568 |

Chance level is 1/7 = 0.143 balanced accuracy, so KDEF -> FER is only about 0.07 above chance.

**Per emotion** (F1, sorted by FER score)

| Emotion | F1 on KDEF (CV) | F1 on FER | FER precision | FER recall |
|---|---|---|---|---|
| Happy | 0.89 | 0.38 | 0.560 | 0.287 |
| Surprise | 0.71 | 0.31 | 0.310 | 0.316 |
| Sad | 0.50 | 0.24 | 0.200 | 0.310 |
| Angry | 0.66 | 0.19 | 0.167 | 0.212 |
| Fear | 0.46 | 0.18 | 0.155 | 0.220 |
| Disgust | 0.69 | 0.04 | 0.020 | 0.146 |
| Neutral | 0.66 | 0.02 | 0.325 | 0.012 |

The predictions do not follow FER's class mix. Neutral is almost never predicted (recall 0.012), while Sad and Disgust are predicted far too often.

**Domain shift measurements**

| Measurement | FER | KDEF |
|---|---|---|
| Share of Happy / Disgust | 25.0% / 1.5% | 14.3% / 14.3% |
| Mean brightness (std over images) | 0.507 (0.131) | 0.327 (0.106) |
| Mean contrast (std over images) | 0.212 (0.054) | 0.161 (0.024) |
| Domain classifier, balanced accuracy | 0.988 (FER and KDEF are almost perfectly separable in HOG space) | |

**HOG features**

- The best-transferring features have correlations up to 0.95 and sit in the centre of the face (eyes, nose, mouth). The worst have correlations down to -0.88 and are mostly at the border of the image (hair, background, face outline).
- In the face map, the eyes, mouth and cheeks keep the same emotion pattern in both datasets. The forehead is around 0, and the chin/jaw corners are slightly negative.

### How each question is answered

| Question | Where | Answer so far |
|---|---|---|
| Adaptations needed | Section 1 | KDEF converted to the FER format (face crop with 20% margin, grayscale, 48x48, [0, 1]); FER only rescaled. Same HOG extractor and settings for both. KDEF label names mapped to FER numbers (afraid -> Fear); all 7 classes match. Best Task 2 configuration retrained on all KDEF images. Nothing tuned or fitted on FER |
| What is domain shift | Sections 2.1–2.3 | Training and test data come from different distributions, P_KDEF(x, y) ≠ P_FER(x, y). Covariate shift: studio vs web images (2.2, 2.3). Label shift: balanced vs imbalanced (2.1). Concept shift: posed, exaggerated expressions with reliable labels vs subtle expressions with noisy labels |
| Why performance degrades | Evaluation, 2.2, 2.3, 3, 3.1 | Macro-F1 falls from 0.679 (KDEF) to 0.195 (FER). 980 frontal studio images of 70 people cannot cover FER's poses, lighting, occlusion and image quality. HOG compares fixed image regions, so differences in face position and scale (mean faces in 2.2) break it; the domain classifier (0.988) shows how different the HOG features are. Predictions collapse onto a few classes (Neutral almost never predicted) |
| Which emotions transfer | Section 3 | Happy and Surprise transfer best (large, distinctive mouth and eye shapes, about 2x chance). Angry, Fear and Sad barely transfer. Disgust is at chance level and Neutral is below chance |
| Which features transfer best | Section 4 | Within HOG: features around the eyes, mouth and cheeks keep their emotion pattern; border features (hair, background, outline) do not. Across feature types (HOG vs landmarks vs PCA): not measured, see open items |
| Confounders | Section 2.2 | Crop/alignment: KDEF faces are smaller in the frame (20% margin). Resolution: FER is low-resolution web images. Colour: KDEF converted from RGB to grayscale |

---

## Open items before the report is final

1. **FER -> KDEF is not covered.** The report template table has a FER -> KDEF row, but notebook 03 deliberately does only KDEF -> FER. Either a teammate covers it, or the report removes the row and states the scope.
2. **The protocol differs from the template, so it needs one sentence of justification in the report.** The template says "retrain on the KDEF training subjects, evaluate on FER PrivateTest". Notebook 03 trains on all 980 KDEF images and evaluates on all of FER. That's valid because the model never sees FER, and it gives a larger test set. The KDEF -> KDEF reference is the Task 2 CV score, not a held-out test score.
3. **The per-emotion F1 table in notebook 02 is wrong.** It uses `average="macro"`, so every emotion gets the same number (`figures/task2_per_emotion_f1_kdef_vs_fer.csv`). Use `average=None` for per-emotion values. The per-emotion table in this document comes from the classification reports, which are correct.
4. **The lighting claim needs evidence.** The error-rate vs brightness/contrast check was removed from section 3.1. HOG largely ignores global brightness and contrast; in a quick check outside the notebook, changing contrast left HOG unchanged. To claim in the report that lighting is not the main cause, re-add that check or phrase the claim as reasoning.
5. **Only HOG is compared across feature types.** "Which features transfer best" is answered within HOG (which face regions). A comparison with landmarks and PCA would need an extra experiment; otherwise argue it qualitatively.
6. **Old figures in `figures/`** are not produced by the current notebook 03 and may be out of date: `task3_class_priors.png`, `task3_class_split.png`, `task3_domain_shift_stats.png`, `task3_error_vs_lighting.png`, `task3_hog_pca_domains.png`, `task3_kdef_to_fer_full.png`, `task3_kdef_to_fer_full_confusion_matrix.png`, `task3_per_class_f1.png`. Delete them or make sure the report doesn't use them.
7. **Notebook 02 assumes it runs from the repository root** (paths like `data/...`, `from src.preprocess import ...`), but it now lives in `notebooks/`. Rerunning it from `notebooks/` will fail until the paths are adjusted like in notebook 03.
8. **Notebook 03 cleanup:** remove section 5 (draft notes) and the empty last cell before submitting.
