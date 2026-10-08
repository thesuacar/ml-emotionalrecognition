This micro-task backlog is tailored for a 3-person team tackling the Emotion Recognition assignment. With one less participant, breaking the six major project requirements into sub-30-minute atomic tasks is critical to prevent bottlenecks and ensure every grading rubric point is hit.

  

### Assignment Digestion & Detailed Workload Distribution

**Purpose:** Develop and evaluate machine learning models to detect 7 facial emotions across two datasets (FER2013 and KDEF), addressing domain shift, implementing a fuzzy classifier, executing a creative performance challenge, and deploying a real-time webcam demo. **Scope:** Strict adherence to evaluating across datasets using standardized 48x48 grayscale formats, extracting specific features (HOG, Landmarks, PCA, geometric), and delivering an 8-page PDF report, notebooks, `.pkl` models, and a `demo.py` script.

  

| **Task ID** | **Phase**          | **Task Description**                                                                    | **Definition of Done**                                                                 | **Est. Time** | **Due Date** | **Assignee** |
| ----------- | ------------------ | --------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------- | ------------ | ------------ |
| **T01**     | 1. Env & Data      | Set up Python env with required packages (`dlib`, `cv2`, `pyFUME`, `imbalanced-learn`). | `requirements.txt` pushed to repo; all 3 team members successfully install it locally. | 15m           |              |              |
| **T02**     | 1. Env & Data      | Run provided FER2013 splitting notebook.                                                | `fer2013_train.csv`, `fer2013_validation.csv`, and `fer2013_test.csv` generated.       | 10m           |              |              |
| **T03**     | 1. Env & Data      | Run KDEF subject split and execute `preprocess.py` on KDEF images.                      | Frontal KDEF faces cropped, converted to grayscale, and resized to 48x48.              | 20m           |              |              |
| **T04**     | 2. Task 1 (Base)   | Implement HOG and PCA feature extraction pipelines.                                     | Functions written to process 48x48 images into HOG and PCA feature arrays.             | 25m           |              |              |
| **T05**     | 2. Task 1 (Base)   | Train Logistic Regression & SVM on FER2013.                                             | Models trained using training split; weights temporarily saved.                        | 25m           |              |              |
| **T06**     | 2. Task 1 (Base)   | Train Random Forest baseline on FER2013 (using provided notebook).                      | Model trained; `rf_fer_model_landmarks.pkl` saved.                                     | 20m           |              |              |
| **T07**     | 2. Task 1 (Base)   | Evaluate Task 1 models on FER validation set.                                           | Acc, Balanced Acc, Macro-F1, and Confusion Matrices generated.                         | 20m           |              |              |
| **T08**     | 3. Task 2 (KDEF)   | Extract best-performing features (from T07) on KDEF images.                             | KDEF images processed into identical feature space as FER2013.                         | 15m           |              |              |
| **T09**     | 3. Task 2 (KDEF)   | Retrain best-performing classifiers on KDEF (80/20 subject split).                      | Subject-independent models trained and evaluated; metrics generated.                   | 25m           |              |              |
| **T10**     | 4. Task 3 (Cross)  | Test KDEF-trained model on FER2013 dataset.                                             | Domain shift performance evaluated; metrics and confusion matrices recorded.           | 20m           |              |              |
| **T11**     | 5. Task 4 (Fuzzy)  | Extract 3-10 handcrafted geometric features (mouth width, eye opening) via `dlib`.      | DataFrame of geometric features generated for KDEF dataset.                            | 30m           |              |              |
| **T12**     | 5. Task 4 (Fuzzy)  | Scaffold and train Fuzzy Classifier using `pyFUME`.                                     | Human-understandable rules generated and evaluated on KDEF.                            | 30m           |              |              |
| **T13**     | 6. Task 5 (Create) | Implement Creative Challenge feature 1 (e.g., Data augmentation or SMOTE).              | Training pipeline updated with technique to handle data imbalance/variety.             | 25m           |              |              |
| **T14**     | 6. Task 5 (Create) | Implement Creative Challenge feature 2 (e.g., Hyperparameter tuning or KDEF insights).  | Best pipeline optimized; performance improvement verified on FER2013.                  | 30m           |              |              |
| **T15**     | 6. Task 5 (Create) | Finalize best model and save production `.pkl` file.                                    | Final model evaluated on Private Test (test) split; `.pkl` saved to repo.              | 15m           |              |              |
| **T16**     | 7. Task 6 (Demo)   | Adapt `baseline_demo_fer2013.py` for new best model.                                    | Script loads the new `.pkl` and handles real-time webcam frame extraction.             | 20m           |              |              |
| **T17**     | 7. Task 6 (Demo)   | Test webcam demo locally and log performance issues.                                    | Script runs without crashing; success/failure cases noted for report.                  | 15m           |              |              |
| **T18**     | 8. Reporting       | Draft Report: Task 1 (Baseline) & Task 2 (KDEF) sections.                               | Feature/classifier ranking, difficult emotions, and dataset differences written.       | 30m           |              |              |
| **T19**     | 8. Reporting       | Draft Report: Task 3 (Cross-Dataset) & Task 4 (Fuzzy) sections.                         | Domain shift explained; fuzzy rules documented and justified.                          | 30m           |              |              |
| **T20**     | 8. Reporting       | Draft Report: Task 5 (Creative) & Task 6 (Demo) sections.                               | Justification of creative choices and webcam adaptation issues written.                | 30m           |              |              |
| **T21**     | 8. Reporting       | Assemble Confusion Matrix Appendix.                                                     | All matrices added to appendix without excessive text.                                 | 15m           |              |              |
| **T22**     | 8. Reporting       | Draft AI usage statement & final 8-page PDF formatting.                                 | Official AI statement filled out; document fits within 8 pages.                        | 20m           |              |              |
| **T23**     | 8. Reporting       | Package final `.zip` deliverable.                                                       | Zip contains 8-page PDF, notebooks, `.pkl` models, and `demo.py`.                      | 10m           |              |              |

As discussed previously, if you want to quickly import this into GitHub Projects, save this table as a `.csv` and use GitHub's CSV import tool, or run a quick `gh issue create` loop in your terminal to generate the board instantly.

  

Since you are a team of 3, you can easily assign "buckets" (e.g., Person A owns Tasks 4-7, Person B owns 8-12, Person C owns 13-17, and reporting is split evenly). Do you want to establish specific feature assignments for the Fuzzy Classifier (Task 11) before you meet?