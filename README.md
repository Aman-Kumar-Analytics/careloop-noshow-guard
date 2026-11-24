**Repository link:** https://github.com/Aman-Kumar-Analytics/careloop-noshow-guard

# CareLoop Clinics: No-Show Guard

From raw bookings to a risk model and an AI-assisted follow-up review.

## Summary (one finding per part)

1. **Part 1 (Python):** The raw export had 1,020 rows; after cleaning, 990 bookings remain, of which 218 are no-shows (22.0%).
2. **Part 2 (SQL):** No-show rate rises sharply with lead time: 16.2% for bookings made 1-7 days ahead against 60.5% for bookings made 31+ days ahead (43 bookings).
3. **Part 3 (ML):** Logistic regression was chosen (cross-validated ROC-AUC 0.774, test ROC-AUC 0.789); at a 0.35 threshold it flags 49 of 198 test bookings with precision 0.57 and recall 0.57.
4. **Part 4 (GenAI):** Of the first 10 call-centre notes, 9 were analysed by the Gemini API and all 9 passed validation; the two notes classed as forgot_or_no_reminder (N001, N007) both belong to bookings with no reminder sent.

## Recommendation

CareLoop should act on two fronts first. On the evening before each appointment, the front desk should phone the bookings the model flags: on the held-out test set the model flags 49 of 198 bookings and catches 57% of the real no-shows, which is far more than the 25% a 0.5 threshold would catch, and the list is short enough for a front desk to call in one evening. Second, CareLoop should make sure every booking gets a reminder and add an extra reminder for bookings made far ahead: bookings made 31+ days ahead are missed 60.5% of the time against 16.2% for bookings made within a week, and in the fitted model the odds of a no-show are about 69% lower when a reminder was sent, holding the other features constant. This is an association in observational data, not proof that reminders cause attendance. The call notes point the same way: forgetting or no reminder was the most common reason in the sample alongside distance and work conflicts (2 notes each out of 9), and both forgot_or_no_reminder notes came from bookings with reminder_sent = No. Because the notes are only 10 bookings, they illustrate rather than prove the reasons, so the first step is reminders and calls, and a larger notes review should follow.

## Model chosen and why

- Features: age, distance_km, fee_inr, lead_days, prior_no_shows, reminder_sent, and one-hot booking_channel and visit_type (11 columns). Split: 792 train, 198 test, random_state=42.
- Baseline accuracy (always predict "Attended"): 0.7525.
- The model was chosen using training data only (5-fold cross-validated ROC-AUC): logistic regression 0.774, depth-3 tree 0.678, random forest 0.747, tuned random forest 0.767 (best_params max_depth=3, n_estimators=200). Logistic regression won.
- Threshold chosen on out-of-fold training predictions: 0.5 gave precision 0.58, recall 0.25, F1 0.35 (73 flagged); 0.35 gave precision 0.51, recall 0.47, F1 0.49 (154 flagged). **0.35** was chosen because it has the higher F1 and much higher recall. Recall matters here because a missed warning costs an empty doctor slot, while an extra call costs a minute.
- One evaluation on the test set at 0.35: precision 0.57, recall 0.57, 49 flagged, ROC-AUC 0.789. Predictions are in `test_risk_scores.csv`.
- In practice a clinic would choose the threshold by how many follow-up calls the front desk can make in an evening, against the cost of an empty slot.

## Part 4: the notes

Cross-tabulation of root cause against reminder_sent (merged on appointment_id, 10 notes):

| root_cause | reminder_sent = No | reminder_sent = Yes |
|---|---|---|
| cost_or_fee | 0 | 1 |
| distance_or_transport | 1 | 1 |
| forgot_or_no_reminder | 2 | 0 |
| illness_or_emergency | 0 | 1 |
| other | 1 | 1 |
| work_or_schedule | 1 | 1 |

(The `other` row includes N004, the blank note recorded as `skipped_blank`.)

- 5 of the 10 notes belong to bookings with reminder_sent = No.
- N004 is blank, so it was recorded as `skipped_blank` with no API call.
- Prompt comparison (N009): the one-line prompt returned free text outside the allowed list, while the CRAFT prompt with the allowed values returned `other`. For N007, the first one-line prompt call failed with a server error (503/504), so I re-ran that single call once: it returned a free-text sentence, also outside the allowed list, while the CRAFT prompt returned forgot_or_no_reminder. Both notes show the same pattern: the one-line prompt drifts outside the allowed values and the CRAFT prompt stays inside them, so I would deploy the CRAFT prompt.
- Models used: N003 and N005 were classified with `gemini-3.8-flash`; N001, N002, N006, N007, N008, N009 and N010 with `gemini-3.5-flash-lite`, because the first model's free quota ran out. The first prompt-comparison attempt used `gemini-3.8-flash` (server errors); the single N007 one-line-prompt re-run used `gemini-3.5-flash-lite`.
- API calls: 32 content-generation calls are counted in `04_genai.ipynb` (1 for Task 4.1, 9 for Task 4.2, 2 for Task 4.3, plus 20 retries and re-runs), plus a few one-word model-test calls made while finding a working model. The counter printed in one cell (24) is lower because the counter variable was reset when I switched models. This is above the brief's target of fewer than 25 because of model changes, 400 errors (an unsupported thinking setting), 429 quota errors and 503 server-busy errors. Notes whose first call failed with a server error were called again; no reply was ever edited or repaired by hand.

## Limitations

1. The 10 notes are an illustration and cannot show the real reasons clinic-wide.
2. The cleaning statistics (medians, quartiles, outlier fences) were calculated on all 990 rows before the train/test split, so the test rows had a small indirect influence on the cleaning.
3. The same patient can appear in several bookings, so a random split can put one patient in both train and test; also the split is random, not past-to-future, as a real system would be.
4. The data are observational, so the reminder effect is an association, not a cause.

## Files

`01_ingest_clean.ipynb`, `rejected_rows.json`, `appointments_clean.csv`, `clinics_enriched.csv`, `load_data.sql`, `02_queries.sql`, `03_model.ipynb`, `test_risk_scores.csv`, `04_genai.ipynb`, `notes_analysed.csv`, `data/` (original files). The API key is not stored anywhere in this repository.
