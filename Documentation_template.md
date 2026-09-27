# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Code Crafter

**Team Members:** Manish Saini, Pushpa Chaudhary, Nisha Kumari, Sadaf

**Submission Date:** 27 September 2026

---

## 1. Executive Summary

Our approach to this challenge was a hybrid one: a multi-rule blocking layer paired with an XGBoost classifier that does the actual matching. The blocking step was the part we spent the most time on, since it's what keeps the Source 1 × (Source 2 + Source 3) comparison space from exploding, and it works by stacking several complementary name and address signals instead of relying on one exact key. Once we have candidate pairs, XGBoost scores them using a mix of exact, fuzzy, token-level, and structural features to tell genuine matches apart from the many near-misses in the data.

For inference, we run the candidate pairs through the trained model and apply a threshold that leans toward precision, since that's what the challenge's F0.5 metric rewards. We arrived at that threshold through a round of validation experiments rather than just picking the default.

---

## 2. Methodology

### 2.1 Problem Analysis

We're working with business records from three sources that don't share any common ID. Source 1 is the clean, deduplicated reference set, while Source 2 and Source 3 can each contain several records that all point to the same real business. Digging into the data, a few things stood out:

- Names vary a lot in punctuation, abbreviations, legal suffixes, spelling, transliteration, and even word order.
- The same business can look quite different across sources just because of how the name was formatted.
- Addresses are messy too — abbreviated, reordered, missing pieces, landmarks thrown in, inconsistent spacing, numbering that doesn't line up.
- A good chunk of Source 2 and Source 3 records simply don't have address info at all.
- Some pairs look like strong name matches but have conflicting addresses — these turned out to be some of the hardest negatives to handle.
- A single Source 1 entity might have zero matches, one, or several across the other two sources.
- The test set throws in France alongside the US and India, so we made sure country was treated as a general input attribute rather than hard-coding for just two countries.

The ground truth also showed this same variability — anywhere from zero to multiple matches per Source 1 entity — which told us early on that a simple one-to-one matching setup wasn't going to cut it.

### 2.2 Solution Strategy

**Approach Type:** Hybrid Blocking + Supervised Classifier

**Core Innovation:** Rather than betting on one exact matching key, we let several independent name and address signals each generate their own candidates, then took the union. That way a true match has multiple chances to make it into the candidate set. XGBoost then does the heavy lifting of separating real matches from look-alikes using the hard negatives we deliberately kept in training.

**Pipeline:** Source 1 + Source 2 + Source 3 → Normalization / Keys → Multi-rule Blocking → Candidate Union → Feature Engineering for Pairs → XGBoost Classifier → Probability Threshold → Final Matching Results

---

## 3. Candidate Generation (Blocking)

Comparing every S1 record against every S2/S3 record was never going to be feasible at this scale, so we built a multi-stage blocking strategy to cut that space down without losing too many true matches in the process. All of the rules lean on normalized names and address-derived signals rather than any outside data.

- **Blocking keys used:**
  - Exact normalized business name — country + the normalized business name.
  - Sorted-name matching — country + alphabetically sorted name tokens, which catches cases where the words are just in a different order.
  - Name-token + address-number matching — country + a normalized name-token signature + the address number.
  - Address-based blocking (R4) — country + address number + address token signature. We capped these keys at a frequency of 500 so they wouldn't balloon into huge, unhelpful candidate groups.
  - First-name-token + address-token blocking (R5) — country + first name token + address token signature, again capped at 500.
  - Name-prefix + address-number blocking (R6) — country + name prefix signature + address number, capped at 250.

  We took the union of everything these rules produced and dropped duplicates to get the final candidate set.

- **Candidate pairs generated:**
  - 100,000-S1 development run: **8,063,782** candidate pairs (down to about 1.24 million once we finished feature generation for training).
  - Full test set (inference): **44,819,572** candidate pairs.

- **How we ensured true matches were not lost:**
  We checked how well blocking actually preserved the true matches by comparing against the ground truth on our validation subset:
  - True match pairs: **68,925**
  - Of those, present in our candidates: **54,319**
  - Candidate pair recall: **78.81%**
  - S1 entities with at least one true match: **18,808**
  - Mean S1-level candidate recall: **78.74%**
  - S1 entities fully covered: **9,812 / 18,808**

  So blocking did its job of shrinking the search space, but it also exposed the ceiling on our whole system: if a true match never makes it into the candidate set, there's nothing the classifier can do to recover it later.

---

## 4. Matching Model

With candidates in hand, we needed something to actually decide which pairs are real matches. We used an XGBoost binary classifier that outputs the probability a given S2/S3 candidate is really the same business as its paired S1 entity.

**Features used:**
- Name features: `name_exact`, `name_sorted_exact`, `name_token_exact`, `first_name_exact`, `name_prefix_exact`, `name_suffix_exact`, `name_fuzz`, `name_token_jaccard`, `name_len_diff`
- Address features: `address_exact`, `address_number_exact`, `address_token_exact`, `address_fuzz`, `address_token_jaccard`, `address_len_diff`
- Other: `s1_name_missing`, `candidate_name_missing`, `s1_address_missing`, `candidate_address_missing`. We also kept country/source as part of the feature set rather than filtering on it upfront.

**Model type:** XGBoost (binary logistic objective)
- max_depth = 8, learning_rate = 0.05
- subsample = 0.85, colsample_bytree = 0.90
- min_child_weight = 5, reg_lambda = 2.0
- GPU histogram training; scale_pos_weight ≈ 3.556
- Up to 700 boosting rounds, with early stopping after 50 rounds of no improvement

We split train and validation at the S1 entity level rather than the pair level — otherwise pairs from the same S1 entity could leak across the split and inflate our validation numbers.

**Threshold selection method:** Since the competition scores on F0.5, which cares more about precision than recall, we didn't just use the default 0.5 cutoff — we tuned the threshold against S1-level validation performance instead. 0.80 came out on top:
- Macro Precision: 0.8588 | Macro Recall: 0.6856 | Macro F0.5: 0.7926
- Micro Precision: 0.9119 | Micro Recall: 0.6856 | Micro F0.5: 0.8554

That threshold reflects how the full S1-level matching process behaves end-to-end, not just raw classifier accuracy on individual pairs.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** Our best public leaderboard score was **F0.5 = 0.571**, using the 0.80 threshold. Locally, our S1-level validation score came out higher at **Macro F0.5 = 0.7926** (also at threshold 0.80). That gap isn't too surprising — the public leaderboard is scored on a different hidden subset, and the private/final evaluation may land somewhere else again.

  We also tried a few nearby thresholds to see how sensitive the score was:

  | Threshold | Public F0.5 |
  |----------:|------------:|
  | 0.80 | **0.571** |
  | 0.85 | 0.570 |
  | 0.90 | 0.553 |
  | 0.95 | 0.533 |

- **Common false positives (wrong merges):** Looking through the errors, most of our false positives came from entities that looked similar on the surface but weren't actually the same business:
  - Same or near-identical names but at different addresses.
  - Common, generic business names that show up at many locations.
  - Pairs with strong name similarity but weak or missing address evidence to back it up.
  - Coincidental address overlaps in dense areas where lots of businesses share a block or building.

  These matter more than they might sound, since F0.5 punishes incorrect matches fairly harshly.

- **Common false negatives (missed matches):** Interestingly, most of our misses trace back to blocking rather than the classifier itself — if a pair never became a candidate, the model never got a chance to evaluate it. The recurring patterns were:
  - Names that were spelled quite differently.
  - Transliteration or multilingual naming variants.
  - Word order swapped around.
  - Addresses with components reordered.
  - Missing house or address numbers.
  - Address fields that were missing or too noisy to use.
  - Pairs that just didn't share any strong exact signal our blocking rules could latch onto.

  This reinforced something we kept coming back to: candidate recall is really the ceiling on the whole system — no amount of classifier tuning fixes a match that blocking already dropped.

---

## 6. Conclusion

Overall, pairing multi-rule blocking with an XGBoost classifier let us resolve entities across S1, S2, and S3 without brute-forcing every possible comparison. Blocking took millions of potential pairs down to something manageable, and the classifier then used exact, fuzzy, and token-level name/address signals to make precision-focused calls on each one. We ended up with **F0.5 = 0.571** on the public leaderboard and **Macro F0.5 = 0.7926** locally at a threshold of 0.80. If we took one lesson from this whole exercise, it's that blocking recall matters more than we initially expected — a stronger classifier just can't make up for a match that got filtered out earlier in the pipeline.

---

## Appendix

### A. Code Artefacts

The full, runnable code is in the submission zip under `code/business_entity_resolution/` (all source in `src/`, with a `README.md` and `requirements.txt`). Here's how it's laid out:

```text
business_entity_resolution/
├── src/
│   ├── blocking/
│   │   ├── build_full_base.py
│   │   ├── build_blocking_train_v3.py
│   │   ├── evaluate_blocking_v3.py
│   │   ├── analyze_ml_missed_pairs_v1.py
│   │   ├── analyze_ml_missed_pairs_v2.py
│   │   └── test_prefix_suffix_number_caps.py
│   ├── pipeline/
│   │   ├── build_ml_train_v1.py
│   │   └── resume_ml_features_v1.py
│   └── model/
│       ├── train_xgb_v2.py
│       ├── evaluate_xgb_v2.py
│       ├── predict_test_xgb_v1.py
│       └── predict_test_xgb_4thresholds.py
├── results/
│   ├── blocking/
│   ├── ml/
│   └── models/
├── README.md
├── requirements.txt
└── Documentation_template.md
```

To reproduce `output/matching_results.tsv` and `output/candidate_pairs.tsv`, run `build_blocking_train_v3.py` for candidate generation, then `build_ml_train_v1.py` / `resume_ml_features_v1.py` for features, `train_xgb_v2.py` to train, and finally `predict_test_xgb_v1.py` / `predict_test_xgb_4thresholds.py` for inference.

### B. Additional Results

We tried a number of different blocking configurations along the way. One pattern that kept showing up: pushing blocking recall higher also pushed candidate volume up a lot, so there was always a trade-off between recall, compute cost, and how much noise the classifier had to filter out downstream. That's why we ended up capping key frequencies for the address-based rules and leaning on several exact/normalized name and address signals together rather than one silver-bullet key. In the end, XGBoost handled the precision-oriented decision-making over whatever candidates came out of blocking, instead of us hand-writing a deterministic matching rule.

---

