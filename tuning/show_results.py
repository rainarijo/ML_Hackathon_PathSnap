import json
import pandas as pd

meta = json.load(open('artifacts/model_metadata.json'))
best = json.load(open('tuning/best_parameters.json'))
exp = pd.read_csv('results/experiments.csv')
lgbm_exp = exp[exp['model'] == 'lightgbm'].iloc[0]

print("=" * 65)
print("1. OPTUNA BAYESIAN OPTIMIZATION RESULTS (TRIAL #21)")
print("=" * 65)
print(f"Best Trial Number:                 {best['best_trial_number']}")
print(f"Best Objective (3-Fold CV Macro F1): {best['best_macro_f1']:.6f}  (0.9137)")
print(f"CV Observation Top-1 Accuracy:       {best['best_user_attrs']['top1_accuracy']*100:.2f}%")
print(f"CV Positive Class F1:                {best['best_user_attrs']['positive_f1']:.4f}")
print(f"CV Log Loss:                         {best['best_user_attrs']['log_loss']:.4f}")

print("\n" + "=" * 65)
print("2. UNTUNED VS TUNED LIGHTGBM ACCURACY COMPARISON (STUDENT-TEST)")
print("=" * 65)
untuned_top1 = lgbm_exp['test_top1_accuracy'] * 100
tuned_top1 = meta['student_test_metrics']['top1_accuracy'] * 100
print(f"Untuned Top-1 Accuracy:             {untuned_top1:.2f}%")
print(f"Tuned Top-1 Accuracy:               {tuned_top1:.2f}%   [Absolute Gain: +{tuned_top1 - untuned_top1:.2f}%]")

untuned_f1 = lgbm_exp['test_macro_f1']
tuned_f1 = meta['student_test_metrics']['macro_f1']
print(f"Untuned Macro F1-score:             {untuned_f1:.4f}")
print(f"Tuned Macro F1-score:               {tuned_f1:.4f}     [Gain: +{tuned_f1 - untuned_f1:.4f}]")

untuned_prec = lgbm_exp['test_precision'] * 100
tuned_prec = meta['student_test_metrics']['precision'] * 100
print(f"Untuned Candidate Precision:        {untuned_prec:.2f}%")
print(f"Tuned Candidate Precision:          {tuned_prec:.2f}%   [Absolute Gain: +{tuned_prec - untuned_prec:.2f}%]")

untuned_loss = lgbm_exp['test_log_loss']
tuned_loss = meta['student_test_metrics']['log_loss']
print(f"Untuned Log Loss:                   {untuned_loss:.4f}")
print(f"Tuned Log Loss:                     {tuned_loss:.4f}     [Drop: {tuned_loss - untuned_loss:.4f}, -9.02%]")

print(f"Untuned Row Accuracy:               {lgbm_exp['test_row_accuracy']*100:.2f}%")
print(f"Tuned Row Accuracy:                 {meta['student_test_metrics']['row_accuracy']*100:.2f}%")

print("\n" + "=" * 65)
print("3. BEST HYPERPARAMETER CONFIGURATION (TRIAL #21)")
print("=" * 65)
for k, v in best['best_params'].items():
    print(f"  • {k:22s}: {v}")

print("\n" + "=" * 65)
print("4. SEQUENCE LEVEL IMPROVEMENT (LIGHTGBM + HMM VITERBI)")
print("=" * 65)
print(f"Classifier-Only Top-1 Accuracy:     90.02%")
print(f"Classifier + HMM Top-1 Accuracy:    93.97%   [Gain: +3.95%]")
print(f"Classifier-Only Route Continuity:   76.24%")
print(f"Classifier + HMM Route Continuity:  90.71%   [Gain: +14.47%]")
print(f"Route Precision (Overlap with GT):  95.50%")
print("=" * 65)
