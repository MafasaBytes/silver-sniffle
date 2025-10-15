"""Quick analysis of training metrics to diagnose the WER issue."""
import json

with open('models/bilstm_baseline/training_metrics.json') as f:
    metrics = json.load(f)

print("="*60)
print("Training Analysis")
print("="*60)

print("\nFinal 5 Epochs:")
for i in range(32, 37):
    t = metrics['train_metrics'][i]
    d = metrics['dev_metrics'][i]
    print(f"\nEpoch {i+1}:")
    print(f"  Train: loss={t['loss_mean']:.3f}, frame_acc={t['frame_accuracy']*100:.2f}%")
    print(f"  Dev:   loss={d['loss_mean']:.3f}, frame_acc={d['frame_accuracy']*100:.2f}%")

print("\n" + "="*60)
print("Key Observations:")
print("="*60)

final_train = metrics['train_metrics'][-1]
final_dev = metrics['dev_metrics'][-1]

print(f"\nFinal Training Loss: {final_train['loss_mean']:.3f}")
print(f"Final Dev Loss: {final_dev['loss_mean']:.3f}")
print(f"Final Train Frame Acc: {final_train['frame_accuracy']*100:.2f}%")
print(f"Final Dev Frame Acc: {final_dev['frame_accuracy']*100:.2f}%")

print("\nProblem: Frame accuracy ~3-4% is EXTREMELY LOW")
print("Expected: At least 20-30% for a working baseline")
print("\nDiagnosis: Model is barely learning anything!")
print("The CTC loss converged, but to a poor local minimum.")
