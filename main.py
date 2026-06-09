from data import generate_patient_data
from model import train_model
from evaluate import evaluate_model
from visualize import (plot_training_loss, plot_confusion_matrix,
                       plot_roc_curve, plot_risk_distribution)


def main():
    print("=" * 50)
    print("  DISEASE RISK PREDICTOR - NEURAL NETWORK")
    print("=" * 50)

    print("\n[1/4] Generating patient data...")
    df = generate_patient_data()
    print(f" {len(df)}  patients generated")

    print("\n[2/4] Training neural network...")
    model, scaler, X_test, y_test, test_probs, train_losses, test_losses = train_model(df)

    print("\nEvaluating model...")
    metrics = evaluate_model(model, X_test, y_test, test_probs)

    print("\n[4/4] Generating visualizations...")
    plot_training_loss(train_losses, test_losses)
    plot_confusion_matrix(y_test, test_probs)
    plot_roc_curve(y_test, test_probs)
    plot_risk_distribution(y_test, test_probs)


    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print("\n  Charts saved to outputs/")
    print("=" * 50)


if __name__ == "__main__":
    main()