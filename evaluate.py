import numpy as np 
import torch
from sklearn.metrics import(
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)


def evaluate_model(model, X_test, y_test, test_probs):

    test_preds = (test_probs > 0.5).astype(int)


    accuracy   = accuracy_score(y_test, test_preds)
    precision  = precision_score(y_test, test_preds)
    recall     = recall_score(y_test, test_preds)
    f1         = f1_score(y_test, test_preds)
    roc_auc    = roc_auc_score(y_test, test_probs)
    cm         = confusion_matrix(y_test, test_preds)


    print("=" * 50)
    print("  NEURAL NETWORK EVALUATION RESULTS")
    print("=" * 50)
    print(f" Accuracy :  {accuracy:.4f}")
    print(f" precision:  {precision:.4f}")
    print(f" Recall: {recall:.4f}")
    print(f" f1: {f1:.4f}")
    print(f"ROC-AUC : {roc_auc:.4f}")
    print("=" * 50)
    print("\nConfusion Matrix:")
    print(f" True Negatives:   {cm[0][0]}   | False Positives: {cm[0][1]}")
    print(f" False Negatives: {cm[1][0]}   | True Positives:  {cm[1][1]}")
    print("\nFull Classification Report:")
    print(classification_report(y_test,test_preds,
                                target_names=["Low Risk", "High Risk"]))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1":f1,
        "roc_auc": roc_auc,
        "cm": cm
    }

if __name__ == "__main__":
    from data import generate_patient_data
    from model import train_model


    print("Training mode...")
    df = generate_patient_data()
    model, scaler, X_test, y_test, test_probs, train_losses, test_losses = train_model(df)

    print("\nEvaluating..")

    metrics = evaluate_model(model, X_test, y_test, test_probs)
