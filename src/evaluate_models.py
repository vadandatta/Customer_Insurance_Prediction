from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

def evaluate_models(models, X_test, y_test, output_dir='plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\n{name} Accuracy: {acc:.4f}")
        print(f"Classification Report:\n{report}")

        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.close()
