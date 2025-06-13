from src.preprocess import load_and_preprocess_data
from src.train_models import train_all_models
from src.evaluate_models import evaluate_models
from src.utils import plot_model_comparison
import os

if __name__ == "__main__":
    DATA_PATH = "data/Social_Network_Ads.csv"
    PLOTS_DIR = "plots"

    print("\n[1] Loading and Preprocessing Data...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data(DATA_PATH)

    print("\n[2] Training Models...")
    models = train_all_models(X_train, y_train)

    print("\n[3] Evaluating Models...")
    evaluate_models(models, X_test, y_test, output_dir=PLOTS_DIR)

    print("\n[4] Plotting Model Comparison...")
    from sklearn.metrics import accuracy_score
    results = {name: accuracy_score(y_test, model.predict(X_test)) for name, model in models.items()}
    plot_model_comparison(results, output_path=os.path.join(PLOTS_DIR, "model_comparison.png"))

    print("\n[✓] All steps completed successfully.")
