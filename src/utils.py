def plot_model_comparison(results, output_path='plots/model_comparison.png'):
    import matplotlib.pyplot as plt

    model_names = list(results.keys())
    accuracies = list(results.values())

    plt.figure(figsize=(8, 5))
    plt.bar(model_names, accuracies, color='skyblue')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()