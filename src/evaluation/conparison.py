import matplotlib.pyplot as plt


def plot_comparison(results: dict):
    names = list(results.keys())
    rmse = [v["rmse"] for v in results.values()]

    plt.figure(figsize=(8, 4))
    plt.bar(names, rmse)
    plt.ylabel("RMSE")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.show()
