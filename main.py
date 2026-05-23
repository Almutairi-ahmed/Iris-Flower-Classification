from src.data_loader import IrisDataLoader
from src.model_handler import IrisModel
from src.visualizer import IrisVisualizer


def main():
    file_path = "data/data.csv"

    loader = IrisDataLoader(file_path)
    df = loader.load_data()
    
    viz = IrisVisualizer(df)
    viz.show_basic_info()
    viz.plot_species_counts()
    viz.plot_feature_histograms()
    viz.plot_scatter()


    X_train, X_test, y_train, y_test = loader.split_data()

    model = IrisModel()
    model.train(X_train, y_train)

    results = model.evaluate(X_test, y_test)
    
    
    print("Model: Logistic Regression")
    print(f"Accuracy: {results['accuracy']}")
    
    print("\nClassification Report:")
    print(results["classification_report"])


if __name__ == "__main__":
    main()