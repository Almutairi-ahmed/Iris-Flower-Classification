import matplotlib.pyplot as plt


class IrisVisualizer:
    def __init__(self, df):
        self.df = df

    def show_basic_info(self):
        print("First 5 rows:")
        print(self.df.head())

        print("\nDataset shape:")
        print(self.df.shape)

        print("\nColumn names:")
        print(list(self.df.columns))

        print("\nMissing values:")
        print(self.df.isnull().sum())

        print("\nSpecies counts:")
        print(self.df["species"].value_counts())

    def plot_species_counts(self):
        self.df["species"].value_counts().plot(kind="bar", color=["skyblue", "orange", "green"])
        plt.title("Number of Samples per Species")
        plt.xlabel("Species")
        plt.ylabel("Count")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig("species_counts.png")
        plt.close()

    def plot_feature_histograms(self):
        features = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]

        self.df[features].hist(bins=15, figsize=(10, 8), color="steelblue", edgecolor="black")
        plt.suptitle("Feature Distributions")
        plt.tight_layout()
        plt.savefig("feature_histograms.png")
        plt.show()
        plt.close()
    def plot_scatter(self):
        species_names = self.df["species"].unique()
        colors = ["red", "blue", "green"]

        plt.figure(figsize=(7, 5))

        for species, color in zip(species_names, colors):
            subset = self.df[self.df["species"] == species]
            plt.scatter(
                subset["petal_length"],
                subset["petal_width"],
                label=species,
                color=color,
                alpha=0.7,
            )

        plt.title("Petal Length vs Petal Width")
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.legend()
        plt.tight_layout()
        plt.savefig("scatter_plot.png")
        plt.show()
        plt.close()