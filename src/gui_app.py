import tkinter as tk
from tkinter import messagebox
import pandas as pd

from src.data_loader import IrisDataLoader
from src.model_handler import IrisModel


class IrisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Iris Flower Classifier")
        self.root.geometry("400x300")

        # Load data and train model once when the app starts
        loader = IrisDataLoader("data/data.csv")
        loader.load_data()
        X_train, X_test, y_train, y_test = loader.split_data()

        self.model = IrisModel()
        self.model.train(X_train, y_train)

        # Title
        title = tk.Label(root, text="Iris Flower Classifier", font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Input fields
        self.entries = {}

        fields = [
            "Sepal Length",
            "Sepal Width",
            "Petal Length",
            "Petal Width",
        ]

        for field in fields:
            frame = tk.Frame(root)
            frame.pack(pady=3)

            label = tk.Label(frame, text=field, width=15, anchor="w")
            label.pack(side="left")

            entry = tk.Entry(frame)
            entry.pack(side="left")
            self.entries[field] = entry

        
        button = tk.Button(root, text="Predict Species", command=self.predict_species)
        button.pack(pady=15)

        # Result label
        self.result_label = tk.Label(root, text="", font=("Arial", 12, "bold"))
        self.result_label.pack(pady=10)

    def predict_species(self):
        try:
            sepal_length = float(self.entries["Sepal Length"].get())
            sepal_width = float(self.entries["Sepal Width"].get())
            petal_length = float(self.entries["Petal Length"].get())
            petal_width = float(self.entries["Petal Width"].get())

            # Create one row of input data
            input_data = pd.DataFrame([[
                sepal_length,
                sepal_width,
                petal_length,
                petal_width
            ]], columns=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width"
            ])

            prediction = self.model.predict(input_data)[0]
            self.result_label.config(text=f"Predicted Species: {prediction}")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers in all fields.")
        except Exception as e:
            messagebox.showerror("Error", str(e))


if __name__ == "__main__":
    root = tk.Tk()
    app = IrisApp(root)
    root.mainloop()