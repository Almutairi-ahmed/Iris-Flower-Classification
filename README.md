# Iris Classification Project

A compact machine learning project that classifies Iris flowers from their sepal and petal measurements. The project demonstrates a reusable workflow for loading a CSV dataset, preparing features, training a classifier, evaluating its performance, and making predictions for new measurements through a simple GUI.


## Overview

The model uses four measurements to identify one of three Iris species:

- Sepal length
- Sepal width
- Petal length
- Petal width

A `LogisticRegression` model from scikit-learn is trained using the local Iris dataset. The dataset contains 150 samples and is loaded without a header; the application assigns these column names:

```text
sepal_length, sepal_width, petal_length, petal_width, species
```

The command-line application prints basic dataset information, creates visualizations, trains the model, and displays its accuracy and classification report. The optional Tkinter GUI accepts measurements and predicts the flower species.

## Features

- CSV data loading with assigned feature and target column names
- Reproducible train/test splitting with a fixed random state
- Logistic Regression classification with `max_iter=1000`
- Accuracy and classification report evaluation
- Species-count, feature-histogram, and petal-measurement scatter plots
- Tkinter interface for predicting a species from new measurements

## Project Structure

```text
.
├── data/
│   └── data.csv              # Iris dataset used by the application
├── src/
│   ├── data_loader.py        # Dataset loading and train/test splitting
│   ├── model_handler.py      # Model training, prediction, and evaluation
│   ├── visualizer.py         # Dataset information and plot generation
│   └── gui_app.py            # Tkinter prediction interface
├── main.py                   # Command-line application entry point
├── README.md
└── .gitignore
```

Running the command-line application also creates these generated image files in the project root. 

```text
species_counts.png
feature_histograms.png
scatter_plot.png
```

## Getting Started

### Requirements

- Python 3.10 or newer
- `pip`
- Tkinter, only if you want to run the GUI

### Installation

Move into the project directory and create a virtual environment:

```bash
cd IT315-project
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

Install the required packages:

```bash
python -m pip install pandas scikit-learn matplotlib
```

On Ubuntu or Debian, install Tkinter separately if it is not already available:

```bash
sudo apt install python3-tk
```

## Usage

Run the command-line application from the project root:

```bash
python main.py
```

The program loads `data/data.csv`, displays basic dataset information, generates three plots, trains the Logistic Regression model, and prints the accuracy and classification report.

To run the graphical prediction application:

```bash
PYTHONPATH=src python src/gui_app.py
```

On Windows PowerShell, use:

```powershell
$env:PYTHONPATH = "src"
python src/gui_app.py
```

Enter the four measurements in the GUI and select **Predict Species** to see the predicted label.

## Visualization

The command-line application generates three visualizations using Matplotlib:

- `species_counts.png`: number of samples for each species
- `feature_histograms.png`: distributions of the four measurements
- `scatter_plot.png`: petal length compared with petal width by species

The histogram and scatter plot are also displayed while the application runs, depending on the local Matplotlib environment.

## Technologies

- Python
- pandas
- scikit-learn
- Matplotlib
- Tkinter


