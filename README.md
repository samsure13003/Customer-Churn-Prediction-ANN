<div align="center">

<img src="photo/banner.svg" alt="Customer churn prediction with an artificial neural network" width="100%">

<br>

### Predict customer churn probability with a TensorFlow neural network and an interactive Streamlit app.

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)](https://customer-churn-prediction-ann-juzlv7qdmy8jrmwwsmlfga.streamlit.app/)

#### [Open the Streamlit app](https://customer-churn-prediction-ann-juzlv7qdmy8jrmwwsmlfga.streamlit.app/)

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Live Streamlit App](#live-streamlit-app)
- [Screenshots](#screenshots)
- [Dataset](#dataset)
- [Workflow and Preprocessing](#workflow-and-preprocessing)
- [Model](#model)
- [Project Structure](#project-structure)
- [Installation and Usage](#installation-and-usage)
- [Results and Limitations](#results-and-limitations)

## Project Overview

This project predicts whether a bank customer is likely to leave the bank. It uses an Artificial Neural Network (ANN) built with TensorFlow/Keras and a Streamlit interface for interactive predictions. The model returns a churn probability and a classification at a 0.50 threshold. The preprocessing objects and trained model are stored in `artifacts/` and reused by the app.

## Problem Statement

Customer churn can reduce revenue and increase the cost of acquiring replacement customers. A churn probability can help teams identify accounts that may need further review and retention efforts. This model is an educational prediction tool; it should not be treated as a definitive decision about an individual customer.

## Key Features

- Interactive customer inputs for geography, gender, age, credit score, balance, tenure, products, and account activity.
- Reuses the saved geography encoder, gender encoder, and scaler during inference.
- Displays a churn probability and a likely/unlikely churn classification.
- Logs training and validation metrics to TensorBoard.

## Live Streamlit App

Try the deployed application here:

**[Customer Churn Prediction — Streamlit](https://customer-churn-prediction-ann-juzlv7qdmy8jrmwwsmlfga.streamlit.app/)**

To run the local app, follow the [Installation and Usage](#installation-and-usage) steps below.

## Screenshots

The images below are TensorBoard views of the training and validation metrics saved in the `photo/` folder.

### Epoch Accuracy

![TensorBoard epoch accuracy chart](photo/Screenshot%202026-11-21%20203205.png)

### Epoch Loss

![TensorBoard epoch loss chart](photo/Screenshot%202026-11-21%20203436.png)

## Dataset

- **File:** `Data source/Churn_Modelling.csv`
- **Records:** 10,000 customers
- **Columns:** 14, including identifiers, customer attributes, and the target
- **Target:** `Exited` (`1` = exited, `0` = retained)
- **Problem type:** Binary classification

The identifier columns `RowNumber`, `CustomerId`, and `Surname` are removed before training. The remaining customer attributes are used to predict `Exited`.

| Feature | Description |
|---|---|
| `CreditScore` | Customer credit score |
| `Geography` | Customer's country; one-hot encoded |
| `Gender` | Customer's gender; label encoded |
| `Age` | Customer age |
| `Tenure` | Years the customer has been with the bank |
| `Balance` | Account balance |
| `NumOfProducts` | Number of bank products held |
| `HasCrCard` | Whether the customer has a credit card |
| `IsActiveMember` | Whether the customer is an active member |
| `EstimatedSalary` | Estimated annual salary |
| `Exited` | Target label indicating whether the customer left |

The dataset has more retained customers than churned customers (7,963 versus 2,037), so accuracy should be considered alongside class-sensitive measures such as precision and recall.

## Workflow and Preprocessing

1. Load the CSV data and remove identifier columns.
2. Label-encode `Gender` and one-hot encode `Geography`.
3. Separate the features and `Exited` target.
4. Split the data into training and validation subsets (80/20, `random_state=42`).
5. Fit a `StandardScaler` on the training features and use it to scale both subsets.
6. Train the ANN and log accuracy/loss metrics to TensorBoard.
7. Save the model and preprocessing objects to `artifacts/`.
8. Load those artifacts in the Streamlit app to score customer inputs.

## Model

The network has 12 input features and the following dense layers:

| Layer | Units | Activation |
|---|---:|---|
| Hidden layer 1 | 64 | ReLU |
| Hidden layer 2 | 32 | ReLU |
| Output layer | 1 | Sigmoid |

Training uses the Adam optimizer (learning rate `0.01`), binary cross-entropy loss, and accuracy as a tracked metric. Early stopping monitors validation loss with a patience of 10 epochs and restores the best weights. TensorBoard logs are written under `logs/fit/`.

## Project Structure

```text
Churn_Customer_ANN/
├── Data source/
│   └── Churn_Modelling.csv
├── artifacts/
│   ├── label_encoder_gender.pkl
│   ├── model.h5
│   ├── onehot_encoder_geo.pkl
│   └── scaler.pkl
├── logs/
│   └── fit/                 # TensorBoard event files
├── photo/                   # README banner and TensorBoard screenshots
│   └── banner.svg
├── app.py                   # Streamlit application
├── experiments.ipynb        # Data preparation and ANN training
├── prediction.ipynb         # Example prediction workflow
├── requirements.txt
└── README.md
```

## Installation and Usage

### 1. Get the project

```bash
git clone https://github.com/samsure13003/Churn_Customer_ANN.git
cd Churn_Customer_ANN
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the Streamlit app

```bash
streamlit run app.py
```

The local app will be available at `http://localhost:8501`. The app loads the saved model and preprocessing files from `artifacts/`.

### 5. View TensorBoard logs (optional)

With the project environment activated, run:

```bash
tensorboard --logdir=logs/fit
```

Then open `http://localhost:6006`. To create new logs, run the training cells in `experiments.ipynb`.

## Results and Limitations

The training notebook uses an 80/20 holdout split as validation data and records accuracy and loss for the training and validation sets. The provided TensorBoard screenshots show example runs. Since the notebook does not report a separate final test evaluation or a model comparison, this README does not claim a definitive test score.

The dataset is imbalanced, and a probability or thresholded prediction is not a guarantee of future customer behavior. For practical use, evaluate the model on a separate test set and review precision, recall, calibration, and the impact of false positives and false negatives.
