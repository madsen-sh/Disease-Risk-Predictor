#  Disease Risk Predictor — Neural Network

A deep learning model that predicts patient disease risk using clinical health
metrics. Built from scratch using PyTorch — no pre-trained models, no shortcuts.

---

##  What It Does

- Predicts whether a patient is **high risk or low risk** for disease
- Trained on 1,000 synthetic patient records with 10 clinical features
- Outputs a **probability score** (0.0 to 1.0) for each patient
- Evaluates performance using Accuracy, Precision, Recall, F1, and ROC-AUC
- Generates 4 research-quality diagnostic charts

---

##  Clinical Features Used

| Feature | Description |
|---|---|
| Age | Patient age (18–90) |
| BMI | Body Mass Index |
| Glucose | Fasting blood glucose (mg/dL) |
| Blood Pressure | Diastolic BP (mmHg) |
| Insulin | Blood insulin level |
| HbA1c | 3-month average blood sugar — strongest diabetes marker |
| Family History | First-degree relative with diabetes (0/1) |
| Physical Activity | Hours of exercise per week |
| Smoking | Current smoker (0/1) |
| Cholesterol | Total cholesterol (mg/dL) |

---

##  Neural Network Architecture

```
Input (10 features)
        ↓
   Linear(10 → 16)
      ReLU
    Dropout(0.2)
        ↓
   Linear(16 → 8)
      ReLU
        ↓
   Linear(8 → 1)
     Sigmoid
        ↓
Output (risk probability 0–1)
```

---

##  Project Structure

```
disease_risk_predictor/
│
├── data.py        # Generates 1,000 synthetic patient records
├── model.py       # Neural network architecture + training loop
├── evaluate.py    # Metrics: accuracy, F1, ROC-AUC, confusion matrix
├── visualize.py   # 4 diagnostic charts
├── main.py        # Runs the full pipeline end to end
└── outputs/       # Generated charts saved here
```

---

##  Visualizations

### 1. Training Loss Curve
Shows train vs test loss over 30 epochs. A gap between the two
lines indicates overfitting. Dropout layers reduce this gap.

### 2. Confusion Matrix
Breaks down predictions into True Positives, True Negatives,
False Positives, and False Negatives. In healthcare, minimizing
False Negatives (missed diagnoses) is the priority.

### 3. ROC Curve
Plots True Positive Rate vs False Positive Rate at every threshold.
AUC score shows how well the model separates high-risk from low-risk
patients regardless of the decision threshold.

### 4. Risk Distribution
Shows predicted probability distributions for actual high-risk vs
low-risk patients. Clear separation = good model calibration.

---

##  How To Run

**1. Install dependencies:**
```
pip install numpy pandas matplotlib scikit-learn torch
```

**2. Run the full pipeline:**
```
python main.py
```

Charts will be saved to the `outputs/` folder automatically.

---

##  How It Works

### Step 1 — Data Generation
1,000 patients are generated with realistic clinical values.
Risk labels are assigned using a weighted formula based on actual
diabetes risk factors — higher HbA1c, BMI, glucose, and family
history all increase risk. Physical activity reduces it.

### Step 2 — Training
The neural network learns by:
1. Making a prediction (forward pass)
2. Measuring how wrong it was (BCE Loss)
3. Calculating which weights caused the error (backpropagation)
4. Adjusting weights to reduce the error (Adam optimizer)

This loop runs 30 times (epochs) over the entire dataset.

### Step 3 — Evaluation
The model is evaluated on 200 held-out patients it never saw
during training. This gives an honest measure of real-world
performance.

---

##  Key Design Decisions

**Why PyTorch over scikit-learn?**
scikit-learn has built-in classifiers but they are fixed architectures.
PyTorch lets you define any architecture from scratch — giving full
control over layers, activation functions, and regularization.

**Why Dropout?**
Randomly disabling 20% of neurons during training prevents the
network from memorizing the training data. This improves
generalization to new patients.

**Why BCELoss?**
Binary Cross Entropy Loss is the standard loss function for
binary classification. It penalizes confident wrong predictions
more than uncertain wrong predictions.

**Why Recall matters more than Accuracy in healthcare?**
Accuracy treats all errors equally. In disease detection, a
False Negative (telling a sick patient they are fine) is far
more dangerous than a False Positive. Recall measures how many
actual high-risk patients the model correctly identifies.

**Why StandardScaler over MinMaxScaler?**
Neural networks train better when input features have mean=0
and standard deviation=1. StandardScaler achieves this.
MinMaxScaler squishes to 0-1 range but doesn't center the data.

---

##  Built With

- Python 3.12
- PyTorch — neural network architecture and training
- NumPy — numerical computing
- Pandas — data manipulation
- scikit-learn — preprocessing, metrics, train/test split
- Matplotlib — visualizations

---

*Part of my ML portfolio. See also:*
*[Music Recommendation System](https://github.com/madsen-sh/music-recommender)*

