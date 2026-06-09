import numpy as np 
import pandas as pd


np.random.seed(42)

N_PATIENTS = 1000

def generate_patient_data(n=N_PATIENTS):

    age     = np.random.normal(loc = 50 , scale = 15, size = n).clip(18, 90)
    bmi     = np.random.normal(loc = 27, scale = 6, size = n).clip(15, 50)
    glucose = np.random.normal(loc = 100, scale = 25, size = n).clip(60, 300)
    blood_pressure  = np.random.normal(loc = 80, scale = 15, size = n).clip(40, 140)
    insulin = np.random.normal(loc = 80, scale = 50, size = n).clip(0, 400)
    hba1c   = np.random.normal(loc = 5.5, scale = 1.2 , size = n).clip(4.0, 14.0)
    family_history  = np.random.binomial(n = 1, p = 0.3, size = n)
    physical_activity = np.random.randint(0, 8, size = n)
    smoking = np.random.binomial(n = 1, p = 0.25, size = n)
    cholesterol = np.random.normal(loc = 200, scale = 40, size = n).clip(100, 400)


    risk_score = (
        0.03 * age +
        0.05 * bmi +
        0.04 * glucose +
        0.02 * blood_pressure +
        0.01 * insulin +
        0.15 * hba1c * 10 +
        0.20 * family_history * 10 +
        (-10) * physical_activity +
        0.15 * smoking * 10 +
        0.01 * cholesterol
    )

    risk_score = risk_score + np.random.normal(0, 3, size = n)

    threshold = np.percentile(risk_score, 65)

    disease   = (risk_score > threshold).astype(int)

    df = pd.DataFrame({
        "age":                  np.round(age, 1),
        "bmi":                  np.round(bmi, 2),
        "glucose":              np.round(glucose, 1),
        "blood_pressure":       np.round(blood_pressure, 1),
        "insulin":              np.round(insulin, 1),
        "hba1c":                np.round(hba1c, 2),
        "family_history":       family_history,
        "physical_activity":    physical_activity,
        "smoking":              smoking,
        "cholesterol": np.round(cholesterol, 1),
        "disease":              disease,
    })

    return df



if __name__ == "__main__":
    df = generate_patient_data()

    print(f"Dataset shape : {df.shape}")
    print(f"\nFirst 3 Patients:")
    print(df.head(3).to_string())
    print(f"\nDisease distribution:")
    print(df["disease"].value_counts())
    print(f"\nFeature statistics:")
    print(df.describe().round(2).to_string())
    