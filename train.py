import pandas as pd                      # pandas: tool for working with tables (DataFrames)
import joblib                            # joblib: saves/loads Python objects (our model) to a file
import os                                # os: lets us create folders from code

from sklearn.model_selection import train_test_split          # splits data into training and testing parts
from sklearn.compose import ColumnTransformer                 # applies different processing to different columns
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # converts text to numbers / scales numbers
from sklearn.pipeline import Pipeline                         # chains preprocessing + model into one object
from sklearn.ensemble import RandomForestClassifier           # the ML algorithm we use
from sklearn.metrics import accuracy_score, classification_report  # to measure how good the model is

# ---------- 1. LOAD DATA ----------
df = pd.read_csv("D:\\churn_prediction\\data\\churn.csv")  # read the CSV file into a table called df

# ---------- 2. CLEAN DATA ----------
# TotalCharges is stored as text because some rows contain blank spaces.
# errors="coerce" turns anything that can't become a number into NaN (missing).
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df = df.dropna()                         # remove rows with missing values (only ~11 rows here)

# Models only understand numbers, so convert Yes/No into 1/0
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------- 3. CHOOSE FEATURES ----------
numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]            # columns that are already numbers
categorical_features = ["Contract", "InternetService",
                        "PaymentMethod", "TechSupport"]                    # columns that are text categories

X = df[numeric_features + categorical_features]   # X = inputs the model learns from
y = df["Churn"]                                   # y = the answer we want to predict

# ---------- 4. TRAIN / TEST SPLIT ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # keep 20% of data hidden for testing
    random_state=42,     # makes the split the same every time (reproducible)
    stratify=y           # keeps the churn/no-churn ratio equal in both parts
)

# ---------- 5. PREPROCESSING ----------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    # StandardScaler: rescales numbers so they have similar ranges
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    # OneHotEncoder: turns "Month-to-month" into columns of 0s and 1s.
    # handle_unknown="ignore": don't crash if the website sends an unseen category
])

# ---------- 6. MODEL PIPELINE ----------
model = Pipeline([
    ("prep", preprocessor),                       # step 1: preprocess the data
    ("clf", RandomForestClassifier(
        n_estimators=100,                         # number of decision trees
        min_samples_leaf=5,                       # prevents overfitting and keeps file size small
        class_weight="balanced",                  # gives extra importance to the rarer "churn" class
        random_state=42                           # reproducible results
    )),
])

model.fit(X_train, y_train)              # TRAINING happens here: the model learns patterns

# ---------- 7. EVALUATE ----------
predictions = model.predict(X_test)      # predict on data the model has never seen
print("Accuracy:", accuracy_score(y_test, predictions))   # % of correct predictions
print(classification_report(y_test, predictions))         # precision, recall, f1 for each class

# ---------- 8. SAVE ----------
os.makedirs("model", exist_ok=True)      # create the model/ folder if it doesn't exist
joblib.dump(model, "model/churn_model.joblib")   # save the whole pipeline to a file
print("Model saved!")