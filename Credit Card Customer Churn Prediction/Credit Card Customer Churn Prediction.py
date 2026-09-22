# Import packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA

# Read file
data_path = 'BankChurners_sample50042.csv'
data = pd.read_csv(data_path)
pd.set_option('display.max_columns', None)

# Show the basic information of the file
print(data.head())
print(data.shape)
print(data.info())

# Check target problem
# target value
attrition_counts = data['Attrition_Flag'].value_counts()

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(attrition_counts, labels=attrition_counts.index, autopct='%1.1f%%', colors = ['#4C72B0', '#DD8452'], startangle=140)
plt.title('Distribution of Attrition Flag')
plt.axis('equal')
plt.show()

# show the content of categorical data
column_indices = [3,5,6,7,8]

# Traverse each specified column index
for index in column_indices:
    column_name = data.columns[index]
    unique_values_count = data[column_name].value_counts()
    unique_values = data[column_name].unique()
    print(f"'{column_name}' The type and number of data in the column:")
    print(unique_values_count)
    print(f"\n'{column_name}' The type of data in the column:")
    print(unique_values)
    print("\n" + "="*40 + "\n")

# solve categorical data
df = data.drop(columns=['CLIENTNUM'])

# binary target
df['Attrition_Flag'] = df['Attrition_Flag'].map({
    'Existing Customer': 0,
    'Attrited Customer': 1
})


categorical_cols = [
    'Gender',
    'Marital_Status',
    'Education_Level',
    'Income_Category',
    'Card_Category'
]

# onr-hot code and bool convert to int
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
df = df.astype(int)
print(df.head())
print(df.shape)

# Heatmap show the numerical relationship
# Select numerical columns only (exclude ID)
numerical_cols = [
    'Customer_Age',
    'Dependent_count',
    'Months_on_book',
    'Total_Relationship_Count',
    'Months_Inactive_12_mon',
    'Contacts_Count_12_mon',
    'Credit_Limit',
    'Total_Revolving_Bal',
    'Avg_Open_To_Buy',
    'Total_Amt_Chng_Q4_Q1',
    'Total_Trans_Amt',
    'Total_Trans_Ct',
    'Total_Ct_Chng_Q4_Q1',
    'Avg_Utilization_Ratio'
]

corr_matrix = data[numerical_cols].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap='coolwarm',
    linewidths=0.5,
    vmin=-1,
    vmax=1
)

plt.title("Correlation Heatmap of Numerical Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Determinate the independent variables and dependent variable
X = df.drop(columns=['Attrition_Flag'])
y = df['Attrition_Flag']

# Split the training set and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# balance data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)


# Baseline 1: SVM
svm = SVC(
    kernel='rbf',
    probability=True,
    random_state=42
)

svm.fit(X_train_resampled, y_train_resampled)

y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]

# SVM report
print("===== SVM Classification Report =====")
print(classification_report(y_test, y_pred_svm))

# SVM: AUC-ROC
print("SVM ROC-AUC:", roc_auc_score(y_test, y_prob_svm))

# confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)
cm_svm_df = pd.DataFrame(
    cm_svm,
    index=['Actual 0', 'Actual 1'],
    columns=['Predicted 0', 'Predicted 1']
)
print(cm_svm_df)
print('='*40)

# Baseline2: Random forest
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train_resampled, y_train_resampled)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# RF report
print("===== Random Forest Classification Report =====")
print(classification_report(y_test, y_pred_rf))

# RF:ROC-AUC
print("Random Forest ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
cm_rf_df = pd.DataFrame(
    cm_rf,
    index=['Actual 0', 'Actual 1'],
    columns=['Predicted 0', 'Predicted 1']
)

print(cm_rf_df)

# Feature importance
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance_df)
print('='*40)

# basic KELM
class KELM:
    def __init__(self, C=1.0, gamma=1.0):
        self.C = C
        self.gamma = gamma
        self.beta = None
        self.X_train = None

    def fit(self, X, y):
        self.X_train = X
        y = y.values.reshape(-1, 1)

        K = rbf_kernel(X, X, gamma=self.gamma)
        n_samples = K.shape[0]

        self.beta = np.linalg.inv(
            K + np.eye(n_samples) / self.C
        ) @ y

    def predict(self, X):
        K_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
        y_pred = K_test @ self.beta
        return (y_pred >= 0.5).astype(int).ravel()

    def predict_proba(self, X):
        K_test = rbf_kernel(X, self.X_train, gamma=self.gamma)
        y_pred = K_test @ self.beta
        return y_pred.ravel()




# Basic KELM
kelm = KELM(C=1.0, gamma=0.05)
kelm.fit(X_train_resampled, pd.Series(y_train_resampled))

# Predict
y_pred_kelm = kelm.predict(X_test_scaled)
y_prob_kelm = kelm.predict_proba(X_test_scaled)

print("===== KELM Report(C=1, gamma = 0.05)=====")
print(classification_report(y_test, y_pred_kelm))
print("KELM ROC-AUC:", roc_auc_score(y_test, y_prob_kelm))

# change parameter
kelm = KELM(C=1.0, gamma=0.5)
kelm.fit(X_train_resampled, pd.Series(y_train_resampled))

# Predict
y_pred_kelm = kelm.predict(X_test_scaled)
y_prob_kelm = kelm.predict_proba(X_test_scaled)

print("===== KELM Report(C=1, gamma = 0.5)=====")
print(classification_report(y_test, y_pred_kelm))
print("KELM ROC-AUC:", roc_auc_score(y_test, y_prob_kelm))


# Optimized version
class QChOA:
    """
    Quantum-inspired Chimpanzee Optimization Algorithm
    """

    def __init__(
        self,
        fitness_func,
        bounds,
        n_agents=20,
        max_iter=50,
        random_state=42
    ):
        self.fitness_func = fitness_func
        self.bounds = np.array(bounds)
        self.n_agents = n_agents
        self.max_iter = max_iter
        self.dim = len(bounds)

        self.lower = self.bounds[:, 0]
        self.upper = self.bounds[:, 1]

        np.random.seed(random_state)

    def optimize(self):
        # --- Initialization ---
        X = np.random.uniform(
            self.lower,
            self.upper,
            size=(self.n_agents, self.dim)
        )

        fitness = np.zeros(self.n_agents)

        # Control parameters
        a_max = 2.0
        a_min = 0.0

        # --- Main loop ---
        for t in range(self.max_iter):

            # 1. Evaluate fitness
            for i in range(self.n_agents):
                fitness[i] = self.fitness_func(X[i])

            # 2. Sort chimps (descending fitness)
            idx = np.argsort(-fitness)
            X = X[idx]
            fitness = fitness[idx]

            X_attacker = X[0]
            X_chaser   = X[1]
            X_barrier  = X[2]
            X_driver   = X[3]

            # 3. Update control parameter a
            a = a_max - (a_max - a_min) * (t / self.max_iter)

            # Mean best position (for quantum update)
            M_best = (X_attacker + X_chaser + X_barrier + X_driver) / 4

            # 4. Update positions
            for i in range(self.n_agents):

                r1, r2 = np.random.rand(), np.random.rand()
                A = 2 * a * r1 - a
                C = 2 * r2

                mu = np.random.rand()

                if mu < 0.5:
                    # Classical ChOA behavior
                    if abs(A) < 1:
                        # Exploitation (encircling prey)
                        D = np.abs(C * X_attacker - X[i])
                        X[i] = X_attacker - A * D
                    else:
                        # Exploration (random chimp)
                        rand_idx = np.random.randint(self.n_agents)
                        D = np.abs(C * X[rand_idx] - X[i])
                        X[i] = X[rand_idx] - A * D
                else:
                    # Quantum-inspired update
                    u = np.random.rand(self.dim)
                    X[i] = M_best + np.sign(u - 0.5) * \
                           np.log(1 / u) * np.abs(M_best - X[i])

            # 5. Boundary control
            X = np.clip(X, self.lower, self.upper)

            print(
                f"Iteration {t+1}/{self.max_iter} | "
                f"Best fitness: {fitness[0]:.5f}"
            )

        return X_attacker, fitness[0]


X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_resampled,
    y_train_resampled,
    test_size=0.25,
    stratify=y_train_resampled,
    random_state=42
)

def kelm_fitness_func(params):
    C, gamma = params

    # Parameter protection
    if C <= 0 or gamma <= 0:
        return 0

    try:
        model = KELM(C=C, gamma=gamma)
        model.fit(X_tr, pd.Series(y_tr))

        y_prob = model.predict_proba(X_val)
        auc = roc_auc_score(y_val, y_prob)

    except Exception:
        auc = 0

    return auc

# Search space
bounds = [
    (1e-2, 10),
    (1e-3, 1)
]

qchoa = QChOA(
    fitness_func=kelm_fitness_func,
    bounds=bounds,
    n_agents=20,
    max_iter=30
)

best_params, best_fitness = qchoa.optimize()

best_C, best_gamma = best_params

print("Best C:", best_C)
print("Best gamma:", best_gamma)
print("Best AUC:", best_fitness)


kelm_choa = KELM(C=best_C, gamma=best_gamma)
kelm_choa.fit(X_train_resampled, pd.Series(y_train_resampled))

y_pred_choa = kelm_choa.predict(X_test_scaled)
y_prob_choa = kelm_choa.predict_proba(X_test_scaled)

print("\n===== QChOA-KELM Final Test Report =====")
print(classification_report(y_test, y_pred_choa))
print("ChOA-KELM ROC-AUC:", roc_auc_score(y_test, y_prob_choa))

cm_choa = confusion_matrix(y_test, y_pred_choa)
cm_choa_df = pd.DataFrame(
    cm_choa,
    index=['Actual 0', 'Actual 1'],
    columns=['Predicted 0', 'Predicted 1']
)

print("Confusion Matrix (QChOA-KELM):")
print(cm_choa_df)
