
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
app_df = pd.read_csv("/Users/mgkvenky/Desktop/application_record.csv")
credit_df = pd.read_csv("/Users/mgkvenky/Desktop/credit_record.csv")
print(app_df.head())
print(app_df.info())
print(app_df.isnull().sum())
merged_df = pd.merge(app_df, credit_df, on='ID', how='inner')
print(merged_df.head())
print(credit_df['STATUS'].value_counts())
bad_statuses = ['1', '2', '3', '4', '5']
target_df = credit_df.groupby('ID')['STATUS'].apply(lambda x: 0 if any(status in bad_statuses for status in x) else 1).reset_index()
target_df.columns = ['ID', 'TARGET']
print(target_df.head())
final_df = pd.merge(app_df, target_df, on='ID' ,how='inner')
print(final_df.head())
print(final_df['TARGET'].value_counts())

print(final_df.isnull().sum())
final_df['OCCUPATION_TYPE'].fillna('Unknown', inplace=True)
categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
df_encoded = pd.get_dummies(final_df, columns=categorical_cols)
scaler = StandardScaler()
numeric_cols = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
# Drop ID column
df_encoded.drop('ID', axis=1, inplace=True)
# Separate features (X) and label (y)
X = df_encoded.drop('TARGET', axis=1)
y = df_encoded['TARGET']

categorical_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
                    'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
df_encoded = pd.get_dummies(final_df, columns=categorical_cols)

scaler = StandardScaler()
numeric_cols = ['AMT_INCOME_TOTAL', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'CNT_FAM_MEMBERS']
df_encoded[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
X = df_encoded.drop(['ID', 'TARGET'], axis=1)
y = df_encoded['TARGET']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

with open('credit_card_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    
feature_names = X.columns.tolist()
with open('feature_names.pkl', 'wb') as file:
    pickle.dump(feature_names, file)
    
print(f"Model and {len(feature_names)} feature names saved successfully")
