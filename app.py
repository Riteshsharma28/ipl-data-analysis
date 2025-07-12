import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv('matches.csv')

# Drop rows where 'winner' is missing
df = df.dropna(subset=['winner'])

# ✅ Standardize all string columns to lowercase
for col in ['team1', 'team2', 'toss_winner', 'toss_decision', 'venue', 'winner']:
    df[col] = df[col].str.lower()

# Select features and target
X = df[['team1', 'team2', 'toss_winner', 'toss_decision', 'venue']]
y = df['winner']

# Label Encoding
le_team1 = LabelEncoder()
le_team2 = LabelEncoder()
le_toss_winner = LabelEncoder()
le_toss_decision = LabelEncoder()
le_venue = LabelEncoder()
le_winner = LabelEncoder()

X['team1'] = le_team1.fit_transform(X['team1'])
X['team2'] = le_team2.fit_transform(X['team2'])
X['toss_winner'] = le_toss_winner.fit_transform(X['toss_winner'])
X['toss_decision'] = le_toss_decision.fit_transform(X['toss_decision'])
X['venue'] = le_venue.fit_transform(X['venue'])
y = le_winner.fit_transform(y)

# Save label encoders
le_dict = {
    'team1': le_team1,
    'team2': le_team2,
    'toss_winner': le_toss_winner,
    'toss_decision': le_toss_decision,
    'venue': le_venue,
    'target': le_winner
}

joblib.dump(le_dict, 'label_encoders.pkl')
print("✅ Label encoders saved successfully.")

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'win_predictor.pkl')
print("✅ Model trained and saved successfully.")
