from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
import joblib
import numpy as np
# from flask import Flask, render_template
# import requests
# from bs4 import BeautifulSoup

app = Flask(__name__)
# @app.route("/live-scores")
# def live_scores():
#     headers = {
#         'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
#         'Accept': 'application/json',
#         'Accept-Language': 'en-US,en;q=0.9',
#     }
#     live_url = "https://api.cricbuzz.com/api/html/cricket-match/live-matches"
#     schedule_url = "https://api.cricbuzz.com/api/html/cricket-series/5125/schedule"

#     live_resp = requests.get(live_url, headers=headers)
#     schedule_resp = requests.get(schedule_url, headers=headers)

#     # Debug prints
#     print("Live matches status:", live_resp.status_code)
#     print("Live matches response snippet:", live_resp.text[:300])

#     # Parse live matches JSON safely
#     try:
#         live_data = live_resp.json()
#     except Exception as e:
#         print("Failed to parse live matches JSON:", e)
#         live_data = {}

#     try:
#         schedule_data = schedule_resp.json()
#     except Exception as e:
#         print("Failed to parse schedule JSON:", e)
#         schedule_data = {}

#     live_matches = []
#     for series in live_data.get("matchList", []):
#         for match in series.get("matches", []):
#             live_matches.append({
#                 "title": match.get("matchDesc"),
#                 "desc": match.get("venue"),
#                 "status": match.get("status")
#             })

#     schedule = []
#     for row in schedule_data.get("schedule", []):
#         schedule.append({
#             "date": row.get("date"),
#             "teams": row.get("matchDesc"),
#             "venue": row.get("venue"),
#             "status": row.get("status")
#         })

#     return render_template("live_scores.html", live_matches=live_matches, schedule=schedule)






# Load model and encoders
try:
    model = joblib.load("win_predictor.pkl")  # The model to predict the winner
    le_dict = joblib.load("label_encoders.pkl")  # Label encoders for categorical data
    print("✅ Model and encoders loaded.")
except Exception as e:
    print(f"❌ Error loading model or encoders: {e}")
    model = None
    le_dict = None

# Read teams, venues, and toss decisions from matches.csv
def load_match_options(matches_file='matches.csv'):
    try:
        df = pd.read_csv(matches_file)
        teams = sorted(set(df['team1'].dropna().unique()) | set(df['team2'].dropna().unique()))
        venues = sorted(df['venue'].dropna().unique())
        decisions = sorted(df['toss_decision'].dropna().unique())
        return teams, venues, decisions
    except Exception as e:
        print(f"❌ Error loading options from matches.csv: {e}")
        return [], [], []

all_teams, all_venues, all_decisions = load_match_options()

# -------------------- Basic Routes --------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/batsman')
def batsman():
    return render_template('batsman.html')

@app.route('/bowler')
def bowler():
    return render_template('bowler.html')

@app.route('/rules')
def rules():
    return render_template('rules.html')

@app.route('/season')
def season():
    return render_template('season_analysis.html')

@app.route('/venue_stats')
def venue_stats():
    return render_template('venue_stats.html')

# -------------------- Predictor Routes --------------------
# @app.route('/predictor')
# def predictor():
#     return render_template('predictor.html', teams=all_teams, venues=all_venues, decisions=all_decisions)


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()

#         team1 = data.get('team1').lower()
#         team2 = data.get('team2').lower()
#         toss_winner = data.get('toss_winner').lower()
#         toss_decision = data.get('toss_decision').lower()
#         venue = data.get('venue').lower()

#         predicted_team, confidence = predict_winner(team1, team2, toss_winner, toss_decision, venue)

#         return jsonify({
#             'predicted_team': predicted_team,
#             'win_prob': confidence  # Percentage probability
#         })

#     except Exception as e:
#         print(f"Error: {e}")
#         return jsonify({
#             'error': 'Internal Server Error',
#             'message': str(e)
#         }), 500


@app.route('/predictor')
def predictor():
    return render_template('predictor.html', teams=all_teams, venues=all_venues, decisions=all_decisions)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        team1 = data.get('team1').lower()
        team2 = data.get('team2').lower()
        toss_winner = data.get('toss_winner').lower()
        toss_decision = data.get('toss_decision').lower()
        venue = data.get('venue').lower()

        predicted_team, confidence = predict_winner(team1, team2, toss_winner, toss_decision, venue)

        return jsonify({
            'predicted_team': predicted_team,
            'win_prob': confidence  # Percentage probability
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({
            'error': 'Internal Server Error',
            'message': str(e)
        }), 500


def predict_winner(team1, team2, toss_winner, toss_decision, venue):
    if not model or not le_dict:
        raise Exception("Model or encoders not loaded")

    input_data = pd.DataFrame([{
        'team1': team1,
        'team2': team2,
        'toss_winner': toss_winner,
        'toss_decision': toss_decision,
        'venue': venue
    }])

    # Encode input
    for col in input_data.columns:
        input_data[col] = le_dict[col].transform(input_data[col])

    proba = model.predict_proba(input_data)[0]  # Probabilities for both teams
    team1_prob = proba[le_dict['target'].transform([team1])[0]]
    team2_prob = proba[le_dict['target'].transform([team2])[0]]

        # Decision logic
    if team1_prob > 0.5:
        predicted_team = team1
        confidence = round(team1_prob * 100, 2)
    elif team2_prob > 0.5:
        predicted_team = team2
        confidence = round(team2_prob * 100, 2)
    else:
        predicted_team =  team2
        confidence = round(max(team1_prob, team2_prob) * 100, 2)

    # If confidence is less than 50%, show 100 - confidence
    if confidence < 50:
        confidence = round(100 - confidence, 2)

    return predicted_team, confidence





# -------------------- Head-to-Head --------------------

def head_to_head(team1, team2, matches_file='matches.csv'):
    df = pd.read_csv(matches_file)
    h2h_df = df[((df['team1'] == team1) & (df['team2'] == team2)) |
                ((df['team1'] == team2) & (df['team2'] == team1))]

    total_matches = h2h_df.shape[0]
    team1_wins = h2h_df[h2h_df['winner'] == team1].shape[0]
    team2_wins = h2h_df[h2h_df['winner'] == team2].shape[0]
    no_result = h2h_df[h2h_df['winner'].isnull()].shape[0]

    return {
        "team1": team1,
        "team2": team2,
        "total_matches": total_matches,
        "team1_wins": team1_wins,
        "team2_wins": team2_wins,
        "no_result": no_result
    }

def get_unique_teams(matches_file='matches.csv'):
    df = pd.read_csv(matches_file)
    teams = pd.unique(df[['team1', 'team2']].values.ravel())
    return sorted([team for team in teams if pd.notna(team)])

@app.route('/head_to_head', methods=['GET', 'POST'])
def head_to_head_page():
    result = None
    teams = get_unique_teams()
    if request.method == 'POST':
        team1 = request.form['team1']
        team2 = request.form['team2']
        result = head_to_head(team1, team2)
    return render_template('head_to_head.html', result=result, teams=teams)

# -------------------- API Endpoints --------------------

@app.route('/api/team-stats')
def team_stats():
    try:
        with open('team_stats.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/top-batsman')
def top_batsman():
    try:
        with open('top_batsmen.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/top-bowlers')
def top_bowlers():
    try:
        with open('top_bowlers.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/season-summary')
def season_summary():
    try:
        with open('season_summary.json') as f:
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/venue-stats')
def get_venue_stats():
    try:
        with open('static/venue_stats.json', 'r') as f:
            venue_data = json.load(f)
        return jsonify(venue_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------- Run the App --------------------

if __name__ == '__main__':
    app.run(debug=True)
