
import pandas as pd
import numpy as np
import random

# Load the ELO POINTS RANKING
elo_points_df = pd.read_csv('2024-09-07.csv')

# Load the CHAMPIONS LEAGUE MATCHES
matches_df = pd.read_excel('ChampionsLeague matches.xlsx')

# Split the "Match" column to extract the home and away teams
matches_df[['Home Team', 'Away Team']] = matches_df['Match'].str.split(' vs ', expand=True)

# Create a mapping for normalization
normalization_dict = {
    'Bayern München': 'Bayern',
    'Club Brugge': 'Brugge',
    'Manchester City': 'Man City',
    'PSG': 'Paris SG',
    'Atlético': 'Atletico',
    'B. Leverkusen': 'Leverkusen',
    'B. Dortmund': 'Dortmund',
    'Shakhtar Donetsk': 'Shakhtar',
    'Stade Brestois': 'Brest',
    'Sporting CP': 'Sporting'
}

# Apply the normalization to both Home and Away teams in the matches dataset
matches_df['Home Team'] = matches_df['Home Team'].replace(normalization_dict)
matches_df['Away Team'] = matches_df['Away Team'].replace(normalization_dict)

# Identify clubs in the matches dataset after normalization
normalized_match_clubs = pd.concat([matches_df['Home Team'], matches_df['Away Team']]).unique()

# Merge the matches data with the ELO points data for both Home Team and Away Team
merged_df = matches_df.merge(elo_points_df[['Club', 'Elo']], left_on='Home Team', right_on='Club', how='left')
merged_df = merged_df.merge(elo_points_df[['Club', 'Elo']], left_on='Away Team', right_on='Club', how='left', suffixes=('_Home', '_Away'))

# Calculate the ELO point difference between Home Team and Away Team
merged_df['Elo Difference'] = merged_df['Elo_Home'] - merged_df['Elo_Away']

#CREATE THE MATCH PROBABILITIES FOR EACH RESULT
# Adjust Elo ratings for home field advantage
hfa = 84.4

# Elo win probability formula
def elo_probability(elo_home, elo_away):
    P_home = 1 / (1 + 10 ** ((elo_away - elo_home-hfa) / 400))
    P_away = 1 - P_home
    return P_home, P_away

# Draw probability formula
k = 0.4
def draw_probability(P_home, P_away):
    return k * (1 - abs(P_home - P_away))

# Calculating probabilities
merged_df['P_Home_Win'], merged_df['P_Away_Win'] = zip(*merged_df.apply(lambda row: elo_probability(row['Elo_Home'], row['Elo_Away']), axis=1))
merged_df['P_Draw'] = merged_df.apply(lambda row: draw_probability(row['P_Home_Win'], row['P_Away_Win']), axis=1)

def normalize_probabilities(P_home, P_away, P_draw):
    total_win_prob = (1 - P_draw)
    P_home_adjusted = P_home * total_win_prob
    P_away_adjusted = P_away * total_win_prob
    return P_home_adjusted, P_away_adjusted

# Apply normalization after calculating the draw
merged_df['P_Home_Win'], merged_df['P_Away_Win'] = zip(*merged_df.apply(lambda row: normalize_probabilities(row['P_Home_Win'], row['P_Away_Win'], row['P_Draw']), axis=1))

matches_with_probabilities = merged_df[['Match', 'Home Team', 'Away Team','Elo Difference', 'P_Home_Win', 'P_Draw', 'P_Away_Win']]


# Number of simulations
num_simulations = 1000

# Initialize a dictionary to hold total points and qualification stats for each team
teams = pd.concat([matches_with_probabilities['Home Team'], matches_with_probabilities['Away Team']]).unique()
team_points = {team: 0 for team in teams}

# Initialize data structures for storing results of knockout stages
direct_qualification = {team: 0 for team in teams}
play_out_qualification = {team: 0 for team in teams}
quarterfinals = {team: 0 for team in teams}
semifinals = {team: 0 for team in teams}
finals = {team: 0 for team in teams}
winners = {team: 0 for team in teams}

# Function to simulate a match based on probabilities
def simulate_match(home_prob, draw_prob, away_prob):
    outcome = np.random.choice(['Home Win', 'Draw', 'Away Win'], p=[home_prob, draw_prob, away_prob])
    return outcome

# Function to simulate a knockout match between two teams based on Elo ratings
def simulate_knockout_match(team1, team2, elo_team1, elo_team2):
    win_prob_team1 = 1 / (1 + 10 ** ((elo_team2 - elo_team1) / 400))**2
    win_prob_team2 = (1 - win_prob_team1)
    return np.random.choice([team1, team2], p=[win_prob_team1, win_prob_team2])

# Function to simulate a knockout match between two teams based on Elo ratings. One match per stage
def simulate_final_match(team1, team2, elo_team1, elo_team2):
    win_prob_team1 = 1 / (1 + 10 ** ((elo_team2 - elo_team1) / 400))
    win_prob_team2 = (1 - win_prob_team1)
    return np.random.choice([team1, team2], p=[win_prob_team1, win_prob_team2])

# Function to simulate knockout stages
def simulate_knockout_stage(teams, elo_df):
    next_round_teams = []
    for i in range(0, len(teams), 2):
        team1 = teams[i]
        team2 = teams[i+1]

        # Get Elo ratings for both teams
        elo_team1 = elo_df[elo_df['Club'] == team1]['Elo'].values[0]
        elo_team2 = elo_df[elo_df['Club'] == team2]['Elo'].values[0]

        # Simulate the match and determine winner
        winner = simulate_knockout_match(team1, team2, elo_team1, elo_team2)
        next_round_teams.append(winner)

    return next_round_teams

# Perform the combined simulation (league + knockout stages)
for _ in range(num_simulations):
    # Temporary points dictionary for this simulation
    sim_points = {team: 0 for team in teams}

    # Simulate league matches
    for _, row in matches_with_probabilities.iterrows():
        if pd.isna(row['P_Home_Win']) or pd.isna(row['P_Draw']) or pd.isna(row['P_Away_Win']):
            continue

        outcome = simulate_match(row['P_Home_Win'], row['P_Draw'], row['P_Away_Win'])

        if outcome == 'Home Win':
            sim_points[row['Home Team']] += 3
        elif outcome == 'Draw':
            sim_points[row['Home Team']] += 1
            sim_points[row['Away Team']] += 1
        elif outcome == 'Away Win':
            sim_points[row['Away Team']] += 3

    # Rank teams by points
    sim_standings = pd.DataFrame(list(sim_points.items()), columns=['Team', 'Points'])
    sim_standings = sim_standings.sort_values(by='Points', ascending=False).reset_index(drop=True)

    # Add points for this simulation
    for team in teams:
        team_points[team] += sim_points[team]

    # Update qualification counters based on rankings
    for i, team in enumerate(sim_standings['Team']):
        if i < 8:  # Top 8 qualify directly
            direct_qualification[team] += 1
        elif i < 24:  # Next 16 go to play-out
            play_out_qualification[team] += 1

    # Step 1: Top 8 teams qualify directly for the round of 16
    top_8_teams = sim_standings.head(8)['Team'].tolist()

    # Step 2: Simulate play-out for the next 16 teams
    play_in_teams = sim_standings.iloc[8:24]['Team'].tolist()

    # Shuffle the play-in teams randomly
    random.shuffle(play_in_teams)
    play_in_winners = simulate_knockout_stage(play_in_teams, elo_points_df)

    # Combine top 8 teams and play-in winners for the round of 16
    random.shuffle(top_8_teams)

    # Intercalate the elements from both lists. The top_8 play with the winners of the previus round
    round_of_16_teams = [item for pair in zip(top_8_teams, play_in_winners) for item in pair]

    # Step 3: Simulate round of 16
    quarterfinal_teams = simulate_knockout_stage(round_of_16_teams, elo_points_df)
    for team in quarterfinal_teams:
        quarterfinals[team] += 1

    # Step 4: Simulate quarterfinals
    semifinal_teams = simulate_knockout_stage(quarterfinal_teams, elo_points_df)
    for team in semifinal_teams:
        semifinals[team] += 1

    # Step 5: Simulate semifinals
    final_teams = simulate_knockout_stage(semifinal_teams, elo_points_df)
    for team in final_teams:
        finals[team] += 1

    # Step 6: Simulate the final
    tournament_winner = simulate_knockout_match(final_teams[0], final_teams[1],
                                                elo_points_df[elo_points_df['Club'] == final_teams[0]]['Elo'].values[0],
                                                elo_points_df[elo_points_df['Club'] == final_teams[1]]['Elo'].values[0])
    winners[tournament_winner] += 1

# Calculate average points over all simulations
for team in team_points:
    team_points[team] /= num_simulations

# Create a DataFrame for final standings (league stage)
final_standings = pd.DataFrame(list(team_points.items()), columns=['Team', 'Average Points'])
final_standings = final_standings.sort_values(by='Average Points', ascending=False).reset_index(drop=True)

# Calculate probabilities for knockout stages
probabilities = {
    'Team': teams,
    'Average Points': [team_points[team] for team in teams],
    'Direct Qualification Probability (%)': [direct_qualification[team] / num_simulations * 100 for team in teams],
    'Play-out Qualification Probability (%)': [play_out_qualification[team] / num_simulations * 100 for team in teams],
    'Quarterfinal Probability (%)': [quarterfinals[team] / num_simulations * 100 for team in teams],
    'Semifinal Probability (%)': [semifinals[team] / num_simulations * 100 for team in teams],
    'Final Probability (%)': [finals[team] / num_simulations * 100 for team in teams],
    'Win Probability (%)': [winners[team] / num_simulations * 100 for team in teams]
}

# Combine the results into a DataFrame for easier viewing
master_prob_df = pd.DataFrame(probabilities)

# Sort the teams by the highest win probability
master_prob_df = master_prob_df.sort_values(by='Win Probability (%)', ascending=False).reset_index(drop=True)

#PLOTS
from matplotlib import pyplot as plt
import numpy as np

# Define sections (Assuming top 8 pass directly, next 16 go to play-outs, last 8 are eliminated)
colors = ['green' if i < 8 else 'blue' if i < 24 else 'red' for i in range(len(teams))]

# Create the plot
plt.figure(figsize=(12, 8))
bars=plt.barh(final_standings['Team'], final_standings['Average Points'],color=colors)

# Add labels and titles
plt.xlabel('Average Points', fontsize=12)
plt.ylabel('Teams', fontsize=12)
plt.title('Predicted Standings for Champions League: Teams Progress', fontsize=14, fontweight='bold')

# Invert y-axis to have the top team at the top
plt.gca().invert_yaxis()

# Add section labels: green for direct qualification, orange for play-outs, red for elimination
plt.text(15, 5, 'Direct Qualification', color='green', fontsize=12, fontweight='bold')
plt.text(12, 20, 'Play-Off', color='blue', fontsize=12, fontweight='bold')
plt.text(7, 32, 'Elimination', color='red', fontsize=12, fontweight='bold')

# Add gridlines for x-axis
plt.grid(axis='x', linestyle='--', alpha=0.7)

# Add value labels to bars
for bar in bars:
    plt.text(
        bar.get_width() - 1,   # X position of the label (slightly inside the bar)
        bar.get_y() + bar.get_height() / 2,   # Y position (centered on the bar)
        f'{bar.get_width():.2f}',   # Label text (bar value)
        va='center', ha='right', color='white', fontsize=10, fontweight='bold'
    )

# Highlight sections with background rectangles
plt.axvspan(14.15, 19, color='green', alpha=0.1)   # Direct qualification section
plt.axvspan(9.78, 14.15, color='orange', alpha=0.1)  # Play-out section
plt.axvspan(0, 9.78, color='red', alpha=0.1)      # Elimination section

# Show the plot
plt.tight_layout()
plt.show()

import seaborn as sns

# Extract relevant columns for the heatmap
heatmap_data = master_prob_df[['Play-out Qualification Probability (%)','Direct Qualification Probability (%)',
                                 'Quarterfinal Probability (%)', 'Semifinal Probability (%)', 'Final Probability (%)', 'Win Probability (%)']]

# Create a heatmap
plt.figure(figsize=(10, 10))
sns.heatmap(heatmap_data.set_index(master_prob_df['Team']), annot=True, cmap='Blues', fmt='.1f')

# Add title
plt.title('Team Probabilities for Each Stage')
plt.ylabel('Teams')
plt.xlabel('Stages')

plt.tight_layout()
plt.show()

