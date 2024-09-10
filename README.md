# Champions League Simulation using ELO Ratings
This project simulates the outcome of the UEFA Champions League based on ELO ratings, visualizing the progression of teams from the group stage to the final using a Python-based simulation. The new format of the Champions League is taken into account, with teams advancing through different stages based on predefined criteria.

# Project Overview
The Champions League tournament is visualized using a directed graph. Teams are categorized into:

Top 8 teams (who directly enter the Round of 16)
Next 16 teams (who must play in the Play-in round to qualify for the Round of 16)
The graph progresses through each round, with the outcome of matches being predicted based on ELO ratings (a popular method for ranking teams).

# Features
Graph-based Visualization: The tournament is visualized using Graphviz to clearly show how teams advance through stages.
ELO-based Predictions: Teams’ ELO ratings determine their probability of winning in each round.
How It Works
The simulation follows this flow:

Group Stage: All teams start in the group stage, and based on their rankings:
Top 8 teams advance directly to the Round of 16.
The next 16 teams enter the Play-in round.
Play-in Round: Teams play in the Play-in round, with winners advancing to the Round of 16.
Round of 16 to Final: The tournament progresses through the knockout stages until a champion is crowned.
The visualization of this flow is created using a directed graph that represents each stage and its participants.

# Requirements
To run this project, you'll need to install the following dependencies:

bash
Copiar código
pip install graphviz
You also need to have Graphviz installed on your system. You can download it from here.

# How to Run the Project
Clone the repository:

bash
Copiar código
git clone https://github.com/your-username/champions-league-simulation.git
cd champions-league-simulation
Run the simulation:

bash
Copiar código
python champions_league_simulation.py
This will generate a file champions_league_simulation.png, which visualizes the tournament structure.

Open the visualization: You can open the generated .png file to view the simulation of the Champions League.


