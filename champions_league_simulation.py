
# champions_league_simulation.py

# Importing necessary libraries
from graphviz import Digraph

# Function to set up and render the Champions League tournament simulation
def create_champions_league_simulation():
    '''
    This function creates a directed graph to simulate the new format of the Champions League.
    It shows the progression from the group stage to the final, indicating which teams
    qualify for each stage.
    '''
    # Create a directed graph
    dot = Digraph(comment='Champions League Simulation')

    # Add nodes representing stages of the tournament
    dot.node('A', 'Group Stage')
    dot.node('B', 'Top 8 Teams\nDirect to Round of 16')
    dot.node('C', 'Next 16 Teams\nPlay-in Round')
    dot.node('D', 'Winners of Play-in\nAdvance to Round of 16')
    dot.node('E', 'Round of 16')
    dot.node('F', 'Quarterfinals')
    dot.node('G', 'Semifinals')
    dot.node('H', 'Final')
    dot.node('I', 'Champion')

    # Add edges to define the flow of the tournament
    dot.edge('A', 'B', label='Top 8 Teams')
    dot.edge('A', 'C', label='Next 16 Teams')
    dot.edge('C', 'D', label='Simulate Play-in')
    dot.edge('B', 'E', label='Top 8 to Round of 16')
    dot.edge('D', 'E', label='Play-in Winners')
    dot.edge('E', 'F', label='Simulate Round of 16')
    dot.edge('F', 'G', label='Simulate Quarterfinals')
    dot.edge('G', 'H', label='Simulate Semifinals')
    dot.edge('H', 'I', label='Simulate Final')

    # Render the graph to a file
    dot.render('champions_league_simulation', format='png', view=False)
    print("Tournament simulation created and saved as 'champions_league_simulation.png'.")

# Main execution
if __name__ == '__main__':
    create_champions_league_simulation()
