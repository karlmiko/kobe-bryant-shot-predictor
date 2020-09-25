
"""
Karl Michel Koerich, 1631968
Friday, May 18
R. Vincent , instructor
Final Project
"""

from extra_trees import extra_trees
from classifier import data_item
from random import shuffle

fp = open('kobe_data.csv') #Open file
firstLine = fp.readline()
descriptions = firstLine.split(',') #List cointaining the labels.

#Training
possible_values = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
dataset = []

#Stats 1
stats1 = [0, 0, 0, 0] #Pts_2_scored, Pts_3_scored, Pts_2_attemped, Pts_3_attemped
index_pts = descriptions.index('shot_type')

#Stats 2
teams = []
points_per_team = []
index_opponents = descriptions.index('opponent')

#Stats 3
dates = []
points_per_date = []
oppo_per_date = []
index_date = descriptions.index('game_date')

for line in fp: #Iteration over every line of the file.
    fields = line.split(',')

    #Stats 1
    points = 0 #Variable used for Stats 2 and 3.
    if fields[index_pts][:3] == '2PT': #Only the first 3 characters matter ('2PT' or '3PT')
        if fields[index_pts-1] == '1':
            stats1[0] += 1
            points = 2 #Scores 2 points with this shot.
        stats1[1] += 1
    else: # same as if fields[index_pts][:3] == '3PT' since '2PT' and '3PT' are the only possibilities.
        if fields[index_pts-1] == '1':
            stats1[2] += 1
            points = 3 #Scores 3 points with this shot.
        stats1[3] += 1

    #Stats 2 and 3
    if points != 0: #Only if he scores the shot (points == 2 or points == 3).
        oppo = fields[index_opponents]
        date = fields[index_date]
        if oppo not in teams:
            teams.append(oppo)
            points_per_team.append(points)
        else:
            ind = teams.index(oppo)
            points_per_team[ind] += points
        if date not in dates:
            dates.append(date)
            oppo_per_date.append(oppo)
            points_per_date.append(points)
        else:
            ind = dates.index(date)
            points_per_date[ind] += points
    
    #Setting data for training
    number_fields = [] 
    counter = 0
    for str_value in fields[0:-1]: #Will skip the shot_id (last index) since it is irrelevent
        if counter == 5 or counter == 6: #Will skip loc_x (index 5) and lox_y (index 6) because they are irrelevent
            counter +=1
            continue
        try:
            number_fields += [int(str_value)]
        except:
            if str_value not in possible_values[counter]:
                possible_values[counter].append(str_value)
                number_fields += [len(possible_values[counter])-1]
            else:
                number_fields += [possible_values[counter].index(str_value)]
        counter += 1
        
    data = [int(x) for x in number_fields[:12]] + [int(x) for x in number_fields[13:]]
    label = int(number_fields[12]) #Flag index is 12
    dataset.append(data_item(label, data))

print("\nRead {} items from kobe_data.csv".format(len(dataset)))
print("There are {} features per shot.\n".format(len(dataset[0].data)))

############################################

#Functions

def train_data():
    """Train dataset using Extra Trees and prints out confusion table with results."""
    n_correct = 0
    n_tested = 0

    confusion = [0, 0, 0, 0] #True negatives, False negatives, False positives, True positives.
    
    global dataset

    print("\nTraining started...")

    for rounds in range (0, 5): #5-fold random sub-sampling cross validation.

        copy_dataset = dataset.copy()
        shuffle(copy_dataset)
        test_fold = copy_dataset[0:(len(copy_dataset)//5)]
        data_fold = copy_dataset[(len(copy_dataset)//5):]

        classi = extra_trees()
        classi.train(data_fold)

        print("Trainings completed: {} of 5".format(rounds+1))

        for point in test_fold:
            pred = classi.predict(point.data)
            l_point = point.label
            if pred == 0 and l_point == 0: #True negative.
                confusion[0] += 1
            if pred == 0 and l_point == 1: #False negative.
                confusion[1] += 1
            if pred == 1 and l_point == 0: #False positive.
                confusion[2] += 1
            if pred == 1 and l_point == 1: #True positive.
                confusion[3] += 1
            n_tested += 1

    print("\n\nExtra Trees' performance (Confusion Matrix).\n")
    print("{:>20}{:>12}{:>12}".format(" ", "Correct 0", "Correct 1"))
    print("{:>20}{:>12}{:>12}".format("Predicted 0", confusion[0], confusion[1]))
    print("{:>20}{:>12}{:>12}".format("Predicted 1", confusion[2], confusion[3]))

    right = confusion[0]+confusion[3]
    print("\nExtra Trees predicted {} correct answers out of {} tests.".format(right, n_tested))
    perce = (right/n_tested)*100
    print("It predicted {}{} of the shots succesfully.\n".format(str(perce)[:5],"%"))

def print_stats():
    """"Formats and prints the statictics computed based on the previous data that was read."""

    global stats1, teams, points_per_team, points_per_date, dates, oppo_per_date

    #Stats 1
        #Indexes: stats1[Pts_2_scored, Pts_3_scored, Pts_2_attemped, Pts_3_attemped]
    print("\nKobe's performance on 2PT and 3PT shots.\n")
    print("{:>5}{:>12}{:>12}{:>8}".format(" ", "Scored", "Attempted", "Ratio"))
    print("{:>5}{:>12}{:>12}{:>8}".format("2PT", stats1[0], stats1[1], str(stats1[0]/stats1[1])[:5]))
    print("{:>5}{:>12}{:>12}{:>8}".format("3PT", stats1[2], stats1[3], str(stats1[2]/stats1[3])[:5]))

    #Stats 2
    count = 0
    print("\n\nKobe's total points against each opponent team.\n")
    print("{:>30}{:>16}".format("Opponent Team", "Points Scored"))
    for team in teams:
        print("{:>30}{:>16}".format(team, points_per_team[count]))
        count += 1

    #Stats 3
    print("\n\nKobe's best and worst performances.\n")
    print("{:>30}{:>16}{:>16}{:>18}".format("Performance", "Points Scored", "Date", "Opponent Team"))
    ind_max = points_per_date.index(max(points_per_date))
    ind_min = points_per_date.index(min(points_per_date))
    print("{:>30}{:>16}{:>16}{:>18}".format("Best", max(points_per_date), dates[ind_max], oppo_per_date[ind_max]))
    print("{:>30}{:>16}{:>16}{:>18}".format("Worst", min(points_per_date), dates[ind_min], oppo_per_date[ind_min]))

############################################

#Main.

TRAIN = '1'
STATISTICS = '2'
EXIT = '3'
user_choice = 0

while user_choice != EXIT:
    print("\n1. Train\n2. Statistics\n3. Exit")
    user_choice = input("\nType one of the options above (1, 2 or 3) and press enter: ")

    if user_choice[0] == TRAIN:
        train_data()
    elif user_choice[0] == STATISTICS:
        print_stats()
    elif user_choice[0] != EXIT:
        print("Please input a valid option.")

print("\nBye bye!")
