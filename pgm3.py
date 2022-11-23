import numpy as np
import pandas as pd
data = pd.read_csv('pgm3.csv')
concepts = np.array(data.iloc[:, 0:-1])
target = np.array(data.iloc[:, -1])


def learn(concepts, target):
    specific_h = concepts[0].copy()
    print("Initialization of specific_h and general_h")
    print("Specific Hypothis: ", specific_h)
    general_h = [["?" for i in range(len(specific_h))]
                 for i in range(len(specific_h))]
    print("General Hypothis: ", general_h)
    print("concepts: ", concepts)
    for i, h in enumerate(concepts):
        if target[i] == "yes":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    specific_h[x] = '?'
                    general_h[x][x] = '?'
        if target[i] == "no":
            for x in range(len(specific_h)):
                if h[x] != specific_h[x]:
                    general_h[x][x] = specific_h[x]
                else:
                    general_h[x][x] = '?'

    print("\nSteps of Candidate Elimination Algorithm: ", i+1)
    print("Specific Hypothis: ", i+1)
    print(specific_h, "\n")
    print("General Hypothis:", i+1)
    print(general_h)
    indices = [i for i, val in enumerate(general_h) if val == [
        '?', '?', '?', '?', '?', '?']]
    print("\nIndices", indices)
    for i in indices:
        general_h.remove(['?', '?', '?', '?', '?', '?'])
    return specific_h, general_h

s_final, g_final = learn(concepts, target)
print("\nFinal Specific Hypothis:", s_final, sep="\n")
print("Final General Hypothis:", g_final, sep="\n")
