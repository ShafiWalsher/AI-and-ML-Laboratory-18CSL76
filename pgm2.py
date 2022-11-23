graphNodes = {
    'A': [[('B', 1)], [('C', 1), ('D', 1)]],
    'B': [[('E', 1)], [('F', 1)]],
    'C': [[('G', 1)], [('H', 1), ('I', 1)]],
    'D': [[('J', 1)]]
}

heuristic = {'A': 1, 'B': 4, 'C': 2, 'D': 3,
             'E': 6, 'F': 8, 'G': 2, 'H': 0, 'I': 0, 'J': 0}

parent = {}
status = {}
solutionGraph = {}


def getNeighbors(v):
    if v in graphNodes:
        return graphNodes[v]
    else:
        return ''


def getStatus(v):
    return status.get(v, 0)


def setStatus(v, val):
    status[v] = val


def getHeuristic(n):
    return heuristic.get(n, 0)


def setHeuristic(n, value):
    heuristic[n] = value


def findMinChild(v):  # return Minimum Cost and Minimum Cost child node/s
    mincost = 0
    childForMinCost = {}
    childForMinCost[mincost] = []
    flag = True
    for neighborList in getNeighbors(v):
        newcost = 0
        nodeList = []
        for (m, weight) in neighborList:
            newcost += getHeuristic(m)+weight
            nodeList.append(m)
            if flag == True:
                mincost = newcost
                childForMinCost[mincost] = nodeList
                flag = False
            else:
                if newcost < mincost:
                    mincost = newcost
                    childForMinCost[mincost] = nodeList
    return mincost, childForMinCost[mincost]


def aoStar(v, backTrack):
    print("HEURISTIC VALUES :", heuristic)
    print("SOLUTION GRAPH :", solutionGraph)
    print("PROCESSING NODE :", v)
    print("-----------------------------------------------------------------------------------------")
    if getStatus(v) >= 0:
        mincost, childList = findMinChild(v)
        setHeuristic(v, mincost)
        setStatus(v, len(childList))
        solved = True

        for childNode in childList:
            parent[childNode] = v
            if getStatus(childNode) != -1:
                solved = solved & False

        if solved == True:
            setStatus(v, -1)
            solutionGraph[v] = childList

        if v != startNode:
            aoStar(parent[v], True)

        if backTrack == False:
            for childNode in childList:
                setStatus(childNode, 0)
                aoStar(childNode, False)


def printSolution():
    print("TRAVERSE THE GRAPH FROM THE START NODE: " +
          startNode + " TO GET THE GRAPH SOLUTION")
    print("------------------------------------------------------------")
    print(solutionGraph)
    print("------------------------------------------------------------")

    
startNode='A'
aoStar(startNode,False)
printSolution()

