import pandas as pd
import numpy as np
dataset = pd.read_csv('pgm4.csv', names=['outlook','temp','humidity','wind','class'])
def entropy(target_col):
 elements,counts = np.unique(target_col,return_counts=True)
 entropy=0.0
 for i in range(len(elements)):
    entropy+=(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts))
 return entropy

def gain(data,split_attribute,target="class"):
 total_entropy = entropy(data[target])
 vals,counts= np.unique(data[split_attribute],return_counts=True)
 Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute]==vals[i]).dropna()[target])

for i in range(len(vals))])
 Information_Gain = total_entropy - Weighted_Entropy
 return Information_Gain

def ID3(data,originaldata,features,target="class",parent_node_class = None):
 if len(np.unique(data[target])) <= 1:
    return np.unique(data[target])[0]
 elif len(features) ==0:
    return parent_node_class
 else:
    parent_node_class = np.unique(data[target])[np.argmax(np.unique(data[target],return_counts=True)[1])]
    item_values=[]
    for feature in features:
        item_values.append(gain(data,feature,target))
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        tree = {best_feature:{}}
        features = [i for i in features if i != best_feature]
    #Grow a branch under the root node for each possible value of the root node feature
    for value in np.unique(data[best_feature]):
        sub_data = data.where(data[best_feature] == value).dropna()
        subtree = ID3(sub_data,dataset,features,target,parent_node_class)
        tree[best_feature][value] = subtree
        return(tree)

tree = ID3(dataset,dataset,dataset.columns[:-1])
print(tree)
testing_data = dataset.iloc[3]
tree = ID3(testing_data,dataset,dataset.columns[:-1])
print("Result of testing data:",tree)