import numpy as np

from src.tree import DecisionTree

x = np.array([[1.0, 2.0],[2.0,1.0],[3.0,3.0],[6.0,2.0],[7.0,1.0],[8.0,3.0]])
y = np.array([0,0,0,1,1,1])

# train tree

tree = DecisionTree(impurity = "gini", max_depth=3, min_split=2)

tree.fit(x,y)

predictions = tree.predict(x)
print("Training Accuracy:", np.mean(predictions == y))

# inspect the root node
print("\nRoot node:")
print("Feature:", tree.root.feature)
print("Threshold:", tree.root.threshold)

# test on new observations
x_new = np.array([
    [1.5, 2.5],
    [7.5, 2.5],
    [4.0, 1.0]
])

new_predictions = tree.predict(x_new)

print("\nNew observations:")
print(x_new)

print("\nNew predictions:")
print(new_predictions)