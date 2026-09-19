from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from src.tree import DecisionTree

data = load_breast_cancer()

x = data.data
y = data.target

# change 1 to be malignant tumors and 0 to be benign tumors
y = np.array([int(not bool(x)) for x in y])

#print(x.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
tree = DecisionTree(impurity="gini", max_depth=5, min_split = 2)

tree.fit(X_train, y_train)

training_predictions = tree.predict(X_train)
test_predictions = tree.predict(X_test)

print("Training Accuracy:", np.mean(training_predictions == y_train))

print("\nTesting Accuracy:", np.mean(test_predictions == y_test))

# construct confusion matrix
TP = (y_test == 1) & (test_predictions == 1)
TN = (y_test == 0) & (test_predictions == 0)
FP = (y_test == 0) & (test_predictions == 1)
FN = (y_test == 1) & (test_predictions == 0)

print("TP:", np.sum(TP))
print("TN:", np.sum(TN))
print("FP:", np.sum(FP))
print("FN:", np.sum(FN))

# compute precision, recall, and F1
precision = np.sum(TP) / (np.sum(TP) + np.sum(FP))
recall = np.sum(TP) / (np.sum(TP) + np.sum(FN))
print("Precision:", precision)
print("Recall:", recall)
F1 = (2 * precision * recall) / (precision + recall)
print("F1:", F1)


# compute train and test accuracies for maximum depths 1 through 15
train_accuracies = []
test_accuracies = []

for d in range(1,16):
    tree = DecisionTree(impurity="gini", max_depth=d, min_split=2)
    tree.fit(X_train, y_train)

    training_predictions = tree.predict(X_train)
    test_predictions = tree.predict(X_test)

    train_accuracy = np.mean(training_predictions == y_train)
    test_accuracy = np.mean(test_predictions == y_test)

    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print("\nTraining Accuracy (max depth " + str(d) + "):", np.mean(training_predictions == y_train))
    print("Testing Accuracy (max depth " + str(d) + "):", np.mean(test_predictions == y_test))

# depths = range(1,16)
#
# plt.plot(depths, train_accuracies, label="Train Accuracy")
# plt.plot(depths, test_accuracies, label="Test Accuracy")
#
# plt.xlabel("Max Depth")
# plt.ylabel("Accuracy")
# plt.title("Training and Testing Accuracy vs Max Tree Depth")
# plt.legend()
# plt.show()

# cross validation
rng = np.random.default_rng(42)

indices = np.arange(len(y_train))
rng.shuffle(indices)

X_train_cv = X_train[indices]
y_train_cv = y_train[indices]

num_folds = 5
fold_indices = np.array_split(np.arange(len(y_train_cv)), num_folds)

depth_cv_accuracies = []

for depth in range(1,16):
    validation_accuracies = []
    for k in range(num_folds):
        training_indices = np.concatenate([fold_indices[j] for j in range(num_folds)
                                           if j != k])
        X_train_fold = X_train_cv[training_indices, :]
        X_val_fold = X_train_cv[fold_indices[k], :]
        y_train_fold = y_train_cv[training_indices]
        y_val_fold = y_train_cv[fold_indices[k]]

        tree = DecisionTree(impurity="gini", max_depth=depth, min_split=2)
        tree.fit(X_train_fold, y_train_fold)

        val_predictions = tree.predict(X_val_fold)
        val_acc = np.mean(val_predictions == y_val_fold)

        validation_accuracies.append(val_acc)

    mean_val_acc = np.mean(validation_accuracies)
    print("\nValidation Accuracy:", mean_val_acc)
    depth_cv_accuracies.append(mean_val_acc)

print(depth_cv_accuracies)