from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
def plot_scatter_tsne(X, y, classes, labels, colors, markers, loc, dir_name, fig_name, random_seed):
    directory = os.path.dirname(dir_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Get the tsne transformed training feature matrix
    X_embedded = TSNE(n_components=2, random_state=random_seed).fit_transform(X)

    # Get the tsne dataframe
    tsne_df = pd.DataFrame(np.column_stack((X_embedded, y)), columns=['x1', 'x2', 'y'])
    print(y)
    # Get the data
    data = {}
    for class_ in classes:
        data_x1 = [tsne_df['x1'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data_x2 = [tsne_df['x2'][i] for i in range(len(tsne_df['y'])) if tsne_df['y'][i] == class_]
        data[class_] = [data_x1, data_x2]

    # The scatter plot
    fig = plt.figure(figsize=(8, 6))

    for class_, label, color, marker in zip(classes, labels, colors, markers):
        data_x1, data_x2 = data[class_]
        plt.scatter(data_x1, data_x2, c=color, marker=marker, s=120, label=label)

    # Set x-axis
    plt.xlabel('x1')

    # Set y-axis
    plt.ylabel('x2')

    # Set legend
    plt.legend(loc=loc)

    # Save and show the figure
    plt.tight_layout()
    plt.savefig(dir_name + fig_name)
    plt.show()
from sklearn.preprocessing import LabelEncoder
target = 'Phase_inshort'
df = pd.read_csv(r'./IM_SS_AM1.csv')

# The LabelEncoder
le = LabelEncoder()
print(df[target])
# Encode categorical target in the combined data
df[target] = le.fit_transform(df[target].astype(str))
df_train = df.iloc[:df.shape[0], :]

df_train1 = df_train.drop(["Phase_inshort"],axis=1)
y_train = df_train[target].values
print(y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(df_train1)
plot_scatter_tsne(X_train,
                  y_train,
                  [0,1,2],
                  ['AM', 'IM',"SS"],
                  ['blue', 'green',"yellow"],
                  ['o', '^','.'],
                  'lower left',
                  'result/figure/',
                  'scatter_plot_baseline.pdf',
                  random_seed=42)