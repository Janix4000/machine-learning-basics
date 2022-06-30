# %%
from statistics import mean
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import pandas as pd
from scipy.stats import shapiro
import seaborn as sns



def plot_clustering(X, y, title):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="nipy_spectral", s=0.1)
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title, fontsize=18)
    return fig, ax

def test_models_size(X, max_size: int, metric: Callable, per_size: int=5, verbose: bool=False):
    metrics = []
    for size in range(2, max_size + 1):
        models = (KMeans(n_clusters=size, random_state=random_state).fit(X) for random_state in range(per_size))
        metrics.append(mean(metric(X, model.labels_) for model in models))
        if verbose: print(f'calculated {size=}')
    return metrics

def adaptive_clustering(data, n_init=10, min_cluster_size=5, alpha=0.05):
    data_labels = ['-2137' for _ in range(len(data))]

    def _adaptive_clustering(cluster, indexes, label_prefix):
        if len(cluster) < min_cluster_size:
            for idx in indexes:
                data_labels[idx] = f'{label_prefix}0'
            return

        model = KMeans(n_clusters=2, n_init=n_init)
        model.fit(cluster)

        p1, p2 = model.cluster_centers_[0], model.cluster_centers_[1]
        l2 = np.sum((p1 - p2) ** 2)
        if l2 == 0:
            return
        line_projection = []
        for p3 in cluster:
            t = np.sum((p3 - p1) * (p2 - p1)) / l2
            line_projection.append(p1 + t * (p2 - p1))

        stat, p = shapiro(line_projection)
        if p > alpha:
            for idx, label in zip(indexes, model.labels_):
                data_labels[idx] = f'{label_prefix}{label}'
        else:
            cl1, cl2, idx1, idx2 = [], [], [], []
            for idx, sample, label in zip(indexes, cluster, model.labels_):
                if label == 0:
                    idx1.append(idx)
                    cl1.append(sample)
                else:
                    idx2.append(idx)
                    cl2.append(sample)
            _adaptive_clustering(cl1, idx1, f'{label_prefix}0')
            _adaptive_clustering(cl2, idx2, f'{label_prefix}1')

    _adaptive_clustering(data, np.arange(len(data)), '')

    label_dict = dict()
    new_label = 0
    for label in data_labels:
        if label not in label_dict:
            label_dict[label] = new_label
            new_label += 1

    labels = [label_dict[data_labels[idx]] for idx in range(len(data))]

    return labels

def plot_elbow(scores, name, better, title):
    fig, ax = plt.subplots(figsize=(10, 10))
    xs = range(2, len(ball_dbs) + 2)
    ax.plot(xs, scores, '--')
    ax.scatter(xs, scores)
    ax.set_xticks(xs)
    ax.grid(color='lightgray', linewidth=0.8, axis='x')
    ax.set_xlabel('k - number of clusters', fontsize=20)
    ax.set_ylabel(f'{name} ({better} is better)', fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.set_title(title, fontsize=30)
    return fig, ax

# %%
df = joblib.load('df.joblib')
X, y = df
plot_clustering(X, y, "MNIST data embedded into two dimensions by UMAP")


# %%
load=True

if load: 
    dbs, ss = joblib.load('metrics.joblib')
else:
    dbs = test_models_size(X, 20, davies_bouldin_score, verbose=True)
    ss = test_models_size(X, 20, silhouette_score, verbose=True)

#%% 
plot_elbow(dbs, 'Davies bouldin score', 'more', 'Mnist')

# %%
plot_elbow(ss, 'Silhouette score', 'less', 'Mnist')

# %%
labels = KMeans(n_clusters=7).fit(X).labels_
plot_clustering(X, labels, f"MNIST, 7 clusters\nDavies bouldin score={davies_bouldin_score(X, labels):.2f}\nSilhouette score={silhouette_score(X, labels):.2f}")

# %%
labels = KMeans(n_clusters=10).fit(X).labels_
plot_clustering(X, labels, f"MNIST, 10 clusters\nDavies bouldin score={davies_bouldin_score(X, labels):.2f}\nSilhouette score={silhouette_score(X, labels):.2f}")


# %%
n_init=1
min_cluster_size=5
alpha=0.05
labels = adaptive_clustering(X, n_init=n_init, min_cluster_size=min_cluster_size, alpha=alpha)
n_clusters = np.unique(labels).shape[0]
plot_clustering(X, labels, 
                f"MNIST\n{n_clusters} clusters, $\\alpha$={alpha:.2g}, min clusters size={min_cluster_size}, starting size={n_init}\nDavies bouldin score={davies_bouldin_score(X, labels):.2f}\nSilhouette score={silhouette_score(X, labels):.2f}")

# %%                
n_init=4
min_cluster_size=200
alpha=5e-4
labels = adaptive_clustering(X, n_init=n_init, min_cluster_size=min_cluster_size, alpha=alpha)
n_clusters = np.unique(labels).shape[0]
plot_clustering(X, labels, 
                f"MNIST\n{n_clusters} clusters, $\\alpha$={alpha:.2g}, min clusters size={min_cluster_size}, starting size={n_init}\nDavies bouldin score={davies_bouldin_score(X, labels):.2f}\nSilhouette score={silhouette_score(X, labels):.2f}")


# %%
data = pd.read_csv('players_22.csv')
columns = ['age', 'club_position', 'height_cm', 'weight_kg', 'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 'attacking_short_passing', 'attacking_volleys', 'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control', 'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance', 'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots', 'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure', 'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle', 'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
data_prep = data[columns]

# %%
data_cleaned = data_prep.copy()
pos_enc = LabelEncoder()
pos_enc.fit(data_cleaned['club_position'])
data_cleaned['club_position'] = pos_enc.transform(data_cleaned['club_position'])
data_cleaned = data_cleaned[columns]
data_cleaned = KNNImputer(missing_values=np.nan).fit_transform(data_cleaned)
data_cleaned = StandardScaler().fit_transform(data_cleaned)
# %%
if load: 
    ball_dbs, ball_ss = joblib.load('ball_met.joblib')
else:
    ball_dbs = test_models_size(data_cleaned, 20, davies_bouldin_score, verbose=True)
    ball_ss = test_models_size(data_cleaned, 20, silhouette_score, verbose=True)
    joblib.dump((ball_dbs, ball_ss), 'ball_met.joblib')
    
# %%
plot_elbow(ball_dbs, 'Davies bouldin score', 'more', 'Football')

#%% 
plot_elbow(ball_ss, 'Silhouette score', 'less', 'Football')

# %%
reducer = umap.UMAP(random_state=42)
data_cast = reducer.fit_transform(data_cleaned)

# %%
n_clusters=11
labels = KMeans(n_clusters=n_clusters).fit(data_cleaned).labels_
# plot_clustering(data_cast, labels, f"Football, {n_clusters} clusters\nDavies bouldin score={davies_bouldin_score(data_cleaned, labels):.2f}\nSilhouette score={silhouette_score(data_cleaned, labels):.2f}")

fig, ax = plot_distributions(data_prep, labels)

# %%
plot_clustering(data_cast, data_cleaned[:, 3], f"Football, height")

# %%
res = adaptive_clustering(data_cleaned, n_init=3, min_cluster_size=20, alpha=1e-7)
# %%
print(davies_bouldin_score(data_cleaned, res), np.unique(res).shape)
# %%


# %%

def plot_distributions(df, labels):
    df['labels'] = labels
    df = df.set_index('labels')
    
    n_features = df.columns.shape[0]
    n_clusters = np.unique(labels).shape[0]

    fig, axes = plt.subplots(1, n_features, figsize=(n_features * 6, n_clusters * 2))

    columns = df.columns
    index = df.index
    for column, ax in zip(columns, axes.ravel()):
        sns.boxplot(ax=ax, data=df, x=index, y=column)

    return fig, axes

# %%
fig, ax = plot_clustering(data_cast, res, '')

# %%
