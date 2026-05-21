import pandas as pd
import networkx as nx
from pathlib import Path
from collections import defaultdict

# ---------- 1. Paths and data loading ----------
data_dir = Path(__file__).parent.parent / "Data"
data_dir_out = data_dir / "network"
comments_csv = data_dir / "diffusions_comments_labeled.csv"
submissions_csv = data_dir / "diffusions_submissions_labeled.csv"

print("Loading data...")
comments_df = pd.read_csv(comments_csv)
submissions_df = pd.read_csv(submissions_csv)

print(f"Comments: {comments_df.shape[0]}, Submissions: {submissions_df.shape[0]}")

# ---------- 2. Map submission ID -> author ----------
submission_author_map = dict(zip(submissions_df["Submission.ID"], submissions_df["Author"]))

# ---------- 3. Gather leave_probability for each user (all occurrences) ----------
user_probs = defaultdict(list)

# From submissions
for _, row in submissions_df.iterrows():
    author = row["Author"]
    prob = row["leave_probability"]
    user_probs[author].append(prob)

# From comments
for _, row in comments_df.iterrows():
    author = row["Author"]
    prob = row["leave_probability"]
    user_probs[author].append(prob)

# Compute average probability per user
user_avg_prob = {}
for user, probs in user_probs.items():
    user_avg_prob[user] = sum(probs) / len(probs)

# ---------- 4. Assign label based on thresholds ----------
def get_label(prob):
    if prob < 0.25:
        return "Anti"
    elif prob > 0.75:
        return "Pro"
    else:
        return "Neutral"

user_label = {user: get_label(prob) for user, prob in user_avg_prob.items()}

# ---------- 5. Build directed graph ----------
G = nx.DiGraph()

# Add all users as nodes with attributes
for user in user_avg_prob:
    G.add_node(user,
               leave_probability=user_avg_prob[user],
               label=user_label[user])

# Add edges: for each top-level comment (parent is a submission ID)
# Count multiple comments from same source to same target as weight
edge_weights = defaultdict(int)

for _, row in comments_df.iterrows():
    parent = row["Parent.ID"]
    # Only consider if parent is a submission ID (exists in submission_author_map)
    if parent in submission_author_map:
        source = row["Author"]
        target = submission_author_map[parent]
        edge_weights[(source, target)] += 1

# Add edges to graph with weight attribute
for (src, tgt), w in edge_weights.items():
    G.add_edge(src, tgt, weight=w)

# ---------- 6. Graph summary and statistics ----------
print("\n" + "="*50)
print("GRAPH STATISTICS")
print("="*50)
print(f"Number of nodes (users): {G.number_of_nodes()}")
print(f"Number of directed edges (direct comments to submission authors): {G.number_of_edges()}")
print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")

# Label distribution among users
label_counts = pd.Series([user_label[u] for u in G.nodes]).value_counts()
print("\nUser label distribution (based on leave_probability thresholds 0.25/0.75):")
for label, count in label_counts.items():
    print(f"  {label}: {count}")

# Probability distribution summary
probs = list(user_avg_prob.values())
print(f"\nLeave probability stats (all users): min={min(probs):.3f}, max={max(probs):.3f}, mean={sum(probs)/len(probs):.3f}")

# ---------- 7. Export graph for further analysis ----------
# Save as GraphML (preserves node/edge attributes)
graphml_path = data_dir_out / "user_interaction_graph.graphml"
nx.write_graphml(G, graphml_path)
print(f"\nGraph saved as GraphML: {graphml_path}")

# Save edgelist with weights
edgelist_path = data_dir_out / "user_interaction_edgelist.csv"
nx.write_edgelist(G, edgelist_path, delimiter=',', data=['weight'])
print(f"Edge list saved: {edgelist_path}")

# Save node attributes as CSV
nodes_df = pd.DataFrame([
    {"user": node,
     "leave_probability": G.nodes[node]["leave_probability"],
     "label": G.nodes[node]["label"]}
    for node in G.nodes
])
nodes_csv_path = data_dir_out / "user_attributes.csv"
nodes_df.to_csv(nodes_csv_path, index=False)
print(f"Node attributes saved: {nodes_csv_path}")

print("\n✓ Graph construction completed.")