import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from pathlib import Path
from collections import Counter

# ------------------------------------------------------------
# 1. Caricamento grafo
# ------------------------------------------------------------
visual_layout_name = {'kk': 'kamada-kawai', 'fr': 'force'}
VISUAL_LAYOUT = 'kk'
data_dir = Path(__file__).parent.parent / "Data/network"
graph_file = data_dir / "user_interaction_graph.graphml"

G = ig.Graph.Read_GraphML(str(graph_file))
# G = G.subgraph([v for v in G.vs if v.degree() > 0])
G = G.connected_components().giant()
edge_weights = G.es["weight"]
n_nodes = G.vcount()

# Passiamo i pesi all'algoritmo layout
# layout = G.layout("kk", weights=edge_weights)
layout = G.layout("fr", weights=edge_weights)
coords = np.array(layout.coords)
scaling_factor = 15.0 
coords = coords * scaling_factor
labels = G.vs["label"]
label_counts = Counter(labels)

# Calcolo e stampa delle metriche
print("\n" + "="*50)
print("METRICHE DELLA GIANT COMPONENT")
print("="*50)
print(f"Numero di nodi: {G.vcount()}")
print(f"Numero di archi: {G.ecount()}")
print(f"Pro: {label_counts['Pro']}")
print(f"Anti: {label_counts['Anti']}")
print(f"Neutri: {label_counts.get('Neutral', 0)}")
print(f"Densità: {G.density():.5f}") # Quanto è connesso il grafo (0-1)
print(f"Diametro: {G.diameter()}")   # Distanza massima tra due nodi
print(f"Clustering Coefficient: {G.transitivity_undirected():.4f}") # Tendenza a formare gruppi chiusi
print(f"Average Path Length: {G.average_path_length():.4f}") # Distanza media tra due utenti
print("="*50 + "\n")

# ------------------------------------------------------------
# 2. Ottimizzazione Archi (LineCollection)
# ------------------------------------------------------------
# Estraiamo le coppie di nodi (sorgente, destinazione) come array numpy
edges = np.array(G.get_edgelist())
# Creiamo un array di coordinate shape (n_edges, 2, 2)
# Ogni riga è: [[x_start, y_start], [x_end, y_end]]
segments = np.stack([coords[edges[:, 0]], coords[edges[:, 1]]], axis=1)

# ------------------------------------------------------------
# 3. Disegno
# ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(16, 12))

# Disegno ultra-veloce degli archi
lc = LineCollection(segments, colors="black", alpha=0.7, linewidths=1, zorder=0)
# lc = LineCollection(segments, colors="red", alpha=1.0, linewidths=8, zorder=3)
ax.add_collection(lc)

# Maschere booleane (già efficienti)
labels = np.array(labels)
is_pro = (labels == "Pro")
is_anti = (labels == "Anti")
is_neutral = (labels == "Neutral")

# Disegno nodi (scatter è già efficiente di suo)
if np.any(is_neutral):
    ax.scatter(coords[is_neutral, 0], coords[is_neutral, 1], s=80, c="#95a5a6", alpha=0.7, edgecolors='none', zorder=1)

if np.any(is_pro):
    ax.scatter(coords[is_pro, 0], coords[is_pro, 1], s=100, c="#2ecc71", alpha=0.95, edgecolors='black', linewidths=1.5, zorder=2)

if np.any(is_anti):
    ax.scatter(coords[is_anti, 0], coords[is_anti, 1], s=100, c="#e74c3c", alpha=0.95, edgecolors='black', linewidths=1.5, zorder=2)

# ------------------------------------------------------------
# 4. Formattazione (Soluzione robusta per i limiti)
# ------------------------------------------------------------
ax.set_title(f"Grafo Giant Component ({visual_layout_name[VISUAL_LAYOUT]} layout)", fontsize=14, fontweight="bold")
ax.axis("off")

# CALCOLO MANUALE DEI LIMITI (per forzare la visibilità di tutto)
# Aggiungi margine per non tagliare i nodi ai bordi
margin = 5
ax.set_xlim(coords[:, 0].min() - margin, coords[:, 0].max() + margin)
ax.set_ylim(coords[:, 1].min() - margin, coords[:, 1].max() + margin)

legend_elements = [
    Patch(facecolor="#2ecc71", edgecolor="black", label="Pro-Brexit"),
    Patch(facecolor="#e74c3c", edgecolor="black", label="Anti-Brexit"),
    Patch(facecolor="#95a5a6", edgecolor="none", label="Neutrale")
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

output_path = data_dir / f"graPhoto_{VISUAL_LAYOUT}.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
print(f"✓ Grafico salvato in: {output_path}")