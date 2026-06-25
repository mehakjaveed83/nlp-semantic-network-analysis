# Word Clustering in Classic Literature

**Discovering semantic word relationships from raw text alone** — no pre-trained embeddings, no external language models. Just sentence co-occurrence, positional statistics, a custom hybrid distance metric, and four unsupervised clustering algorithms compared head-to-head.

`Python` `NLTK` `scikit-learn` `SciPy` `NumPy` `Matplotlib`

---

## Why this project

Most NLP projects reach straight for Word2Vec or a transformer embedding. This one deliberately doesn't — it builds a semantic space **from scratch**, using only statistics derived from the corpus itself: how often words appear together, and how far apart they tend to sit. The goal was to see how much real semantic structure can be recovered with nothing but counting, graph theory, and classic clustering — and to compare multiple algorithms honestly, including where they break down.

## Dataset

Three public-domain novels (Project Gutenberg) were concatenated into a single corpus:

| Novel | Author |
|---|---|
| *Dracula* | Bram Stoker |
| *Metamorphosis* | Franz Kafka |
| *Alice's Adventures in Wonderland* | Lewis Carroll |

Chosen for contrasting vocabulary and narrative style — gothic/character-driven, psychological/domestic, and dialogue-heavy/whimsical — to stress-test whether the pipeline could separate distinct literary worlds.

## Pipeline

```
Raw .txt files
   → strip Project Gutenberg headers/footers
   → lowercase, strip punctuation/digits
   → sentence-tokenise (NLTK)
   → word-tokenise, remove stop words
   → top 100 most frequent content words
```

This produced a cleaned corpus of roughly **400k–500k tokens**, reduced to the **100 most frequent content words** (verbs, nouns, adjectives, proper nouns) for distance modelling.

## Custom distance metric

Two complementary signals were computed for every pair of the 100 words:

1. **Sentence co-occurrence** — how many sentences contain both words
2. **Average positional distance** — mean token-index gap between every occurrence of word *i* and word *j*, across the whole corpus

These combine into a hybrid distance:

```
D(i, j) = 1 / (1 + co_occurrence(i, j)) + avg_positional_distance(i, j)
```

Frequent co-occurrence pulls two words closer; large average separation pushes them apart. No embedding model, vector database, or pre-trained weights are involved — the entire distance space is derived from the three novels alone.

## Graph refinement with Dijkstra's algorithm

The 100×100 distance matrix was treated as a **weighted graph** (words = nodes, hybrid distances = edge weights), and Dijkstra's shortest-path algorithm was run from every node to every other node. This produces a refined distance matrix that captures **transitive** relationships — two words that rarely sit in the same sentence can still end up close if they're connected through shared intermediary words. Clustering was then run on **both** the raw and the Dijkstra-refined matrices, to directly compare the effect of graph smoothing.

## Clustering methods compared

| Method | Library | Run on |
|---|---|---|
| K-Means (`k=5`) | scikit-learn | raw distances + Dijkstra-refined distances |
| Agglomerative / Hierarchical (Ward linkage) | SciPy | raw distances + Dijkstra-refined distances |
| Spectral Clustering (RBF kernel) | scikit-learn | raw distances + Dijkstra-refined distances |
| DBSCAN (precomputed metric) | scikit-learn | Dijkstra-refined distances |

Dimensionality reduction (PCA and t-SNE) was used throughout purely for 2D visualisation — clustering itself was always performed on the full distance matrix.

## Results

**K-Means on raw distances** correctly isolates character names from different novels as extreme outliers — `gregor` (Metamorphosis) and `alice` (Wonderland) sit far from the dense main group, while `professor`/`helsing`/`lucy` form a tight *Dracula*-specific corner.

<img src="assets/kmeans_clusters.png" width="650">

**After Dijkstra refinement**, the same character names remain isolated, but the dense core reorganises — words like `would`, `still`, `even`, and `without` (high-frequency function/modal words shared across all three novels) drift toward the centre, away from the novel-specific names.

<img src="assets/kmeans_clusters_dijkstra.png" width="650">

**Hierarchical clustering** (Ward linkage) on both distance matrices shows the same pattern in dendrogram form — a few long branches for character names, and a dense, hard-to-separate cluster of common verbs and function words:

<img src="assets/dendrogram_dijkstra.png" width="650">

## Honest limitations

This project also surfaced two clear failure modes worth being upfront about, rather than only showing the clean results:

- **Spectral clustering collapsed almost everything into a single cluster.** With an RBF kernel on this distance scale, only the most extreme outliers (`alice`, `gregor`/`gutenberg`) separated out — the rest of the 100 words were assigned to one giant cluster. This points to the RBF `gamma` parameter needing tuning against this specific distance distribution rather than using a single global heuristic.
- **DBSCAN labelled every point as noise** at `eps=1.5`. The hybrid/Dijkstra distances live on a scale of thousands, not the 0–2 range `eps=1.5` assumes — a direct lesson in checking a distance matrix's scale before picking density-based hyperparameters.

Both are left in deliberately: a clustering pipeline that only ever shows clean, separated clusters is usually one where the failure cases were quietly dropped.

## What this demonstrates

- Building an NLP feature space **from first principles** — no pre-trained embeddings
- Designing and justifying a **custom distance metric** from two independent signals
- Applying **graph algorithms** (Dijkstra) outside their typical routing context, to NLP
- Running and **honestly comparing** four different unsupervised learning paradigms (centroid-based, hierarchical, graph-spectral, density-based) on the same data
- Diagnosing **why** an algorithm fails on a given distance scale, not just reporting that it does

## Tech stack

`Python` · `NLTK` (tokenisation, stop words) · `NumPy` (distance matrices) · `scikit-learn` (KMeans, SpectralClustering, DBSCAN, PCA) · `SciPy` (hierarchical linkage, dendrograms) · `Matplotlib` (visualisation)

---

*Originally built as coursework for an Artificial Intelligence module; cleaned up here as a standalone portfolio project.*
