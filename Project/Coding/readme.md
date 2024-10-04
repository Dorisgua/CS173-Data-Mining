# README

## Data Preprocessing Files
1. **Sel Syria.py** - Filters data from Syria. 
2. **Date to grid mapping.py** - Establishes a date-to-grid mapping dictionary (helps align grid granularity). 

## Data Files
- **GTDdata.csv** - Official dataset from the University of Maryland.
- **GTDdata200.csv** - Subset of the GTD dataset for country code 200 (Syria).
- **GTDdata200_clustered.csv** - Clustered subset of GTD data.
- **GTDdata200_clustered_86_filtered_li.csv** - Data scored using parameters from Li's paper.
- **GTDdata200_clustered_86_filtered_li_grid.csv** - Data with grid labels.
- **GTDdata200_score.csv** - Data with risk assessment scores.
- **event_pairs.csv** - Used to demonstrate the relationship between time and similarity. Columns: `datetime` (time between events), `eventid1`, `eventid2`.
- **distance_edge.csv** - Used to demonstrate the relationship between distance and similarity. Columns: `cluster1`, `cluster2`, `distance(km)`, `normalized_distance`, `1/std_distance`.
- **vector200.csv** - Events retaining 11 values for risk calculation.

## DBSCAN
1. **Cartographic visualization.py** - Visualizes the distribution of events in the GTD dataset. 
2. **Clustering effect.py** - Evaluates WSS and BSS. 
3. **Map cluster to grid.py** - Establishes mapping from clusters to grids. 

## Similarity
1. **validation of similarity(time).py** - Validates that similarity decreases over time.
2. **validation of similarity(distance).py** - Validates that similarity decreases with distance.

## Risk Quantification
1. **Risk Degree.py** - Quantifies risk levels. 
2. **Generate embedding.py** - Generates embeddings from risk scores. 

## Model
1. **linear baseline.py**
2. **LSTM.py**
3. **LSTM+time decay.py**
4. **CNN LSTM.py**
5. **CNN LSTM+time decay.py**
