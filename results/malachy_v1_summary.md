## Spatial-gene benchmark

Gene selection: top 30 genes ranked by KNN-smoothed spatial structure.

### Stripe holdout, MLP

| Method | Mean MSE | Mean MAE | Mean Pearson r |
|---|---:|---:|---:|
| KNN | 0.4301 | 0.2409 | 0.1125 |
| MLP neural field | 0.4132 | 0.2382 | 0.0986 |

Interpretation: spatial gene selection increases the amount of predictable spatial signal. KNN remains stronger on Pearson for stripe holdout, while MLP is slightly better on MSE/MAE.

### Quadrant holdout, GridField

| Method | Mean MSE | Mean MAE | Mean Pearson r |
|---|---:|---:|---:|
| KNN | 0.5610 | 0.2736 | 0.0320 |
| GridField neural field | 0.4513 | 0.2355 | 0.0476 |

Interpretation: GridField beats KNN on the harder quadrant spatial-gene benchmark across Pearson, MSE, and MAE.



# MALACHY v1 preliminary results

Dataset: 10x Visium HD Human Breast Cancer TMA, square_016um  
Input: x,y coordinates  
Output: 100 highly variable genes  
Task: held-out spatial prediction within one 2D tissue section  
Models: KNN, CoordinateMLP, GridField/hash coordinate field  

## Stripe holdout

| Method | Mean MSE | Mean MAE | Mean Pearson r | Median Pearson r |
|---|---:|---:|---:|---:|
| KNN | 0.22206 | 0.12791 | 0.0614 | 0.0337 |
| MLP neural field | 0.20799 | 0.13763 | 0.0700 | 0.0534 |
| GridField neural field | 0.20753 | 0.11905 | 0.0600 | 0.0470 |

Interpretation: MLP has the best Pearson correlation; GridField has the best MSE/MAE.

## Quadrant holdout

| Method | Mean MSE | Mean MAE | Mean Pearson r | Median Pearson r |
|---|---:|---:|---:|---:|
| KNN | 0.25946 | 0.14137 | 0.0195 | 0.0096 |
| MLP neural field | 0.22602 | 0.12971 | 0.0156 | 0.0110 |
| GridField neural field | 0.21345 | 0.10248 | 0.0365 | 0.0306 |

Interpretation: GridField beats both KNN and MLP across all aggregate metrics in the harder quadrant holdout.

## Takeaway

MALACHY v1 runs end-to-end on real Visium HD data. Coordinate neural fields are competitive with KNN, and multiresolution GridField encoding improves performance in the quadrant holdout. Current coordinate-only models still underfit sharp local expression hotspots, motivating spatially variable gene selection, H&E conditioning, and GNN neighborhood context.
