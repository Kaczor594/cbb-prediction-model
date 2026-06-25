# Symmetric Feature Engineering — 4-Variant Comparison


## Comparison Table


|    Variant |     AUC |   Brier |  LogLoss |    Acc |   N-AUC |  N-Brier |  N-Acc |  Sym Max |  Sym Mean |
|------------|------------|------------|------------|------------|------------|------------|------------|------------|------------|
|     Var A1 |  0.7961 |  0.1844 |   0.5430 | 0.7139 |  0.6915 |   0.2205 | 0.6427 | 0.189597 |  0.031282 |
|     Var A2 |  0.7971 |  0.1839 |   0.5417 | 0.7139 |  0.6932 |   0.2191 | 0.6440 | 0.160763 |  0.031238 |
|      Var B |  0.7982 |  0.1837 |   0.5416 | 0.7162 |  0.6995 |   0.2162 | 0.6396 | 0.208334 |  0.031943 |
|      Var C |  0.7991 |  0.1829 |   0.5395 | 0.7187 |  0.6975 |   0.2162 | 0.6466 | 0.178917 |  0.031196 |
|   Baseline |  0.7689 |  0.1845 |   0.5439 | 0.7145 |  0.6992 |   0.2183 | 0.6483 | 0.493633 |  0.168353 |

## Symmetry Verification


Max |P(A→B) + P(B→A) − 1| across 500 random test games.
Symmetric variants should show < 0.001; baseline shows positional bias.

## Variant Descriptions


- **A1**: Fully blind — no home/away signal at all
- **A2**: neutral_site only — knows 'this is a neutral game'
- **B**: is_home + neutral_site — knows which team is home
- **C**: is_home + is_away — separate home/away effects (neutral_site dropped)
- **Baseline**: Original v4 model with home_*/away_* features, no augmentation

All symmetric variants use data augmentation (both team orderings in training).
Differentials use uniform team_a − team_b convention.