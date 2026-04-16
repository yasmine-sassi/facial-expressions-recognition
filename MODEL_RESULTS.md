# Résultats des Modèles Testés - Facial Emotion Recognition

## 📊 Résumé Exécutif

**Meilleur Modèle:** Exp13 (DenseNet) avec **Accuracy = 0.6571** en validation  
**Date:** Projets d'optimisation PyTorch (2024-2025)  
**Dataset:** FER2013 - 7 classes d'émotions (angry, disgust, fear, happy, neutral, sad, surprise)

---

## 1. Expériences d'Optimisation PyTorch (Exp1-Exp13)

Résultats systématiques du notebook `04_pytorch_optimization.ipynb`

| Exp | Architecture | Learning Rate | Batch Size | Val Accuracy | Temps d'entraînement | Statut             |
| --- | ------------ | ------------- | ---------- | ------------ | -------------------- | ------------------ |
| 1   | ResNet       | 1e-3          | 32         | **0.6553**   | 40.0 min             | ✓ Nouveau meilleur |
| 2   | SE-ResNet    | 1e-3          | 32         | 0.6536       | 39.9 min             | -                  |
| 3   | SE-ResNet    | 5e-4          | 32         | 0.6553       | 50.4 min             | -                  |
| 4   | DenseNet     | 1e-3          | 32         | 0.6014       | 47.5 min             | -                  |
| 5   | DenseNet     | 5e-4          | 32         | 0.5897       | 46.9 min             | -                  |
| 6   | Inception    | 1e-3          | 32         | 0.5376       | 48.9 min             | -                  |
| 7   | Inception    | 5e-4          | 48         | 0.5603       | 16.4 min             | -                  |
| 8   | MobileNetV2  | 1e-3          | 32         | 0.5498       | 28.3 min             | -                  |
| 9   | MobileNetV2  | 5e-4          | 48         | 0.6336       | 36.8 min             | -                  |
| 10  | MobileNetV2  | 5e-4          | 48         | 0.6238       | 35.2 min             | -                  |
| 11  | Advanced CNN | 1e-3          | 32         | 0.5547       | 42.8 min             | -                  |
| 12  | SE-ResNet    | 1e-4          | 16         | **0.6571**   | 40.0 min             | ✓ **MEILLEUR**     |
| 13  | DenseNet     | 1e-3          | 48         | -            | -                    | ⏸ Incomplète       |

### 📈 Insights Exp1-Exp13

- **Meilleure configuration:** Exp12 (SE-ResNet, LR=1e-4, BS=16)
- **Architectures les plus performantes:** ResNet et SE-ResNet
- **LR optimal:** 1e-3 pour la plupart, 1e-4 pour configurations précises
- **Batch Size optimal:** 32 (32 > 16 > 48 dans la plupart des cas)
- **Modèles faibles:** Inception, MobileNetV2 (sauf exp9)

---

## 2. Modèles PyTorch de Base

### Baseline CNN

- **Notebook:** `02_pytorch_baseline.ipynb`
- **Architecture:** 3 blocs de convolution
- **Hyperparamètres:** LR=0.0001, BS=32, 150 epochs
- **Val Accuracy:** **0.5556**
- **Val Loss:** 1.2640
- **Modèle sauvegardé:** `pytorch_baseline_cnn_best.pt`

### Advanced CNN

- **Notebook:** `03_pytorch_advanced.ipynb`
- **Architecture:** 4 blocs de convolution
- **Hyperparamètres:** LR=0.001, BS=32, 100 epochs
- **Val Accuracy:** ~0.57-0.60 (estimé)
- **Modèle sauvegardé:** `pytorch_advanced_cnn_best.pt`

### ResNet (Custom)

- **Notebook:** `02_pytorch_baseline.ipynb` ou `03_pytorch_advanced.ipynb`
- **Architecture:** Residual Network
- **Hyperparamètres:** LR=0.001, BS=32, 100 epochs
- **Val Accuracy:** ~0.60-0.62 (estimé)

---

## 3. Transfer Learning - ResNet50

**Notebook:** `06_resnet50_transfer_learning.ipynb`

| Métrique            | Valeur                                                            |
| ------------------- | ----------------------------------------------------------------- |
| Architecture        | ResNet50 (ImageNet pré-entraîné)                                  |
| Learning Rate       | 1e-4                                                              |
| Batch Size          | 64                                                                |
| Epochs              | 100                                                               |
| Val Accuracy        | Élevée (estimée > 0.70)                                           |
| Test Accuracy       | Arrêt au epoch 53 (early stopping)                                |
| Modèles sauvegardés | `exp_resnet50_transfer_best.pt`, `exp_resnet50_transfer_final.pt` |

⚠️ **Note:** Entraînement arrêté tôt (early stopping) à cause d'un plateau de validation

---

## 4. Modèles Alternatifs et Ensembles

### VGG et Ensemble

**Notebook:** `09_medium_vgg_0.7_resnet50_0.68.ipynb`

- **VGG:** Accuracy = **0.70**
- **ResNet50:** Accuracy = **0.68**
- Approche d'ensemble pour améliorer les résultats

### Baseline VGG

**Notebook:** `baseline_vgg_0.695.ipynb`

- **Accuracy:** **0.695**
- **Architecture:** VGG standard
- Meilleure performance parmi les modèles alternatifs

### FER CNN

**Notebook:** `fer_cnn_0.66.ipynb`

- **Accuracy:** **0.66**
- **Modèle:** CNN générique pour FER
- **Fichier:** `fer_cnn_0.66.pth`

### ResNet50 sur FER2013

**Notebook:** `fer2013-resnet50.ipynb`

- **Accuracy:** ~0.65 (estimée)
- Transfer learning sur FER2013

---

## 5. Fichiers Modèles Sauvegardés

### Meilleurs Modèles

```
notebooks/saved_models/
├── exp12_advanced_lr1e-03_bs32_best.pt       (Val Acc: 0.6571) ✓ MEILLEUR
├── exp1_resnet_lr1e-03_bs32_best.pt         (Val Acc: 0.6553)
├── exp3_seresnet_lr5e-04_bs32_best.pt       (Val Acc: 0.6553)
├── exp2_seresnet_lr1e-03_bs32_best.pt       (Val Acc: 0.6536)
└── exp9_mobilenetv2_lr1e-03_bs32_best.pt    (Val Acc: 0.6336)
```

### Transfer Learning

```
├── exp_resnet50_transfer_best.pt
├── exp_resnet50_transfer_final.pt
```

### Modèles de Base

```
├── pytorch_baseline_cnn_best.pt              (Val Acc: 0.5556)
├── pytorch_advanced_cnn_best.pt
├── pytorch_advanced_cnn_final.pt
├── pytorch_resnet_emotion_best.pt
└── fer_cnn_0.66.pth
```

---

## 6. Comparaison des Architectures

### Classement par Performance

**Tier 1 - Excellent (>0.65)**

1. SE-ResNet (Exp12) - **0.6571** ✓
2. ResNet (Exp1) - **0.6553**
3. SE-ResNet (Exp3) - **0.6553**
4. VGG (Legacy) - **0.695** (meilleur absolu)

**Tier 2 - Bon (0.60-0.65)**

1. SE-ResNet (Exp2) - 0.6536
2. MobileNetV2 (Exp9) - 0.6336
3. DenseNet (Exp4) - 0.6014
4. ResNet50 Transfer - >0.70 (estimé)

**Tier 3 - Moyen (0.55-0.60)**

1. MobileNetV2 (Exp10) - 0.6238
2. Advanced CNN (Exp11) - 0.5547
3. Inception (Exp7) - 0.5603
4. MobileNetV2 (Exp8) - 0.5498
5. Baseline CNN - 0.5556

**Tier 4 - Faible (<0.55)**

1. Inception (Exp6) - 0.5376
2. DenseNet (Exp5) - 0.5897

### Tendances des Hyperparamètres

- **Learning Rate:** 1e-3 et 1e-4 sont optimaux (1e-3 > 5e-4 > 1e-4)
- **Batch Size:** 32 > 16 > 48 (32 généralement meilleur)
- **Temps d'entraînement:** Inverse de la performance parfois (trade-off vitesse/accuracy)

---

## 7. Résumé Statistique

| Métrique                         | Valeur                   |
| -------------------------------- | ------------------------ |
| **Total d'expériences**          | 13+                      |
| **Meilleure accuracy**           | 0.6571 (Exp12 SE-ResNet) |
| **Accuracy moyenne**             | ~0.60                    |
| **Modèles testés**               | 8+ architectures         |
| **Variations d'hyperparamètres** | 20+ combinaisons         |
| **Meilleure approche legacy**    | VGG (0.695)              |

---

## 8. Recommandations

### Pour Production

1. **Première priorité:** Utiliser Exp12 (SE-ResNet, 0.6571) ou VGG legacy (0.695)
2. **Alternative:** Ensemble de Exp12 + VGG pour améliorer la robustesse
3. **Transfer Learning:** Investiguer pourquoi ResNet50 a été arrêté tôt

### Pour Amélioration

1. **Tuning fin:** Tester LR=5e-4 avec BS=32 pour SE-ResNet
2. **Data Augmentation:** Augmenter les données d'entraînement
3. **Ensemble Methods:** Combiner les 3-4 meilleurs modèles
4. **Class Rebalancing:** Utiliser `class_weights.json` existant

### Données Disponibles

- ✓ Données prétraitées: `data/preprocessed/`
- ✓ Poids des classes: `class_mapping.json`, `class_weights.json`
- ✓ Split info: `split_info.json`
- ✓ Modèles sauvegardés: `notebooks/saved_models/`

---

## 9. Fichiers Associés

| Fichier                                         | Description                |
| ----------------------------------------------- | -------------------------- |
| `notebooks/04_pytorch_optimization.ipynb`       | Expériences Exp1-Exp13     |
| `notebooks/06_resnet50_transfer_learning.ipynb` | Transfer learning ResNet50 |
| `notebooks/baseline_vgg_0.695.ipynb`            | Meilleur modèle legacy     |
| `notebooks/fer_cnn_0.66.ipynb`                  | CNN générique              |
| `data/preprocessed/`                            | Données prétraitées        |
| `data/output/`                                  | Modèles finaux émis        |

---

## 10. Notes Importantes

- Les modèles Exp1-Exp13 sont des tentatives de surpasser le baseline VGG (0.695)
- L'accuracy VGG est supérieure aux modèles PyTorch optimisés actuels
- ResNet50 Transfer Learning était très prometteur avant arrêt précoce
- Les données prétraitées sont disponibles pour réentraînement
- Les class weights suggèrent un déséquilibre dans le dataset

---

**Dernière mise à jour:** 2025  
**Dataset:** FER2013 (Facial Expression Recognition 2013)  
**Classes:** anger, disgust, fear, happiness, neutral, sadness, surprise
