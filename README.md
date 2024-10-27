# Projet 7 : Développez une preuve de concept

## Description

Ce projet vise à comparer les performances des modèles CNN (notamment **Xception**) et des **Vision Transformers (ViT)** pour la classification d'images de races de chiens. Il explore l’efficacité des modèles de transfert d’apprentissage pour la vision par ordinateur.

## Objectifs

1. Comparer les performances entre un modèle CNN et un Vision Transformer.
2. Appliquer le fine-tuning et le transfert d'apprentissage sur un dataset réduit.

## Données

Le projet utilise 827 images issues du **Stanford Dogs Dataset** représentant 5 races de chiens.

## Résultats

- **Xception (CNN)** : 97% de précision après fine-tuning.
- **Vision Transformer (ViT)** :
  - **From scratch** : Performances limitées avec un petit dataset.
  - **Transfert d’apprentissage** : 99% de précision, surpassant Xception.

## Conclusion

Le modèle ViT en transfert d’apprentissage a surpassé le modèle Xception en précision. Des améliorations futures incluent l’augmentation des données et l'exploration de modèles hybrides.
