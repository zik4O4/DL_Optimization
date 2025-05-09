# Optimisation des Modèles de Deep Learning : Pruning et Quantization  
**Auteur** : [Votre Nom]  
**Date** : [Date]  
**Technologies** : TensorFlow, Keras, TensorFlow Model Optimization  

---

## **Description**  
Ce notebook Jupyter démontre comment optimiser un modèle de deep learning pour la reconnaissance de chiffres manuscrits (MNIST) en utilisant deux techniques principales :  
1. **Pruning** : Réduction des connexions redondantes dans le réseau.  
2. **Quantization** : Conversion des poids en nombres de plus faible précision pour accélérer l'inférence.  

L'objectif est de réduire la taille du modèle tout en conservant une bonne précision, ce qui est crucial pour le déploiement sur appareils mobiles ou embarqués.  

---

## **Structure du Notebook**  
1. **Chargement et préparation des données**  
   - Dataset MNIST (60 000 images d'entraînement, 10 000 de test).  
   - Normalisation des pixels (valeurs entre 0 et 1).  

2. **Modèle de base**  
   - Architecture simple : `Flatten → Dense(100, ReLU) → Dense(10, Softmax)`.  
   - Entraînement sur 5 époques → **97,5 % de précision**.  

3. **Optimisation par Pruning**  
   - Suppression de 50 % des poids les moins importants.  
   - Résultat : **97,9 % de précision**, modèle plus léger.  

4. **Optimisation par Quantization**  
   - **Post-entraînement** : Réduction à 8 bits (~85 Ko, -75 % de taille).  
   - **Aware Training** : Simulation pendant l'entraînement (~83 Ko, 97,6 % de précision).  

5. **Comparaison et conclusions**  
   - Tableau comparatif des tailles et performances.  
   - Applications possibles (mobile, IoT, edge computing).  

---

## **Comment Exécuter le Notebook**  
1. **Prérequis** :  
   - Python 3.8+  
   - Librairies :  
     ```bash
     pip install tensorflow tensorflow-model-optimization numpy matplotlib
     ```  

2. **Lancement** :  
   ```bash
   jupyter notebook DL_opt.ipynb
