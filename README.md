# Multichannel Texture Analysis
A project towards Efficient &amp; Interpretable Multichannel Texture Image Analysis with Limited Training Data (MSc Thesis, University of Groningen)
Author: Mariya Shumska
Supervisors: Kerstin Bunte, Michael Wilkinson

## Structure
- `alpha_trees`: contain code related to the alpha tree structure (based on C++ the implementation of Jeroen Lammers https://github.com/JeroenLam/Research-Internship-Finding-an-optimal-dissimilarity-measure-for-hierarchical-segmentation)
- `lvq`: contains GMLVQ-based classifiers:
  * `IALVQ` (Image Analysis LVQ with Quadratic Form distance, inspired by  implementation of https://github.com/MrNuggelz/sklearn-lvq)
  * `IAALVQ` (Image Analysis Angle LVQ with Parametrized Angle distance, based on MATLAB implementation of Kerstin Bunte https://github.com/kbunte/LVQ_toolbox/tree/main/algorithms)
  * `GIALVQ` (Globalized IALVQ, charting implementation is based on MATLAB code of Laurens van der Maaten https://lvdmaaten.github.io/drtoolbox/)
- `utils`: helping functions wrt input/output management, preprocessing, segmentation (still in development), visualization
- `demo_roses.py`: demo with roses data set which contains training, charting, and segmentation of an image
- `demo_synthetic_data`: demo of training, visualization, and charting on synthetic 3D toy data
- `data`: at this point contains preprocessed data. To retrieve raw data, consider sources from the report
