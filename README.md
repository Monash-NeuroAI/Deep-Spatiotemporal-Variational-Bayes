# Deep-Spatiotemporal-Variational-Bayes

Abstract: Recent applications of pattern recognition techniques on brain connectome classification using functional connectivity (FC) neglect the non-Euclidean topology and causal dynamics of brain connectivity across time. 
In this paper, a deep probabilistic spatiotemporal framework developed based on variational Bayes (DSVB) is proposed to learn time-varying topological structures in dynamic brain FC networks for autism spectrum disorder (ASD) identification. 
The proposed framework incorporates a spatial-aware recurrent neural network to capture rich spatiotemporal patterns across dynamic FC network, followed by a fully-connected neural network to exploit these learned patterns for subject-level classification. To overcome model overfitting on limited training datasets, an adversarial training strategy is introduced to learn graph embedding models that generalize well to unseen brain networks. Evaluation on selected subjects from the ABIDE resting-state functional magnetic resonance imaging dataset shows that our proposed framework significantly outperformed state-of-the-art methods in identifying ASD. Dynamic FC analyses with DSVB learned embeddings reveal apparent group difference between ASD and healthy controls in network profiles and switching dynamics of brain states.
