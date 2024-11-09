# Deep Spatiotemporal Variational Bayes

Pytorch implementation of the IJCAI 2024 paper [A Deep Probabilistic Spatiotemporal Framework for Dynamic Graph
Representation Learning with Application to Brain Disorder Identification](https://www.ijcai.org/proceedings/2024/0592).

Recent applications of pattern recognition techniques on brain connectome classification using functional connectivity (FC) are shifting towards acknowledging the non-Euclidean topology and causal dynamics of brain connectivity across time. In this paper, a deep probabilistic spatiotemporal framework developed based on variational Bayes (DSVB) is proposed to learn time-varying topological structures in dynamic brain FC networks for autism spectrum disorder (ASD) identification. The proposed framework incorporates a spatial-aware recurrent neural network with an attention-based message passing scheme to capture rich spatiotemporal patterns across dynamic FC networks. To overcome model overfitting on limited training datasets, an adversarial training strategy is introduced to learn graph embedding models that generalize well to unseen brain networks. Evaluation on the ABIDE resting-state functional magnetic resonance imaging dataset shows that our proposed framework substantially outperforms state-of-the-art methods in identifying ASD. Dynamic FC analyses with DSVB-learned embeddings reveal apparent group differences between ASD and healthy controls in network profiles and switching dynamics of brain states.

## ABIDE Dataset
We used a C-PAC pipeline (Cameron et al., 2013) preprocessed fMRI dataset from the [Autism Brain Imaging Data Exchange (ABIDE I)](https://fcon_1000.projects.nitrc.org/indi/abide/abide_I.html) open source database (Di Martino et al., 2009), with a sample of 144 subjects (70 with Autism Spectrum Disorder (ASD) and 74 healthy controls (HC)) resting-state fMRI included in this case study. The inclusion criteria (Plitt et al., 2015) are as follows:
- Males with a full-scale IQ > 80
- Ages between 11 and 23
- fMRI acquisition sites: New York University, University of California Los Angeles 1, and University of Utah, School of Medicine

** Note: The preprocessed dataset is located in the `/data` folder, and it includes two files: `power_asd.npy` and `power_td.npy`. These files contain time series signals extracted using the Power et al. brain atlas (Power et al., 2011), which defines 264 Regions of Interest (ROIs). The dataset comprises 70 subjects with autism spectrum disorders (ASD) and 74 typically developed (TD) subjects.

## References:
- Cameron, C.; Sharad, S.; Brian, C.; Ranjeet, K.; Satrajit, G.; Chaogan, Y.; Qingyang, L.; Daniel, L.; Joshua, V.; Randal, B.; Stanley, C.; Maarten, M.; Clare, K.; Adriana, D. M.; Francisco, C.; and Michael, M. 2013. Towards automated analysis of connectomes: The configurable pipeline for the analysis of connectomes (C-PAC). Front. Neuroinform., 7.
- Di Martino, A.; Ross, K.; Uddin, L. Q.; Sklar, A. B.; Castellanos, F. X.; and Milham, M. P. 2009. Functional brain correlates of social and nonsocial processes in autism spectrum disorders: an activation likelihood estimation meta-analysis. Biological psychiatry, 65(1): 63–74.
- Plitt, M.; Barnes, K. A.; and Martin, A. 2015. Functional connectivity classification of autism identifies highly predictive brain features but fallsshort of biomarker standards. NeuroImage Clin., 7: 359–366.
- Power, J. D.; Cohen, A. L.; Nelson, S. M.; Wig, G. S.; Barnes, K. A.; Church, J. A.; Vogel, A. C.; Laumann, T. O.; Miezin, F. M.; Schlaggar, B. L.; and Petersen, S. E. 2011. Functional network organization of the human brain. Neuron, 72(4): 665–678.
