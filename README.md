# NCI-DOE-Collab-Pilot1-Normal_Tumor_Pair_Classifier

### Description:
The Normal Tumor classifier capability (NT3) shows how to train and use a neural network model to classify molecular features (e.g., RNASeq expressions) as either normal or tumor. The RNASeq expressions are provided in Genomics Data Commons (GDC).

### User Community:	
Researchers interested in cancer susceptibility/histology; classification of diseases for oncology; cancer biology 


### Usability:	
The provided untrained model can be used by a data scientist to be trained on their own data, or use the trained model to classify the provided test samples. The provided scripts use data that has been downloaded and normalized from GDC.

### Uniqueness:	
Using machine learning to classify molecular data has been comonly used. This capability shows how neural networks can be used in classification of genomic profiles without downsampling the provided expressions.

### Components:	

Untrained model: 
* Untrained neural network model is defined in nt3.model.json. Also available in yaml format.

Data:
* Processed training and test data in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363). 

Trained Model:
* Trained model is defined by combining the untrained model + model weights.
* Trained model weights are used in inference [nt3.weights.h5](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363)

### Technical Details:
Please refer to this [README](./Pilot1/NT3/README.md)
