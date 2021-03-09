# NCI-DOE-Collab-Pilot1-Normal_Tumor_Pair_Classifier

### Description
The Normal/Tumor classifier capability (NT3) shows how to train and use a neural network model to classify molecular features (such as RNA-Seq expressions) as either normal or tumor. The RNA-Seq expressions are provided in Genomic Data Commons (GDC).

### User Community
Researchers interested in cancer susceptibility/histology; classification of diseases for oncology; cancer biology. 


### Usability
A data scientist can train the provided untrained model on their own data or use the trained model to classify the provided test samples. The provided scripts use data that has been downloaded from GDC and normalized.

### Uniqueness
Researchers have commonly used machine learning to classify molecular data. This capability shows how you can use neural networks in classification of genomic profiles without downsampling the provided expressions. &#x1F534;_**(Question: I added "Researchers" as a guess to avoid passive voice. If it's not correct, then who commmonly uses it?)**_

### Components
* Untrained model: 
  * The untrained neural network model is defined in [nt3.model.json](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363), and is also available in YAML format.
* Data:
  * The processed training and test data are in [MoDaC](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363). 
* Trained model:
  * The trained model is defined by combining the untrained model and model weights.
  * The trained model weights are used in inference [nt3.weights.h5](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363).

### Technical Details
Refer to this [README](./Pilot1/NT3/README.md).
