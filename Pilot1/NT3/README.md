### Model Description
The Normal/Tumor classifier capability (Pilot 1 Benchmark 4), commonly known as NT3, is a 1D convolutional network for classifying RNA-Seq gene expression profiles into normal or tumor tissue categories. The network follows the classic architecture of convolutional models with multiple 1D convolutional layers interleaved with pooling layers, followed by final dense layers. The network can optionally use the following layers:
* 1D locally-connected layers in place of convolution layers.
* Dropout layers for regularization.  

It is useful for studying the difference and transformation of latent representation between normal and tumor tissues. The model also acts as a quality control check for synthetically-generated gene expression profiles.

### Data
The model is trained on the balanced 700 matched normal/tumor gene expression profile pairs available from the Genomic Data Commons (GDC). The full set of expression features contains 60,483 float columns from RNA-Seq [FPKM-UQ](https://docs.gdc.cancer.gov/Encyclopedia/pages/HTSeq-FPKM-UQ/) values. This model achieves around 98% classification accuracy. The associated metadata for the samples (such as normal/tumor) can be found in the file [matched_normal_samples.metadata.tsv](https://modac.cancer.gov/searchTab?dme_data_id=NCI-DME-MS01-7372363). 

### Software Setup
To set up the Python environment needed to train and run this model:
1. Install [conda](https://docs.conda.io/en/latest/) package manager.
2. Clone this repository. 
3. Create the environment as shown below.

```bash
   conda env create -f environment.yml -n nt3 
   conda activate nt3 
   ```

To download the processed data needed to train and test the model, and the trained model files:
1. Create an account first on the Model and Data Clearinghouse [MoDaC](https://modac.cancer.gov). 
2. Follow the instructions in the Training section below. 
3. When prompted by the training and test scripts, enter your MoDac credentials.

### Training
To train the model from scratch, execute the script [nt3_baseline_keras2.py](nt3_baseline_keras2.py). This script does the following:
* Reads the model configuration parameters from [nt3_default_model.txt](nt3_default_model.txt).
* Downloads the training data and splits it to training/validation sets.
* Creates and trains the Keras model.
* Saves the best trained model based on the validation accuracy.
* Evaluates the best model on the test dataset. 

Here is example output from running the script:

```bash
   python nt3_baseline_keras2.py
   ...
    Loading data...
    done
    df_train shape: (1120, 60484)
    df_test shape: (280, 60484)
    X_train shape: (1120, 60483)
    X_test shape: (280, 60483)
    Y_train shape: (1120, 2)
    Y_test shape: (280, 2)
    X_train shape: (1120, 60483, 1)
    X_test shape: (280, 60483, 1)
    0 128 20 1
    1 128 10 1
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv1d_1 (Conv1D)            (None, 60464, 128)        2688      
    _________________________________________________________________
    activation_1 (Activation)    (None, 60464, 128)        0         
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 60464, 128)        0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 60455, 128)        163968    
    _________________________________________________________________
    activation_2 (Activation)    (None, 60455, 128)        0         
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 6045, 128)         0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 773760)            0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 200)               154752200 
    _________________________________________________________________
    activation_3 (Activation)    (None, 200)               0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 200)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 20)                4020      
    _________________________________________________________________
    activation_4 (Activation)    (None, 20)                0         
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 20)                0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 2)                 42        
    _________________________________________________________________
    activation_5 (Activation)    (None, 2)                 0         
    =================================================================
    Total params: 154,922,918
    Trainable params: 154,922,918
    Non-trainable params: 0
    _________________________________________________________________
    Train on 1120 samples, validate on 280 samples
    ....
    Epoch 44/400
    loss: 0.1119 - acc: 0.9723 - val_loss: 0.1305 - val_acc: 0.9821
 
```

### Inference
To test the trained model in inference, execute the script [nt3_infer.py](nt3_infer.py). This script does the following:
* Downloads the trained model.
* Downloads the processed test dataset with the corresponding labels.
* Performs inference on the test dataset.
* Reports the accuracy of the model on the test dataset.

Here is example output from running the script:

```bash
   python nt3_infer.py

   Loading data...
   done
   df_train shape: (1120, 60484)
   df_test shape: (280, 60484)
   X_test shape: (280, 60483)
   Y_test shape: (280, 2)
   json Test score: 0.09439048415848188
   json Test accuracy: 0.982142858845847
   json acc: 98.21%

```
