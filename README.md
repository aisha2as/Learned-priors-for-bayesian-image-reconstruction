# **Learned Priors for Image Reconstruction**
This repository contains the code and datasets for implementing and evaluating Adversarial Convolutional Regularizer (ACR) models for Bayesian image reconstruction. The models aim to enhance image quality by utilizing convolutional neural networks (CNNs) for regularization, applied specifically to noisy image data, such as those generated from medical imaging techniques like CT scans.
## **Dataset** 
The zipped dataset file contains four primary directories:
- **train_data**: Consists of 500 true images and their corresponding FBP-reconstructed images. Used for training the model.
- **test_data**: Contains 100 true images and their corresponding FBP-reconstructed images. Used for testing the regularizer.
- **dataset_for_optimization**: Includes 20 true images. Specifically used in the Gradient Descent (GD) algorithm.
- **dataset_for_testing_GD**: Contains images used to test the performance of the Gradient Descent algorithm.
All datasets were generated using the code provided in the notebooks within this repository. The FBP-reconstructed images were generated from sinograms with 90 projections, and Gaussian noise with a mean of 0 and a variance of 0.5 was applied.
# **Project Structure**
## **Notebooks**:
-	**model_3_convolution_layers_data_used_90_projections.ipynb**: Jupyter notebook for training the 3-layer convolutional model.
-	**model_6_convolution_layers_data_used_90_projections.ipynb**: Jupyter notebook for training the 6-layer convolutional model.
## **Setup and Requirements**
The code in this repository uses the Core Imaging Library (CIL), which provides advanced tools for tomographic imaging. You can find detailed installation instructions for CIL [here](https://github.com/TomographicImaging/CIL#installation-of-cil).
### Libraries and Tools Used
•	PyTorch
•	NumPy
•	Matplotlib
•	Core Imaging Library (CIL) – for tomographic imaging and reconstruction tasks
## **Using Google Colab for Notebooks**
This project requires a GPU to run efficiently, especially when training convolutional neural networks and performing image reconstruction. We recommend using Google Colab with a T4 GPU to execute the Jupyter notebooks in this repository.
### To use Google Colab:
1.	Open the Colab notebook (click on the .ipynb files in the repository).
2.	Once the notebook is loaded, navigate to Runtime > Change runtime type.
3.	Select GPU from the Hardware accelerator dropdown, ensuring that it selects a T4 GPU if available.
4.	Click Save and proceed with running the notebook cells.
Alternatively, you can run the notebooks directly on a local machine if you have a CUDA-enabled GPU.
## **How to Run the Notebooks**
1.	Clone the repository: git clone https://github.com/aisha2as/Learned-priors-for-bayesian-image-reconstruction
2.	Upload the notebook and dataset to Google Colab 
3.	Run the cells to train the models or test the regularizer on your dataset.
## **Pre-trained Models**
This repository includes pre-trained models for both the 3-layer and 6-layer convolutional regularizer models, saved as .pth files. These models were trained using data with 90 projections and a regularization term (lambda_gp) set to specific values after hyperparameter tuning.
-**3-layer Conv Model**: best_model_3_conv_layer_90_proj_reg_lambda.pth
	This model was trained with 90 projections and the regularization parameter lambda_gp=0.1 and learning_rate=0.1.
-**6-layer Conv Model**: best_model_6_conv_layer_90_proj_reg_lambda.pth
	This model was trained with 90 projections and a regularization parameter lambda_gp=0.1 and learning_rate=0.001.
You can load these models directly in your scripts or notebooks to reproduce the results, or fine-tune them for further experiments.
## **Results**
After running the notebooks, you will generate:
•	Training and validation loss curves: Plots showing model performance during training.
•	FBP and optimized image comparisons: Histograms and image outputs comparing the quality of FBP reconstructed images and those optimized by the regularizer.
•	MSE : Analysis of Mean Squared Error (MSE) between true images and their reconstructions.
All these plots are automatically displayed during execution.

## **Acknowledgments**
This project utilized the **Core Imaging Library (CIL)** for Filtered Back Projection reconstruction and Gradient Descent optimization. Additionally, this work was inspired by the regularization techniques described in **Mukherjee et al. (2021)**, which provided the foundation for the model architecture in this repository. The following sources were referenced:
-	E. Pasca, J. S. Jørgensen, E. Papoutsellis, E. Ametova, G. Fardell, K. Thielemans, L. Murgatroyd, M. Duff, and H. Robarts (2023)
Core Imaging Library (CIL)
Zenodo [software archive]
DOI: https://doi.org/10.5281/zenodo.4746198
-	J. S. Jørgensen, E. Ametova, G. Burca, G. Fardell, E. Papoutsellis, E. Pasca, K. Thielemans, M. Turner, R. Warr, W. R. B. Lionheart, and P. J. Withers (2021)
Core Imaging Library - Part I: A versatile Python framework for tomographic imaging.
Phil. Trans. R. Soc. A. 379: 20200192.
DOI: https://doi.org/10.1098/rsta.2020.0192
Code: https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I 
-	Mukherjee, S., Dittmer, S., Shumaylov, Z., Lunz, S., Öktem, O., & Schönlieb, C.-B. (2021). Learned convex regularizers for inverse problems. arXiv preprint arXiv:2008.02839.
