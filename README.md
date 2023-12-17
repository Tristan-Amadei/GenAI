## Main results
The final results are to be found in the folder "Final results", which contains the sample of data generated with the VAE model with a latent space of 3 dimensions, as well as the latent variables sampled from the latent space by the decoder, associated to this generated sample. <br>
This folder also contains the folder "VAE_checkpoint", where are stored the weights of the trained VAE model. In the notebook called "VAE.ipynb", those are the weights that are loaded by default. Those were also the weights used to generated the new datapoints, as mentioned above.

### Other experiments and results
The folder "Other_results" contains the results we got from our experiments with different GAN and VAE models, which ultimately we did not choose as they provided results not as good as the VAE model with a latent space of 3 dimensions. 

### Metrics
The python script "metrics.py" contains the code we used throughout this study, to compute the Anderson-Darling and Absolute Kendall error metrics. 
