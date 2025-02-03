# Source Code for Uncovering Hidden Subspaces in Video Diffusion Models Using Re-Identification 

Details on Model-Training of the diffusion models as well as training of the autoencoder can be found at [this repo](https://github.com/HReynaud/EchoNet-Synthetic). 

Training in image space only requires a filelist with relative paths to the datasets. It should be a csv with spltits. We have added an example csv to this repo in `./dataset`. 
For image space training, set mode to image_space and the path to `<config:image_path>/Videos`. 
For latent space training, precompute the entire dataset with [this vae](https://github.com/HReynaud/EchoNet-Synthetic?tab=readme-ov-file) and put it in `<config:image_path>/Latents`.


### Training Image Space (IS) Re-Identification Models: 

Select the correct config file (config_{a4c,psax,dynamic}_is.json): 
    
    python main.py --config "config_dynamic.json"

Have a look at ./experiments for evaluation techniques. 

Downstream tasks evaluation follow the details in [this repo](https://github.com/HReynaud/echonet).