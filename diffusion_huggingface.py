'''
Simple diffusion model implementation using Hugging Face's diffusers library.
Acknowledgements: Adapted from [Accelerate resources](https://docs.science.ai.cam.ac.uk/diffusion-models/DDPM/6_hf_diffusers/)
Installed via:
   pip install -r requirements_diffusion.txt
'''

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

# define model
model = UNet2DModel(sample_size=28,        # 28 x 28 input images
                    in_channels=1,         # 1 channel (grayscale)
                    out_channels=1,        # 1 channel (output)
                    layers_per_block=1,    # 1 layer per block
                    block_out_channels=(4, 8, 16), # 3 blocks with 4, 8 and 16 channels
                    down_block_types=(
                        "DownBlock2D",   # a resnet block
                        "DownBlock2D",   # a resnet block
                        "DownBlock2D"    # a resnet block      
                                     ),
                    up_block_types=(
                        "UpBlock2D",     # up sampling block
                        "UpBlock2D",     # up sampling block
                        "UpBlock2D"      # up sampling block
                                   ),
                    num_class_embeds=10, # 10 class embeddings for 10 digits
                    norm_num_groups=2    # 2 groups for nromalization                  
                    )

print(model)

# Load dataset
# dataset = datasets.load_dataset("mnist")
transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize( (0.5,), (0.5,) ) 
                               ] 
                               )

# mnist_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)  
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)


# Initialize the noise scheduler
#.    in the notes this is called beta
noise_scheduler = DDPMScheduler(num_train_timesteps = 200,
                                beta_start = 0.0001, # starting covariance
                                beta_end = 0.02, 
                                beta_schedule = 'linear',
                                prediction_type = 'epsilon' # predict noise
)

optimizer = torch.optim.AdamW( model.parameters(),
                               lr = 1e-3
                             )

num_train_steps = len(train_loader) * 3

# you need cosine similarity to compare how "close" image is to text
lr_scheduler = get_cosine_schedule_with_warmup( optimizer = optimizer,
                                                num_warmup_steps = 50,
                                                num_training_steps = (num_train_steps),
                                               )

# Train the model
def train(model: UNet2DModel,
          train_loader: DataLoader,
          optimizer: optim.Optimizer,
          noise_scheduler: DDPMScheduler,
          lr_scheduler,
          epochs: int,
          ):
  
    # train loop
    for epoch in range(epochs):
        model.train()

        for i, (clean_images, labels) in tqdm(enumerate(train_loader)):
            
            # geneate random noise?
            # actual REAL noise
            noise = torch.rand(clean_images.shape)

            bs = clean_images.shape[0]

            # what labels are these?
            labels = labels

            
            # add noise to the images according to the noise magnitude at each timestep
            timesteps = torch.randint(0, 
                                      noise_scheduler.num_train_timesteps,
                                      (bs,),
                                      device = clean_images.device
                                      ).long()
            
            # noise the images
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            # predict the noise residual
            noise_pred = model(noisy_images, timesteps, labels, return_dict=False)[0]

            # loss between the predicted noise and the true noise
            loss = F.mse_loss(noise_pred, noise)

            # backpropagation
            loss.backward()

            optimizer.step() # update weights

            lr_scheduler.step() # update learning rate

            optimizer.zero_grad() # reset gradients

        # print loss every epoch
        print(f"Epoch: {epoch+1} Loss: {loss.item()}")



# training
train(model = model,
      train_loader = train_loader,
      optimizer = optimizer,
      noise_scheduler = noise_scheduler,
      lr_scheduler = lr_scheduler,
      epochs = 10
      )
             



# sample from the model
def sample(model: UNet2DModel,
           scheduler: DDPMScheduler,
           batch_size: int,
           generator: torch._C.Generator,
           num_inference_steps: int,
           label: int,
) -> np.ndarray:
    
    ''' Generate samples from the diffusion model 
    Args:
        model (UNet2DModel): trained diffusion model
        scheduler (DDPMScheduler): noise scheduler
        batch_size (int): number of images to generate
        generator (torch.Generator): random generator for reproducibility
        num_inference_steps (int): number of denoising steps
        label (int): class label for conditioning
        Returns:
            np.ndarray: generated images
    '''

    image_shape = (batch_size, 1, 28, 28)

    # if label is a list, convert to tensor
    if isinstance(label,list):
        labels = torch.tensor(label)
    else:
        labels = torch.full((batch_size,), label)


    image = torch.rand(image_shape)

    # set scheduler timesteps
    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # 1. predict noise model_output
        model_output = model(image, t, labels).sample

        # 2. compute previous image: x_t -> x_t-1
        image = scheduler.step(model_output, t, image, generator = generator).prev_sample

    image = (image/2 + 0.5).clamp(0, 1) # unnormalize the image
    image = image.permute(0, 2, 3, 1) # BCHW -> BHWC

    return image.detach().numpy() # convert to numpy array




# Generate samples
images = sample(model=model, # trained model
                scheduler = noise_scheduler, # use the same noise scheduler for sampling
                batch_size = 10, # generate 10 images
                generator = torch.manual_seed(1337), # for reproducibility
                num_inference_steps = 200, # number of denoising steps
                label = [0,1,2,3,4,5,6,7,8,9] # generate images of all digits
                )


# plot the generated images
fig, ax = plt.subplots(2, 5)
for i in range(10):
    ax[i // 5, i % 5].imshow(images[i], cmap='gray') # plot in grayscale
    ax[i // 5, i % 5].axis('off') # turn off axis
plt.show()








  