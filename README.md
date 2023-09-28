### This is the code for the experimental part of my Bachelor's Thesis, which can be read here:  [thesis.pdf](https://github.com/eduardpauliuc/fedpix/files/12746011/Thesis_Eduard_Pauliuc.pdf)

![system](https://github.com/eduardpauliuc/fedpix/assets/41806656/9b2d4069-dcd2-42f7-89cd-711a56bc80ba)

# Thesis overview

## Objective
The objective of this thesis is to answer whether Federated Learning is a practical solution to the scenario of multiple clients willing to collaborate in order to train a stat-of-the-art conditional generative adversarial network on sensitive data, without compromising its privacy. The focus points are finding the most adequate architecture for a distributed scenario, observing performance differences (in terms of resulting model quality, communication costs and set-up overhead) and analysing the privacy of the data during and after the training process.

## Original contributions
While this research is not the first to explore Federated Learning outcomes, it stands out by focusing on practical implementations. There is a wealth of existing research on various algorithms and strategies, but many models trained with Federated Learning remain in the experimental phase and have not analyzed real-world usage.

The model used for evaluation in this work, Pix2Pix, a large Conditional Generative Adversarial Network, is known for its useful medical applications. However, it has not yet been trained in a federated scenario, in order to analyze the performance loss and to determine whether it is suitable for a multi-party training collaboration. This thesis aims to find whether current frameworks and cloud technologies provide the infrastructure required to train a large conditional GAN models in a federated method without major impact on model performance or training complexity.

## Experiments
The aim of the experimental part of this work is to determine whether federated learning is a practical solution for entities aiming to collaboratively develop high-quality generative models, all while safeguarding their data against potential privacy breaches. 
The use case chosen for this experiment is one of the use cases presented in the Pix2Pix paper, where the model is trained to generate aerial photo from a map image. 

The data set was provided by the authors of the paper. It consists of 2196 pairs of images from Google Maps, each having dimensions of 600 pixels by 600 pixels, as both aerial photo and map format. The images are of the region of New York City, split into train and test about the median latitude of the sampling region, with no training pixels appearing in the test set.

This particular use case does not use sensitive data, but it was chosen so that it can be compared with the original results when trained with the same hyper-parameters.

The scenario considered in this work is of three entities with the same amount of images. The data is split randomly into three batches, the data therefore is independent and identically distributed (IID). Non-IID scenarios should also be analyzed in future work. 

Implementation references:
* Pix2Pix
    * [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)
    * [htps://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix](https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/GANs/Pix2Pix)
* Federated Learning in Azure
    * [https://github.com/Azure/medical-imaging/tree/main/federated-learning/pneumonia-federated](https://github.com/Azure/medical-imaging/tree/main/federated-learning/pneumonia-federated)
 
### Centralized scenario
The first naive solution when multiple parties wish to collaborate on a shared model is to gather all the data in one central site and train the model on the whole set of images. Clearly, this approach compromises all the data, or at least some part of it, if the central entity is one of the participating parties. However, this method was chosen to be tested in order to analyze the performance and compare it with further, more secure methods.

The model was trained for 200 epochs, with batch size of 1 and with random horizontal flipping and color jitter, as mentioned in the original paper. Each epoch trains the model over the whole data set of 1096 training images. The training was performed with $0.0002$ learning rate, and Adam optimizer parameters $\beta_1 = 0.5$ and $\beta_2=0.999$. These hyper-parameters will also be used in the further scenarios. The expected results of this training was to obtain qualitative generated images, as the ones presented in the original paper.

### Localized scenario
One of the objectives of this work is to determine whether federated learning is worth implementing in a cross-silo scenario. To answer this, a comparison has to be made between the results a participant obtains when training the model with its own data, and the model obtained from federated learning.

Therefore, a training run is required on the data of a single client. The experiment has the same settings as the previous centralized one, only with a third of the data, and it was run using the data of North Europe Site. The results of this run will be shown in the comparison chapter.

### Simulated federated learning
The NVFlare framework is used to implement the federated learning scenarios. In the first experiment, the Proof of Concept (POC) mode of the framework is used. This performs federated learning locally, without TLS (Transport Layer Security) certificates. The server and clients run on separate processes. This mode is recommended for simulating a real-world scenario, making it a favorable choice for this experiment.

Four processes are launched simultaneously. A server and three participating clients, each using a third of the data. The training was performed for 20 rounds, each of 10 local epochs, for a total of 200 training epochs over the whole data set. In this implementation, after training locally, clients send the model weights to the server, which will prove to be a difficult case for adding differential privacy solutions in the following experiment.

### Distributed federated learning with Differential Privacy
The first three scenarios were run on Google Colab. However, running multiple GPU instances in Colab required a premium subscription at the time this work was done. Also, in order to create a scenario closer to reality, another approach was needed.

The solution was to use the Azure Machine Learning platform \cite{azureml}, creating a separate Virtual Machine (VM) for each previous process, the server and the clients. The server was created on a VM without GPU, as the server does not use the GPU for the aggregation operation. The server was created in the North Europe region.

The clients were created in separate Azure ML workspaces, each in a different region. The first client was created in North Europe region, which we will refer to as North Europe Site. The second client was created in the West Europe region, with the name West Europe Site. Lastly, the third client was created in East US and will be referred to as US Site.

## Results
As of the chosen metrics for this work is the FID score, the generator state was saved at epochs $50$, $100$, $150$ and $200$. In the case of federated learning runs, these correspond to round $5$, $10$, $15$ and $20$, respectively. These models were used to generate outputs for each of the evaluation images. These generated collections of images were then used to compute the FID scores, comparing to the ground truth of the evaluation data. The lower the FID score, the better the generator performs.

![fid](https://github.com/eduardpauliuc/fedpix/assets/41806656/a08c6f0d-c24a-4c04-8c83-96c87d51115b)

The other metric used to analyze the models is visual evaluation, as the goal is to produce images indistinguishable from reality by humans. The locally trained models are the farthest from the ground truth, as the model produces detailed but fuzzy images. The images trained in the Federated Learning setting with Differential Privacy are close in quality to the ones generated by the centralized model, albeit less detailed.

These visual results support the results of the FID scores comparison. Federated training produces better results than training locally, but there is a loss in quality compared to the centralized model.

![compare1](https://github.com/eduardpauliuc/fedpix/assets/41806656/434d4ea7-a70a-4449-8c87-fae3087c223f)
