# _[Vanilla GAN](https://arxiv.org/pdf/1406.2661.pdf)_

<br/>

## Results

**_Training (200 Epochs)_**  
![mnist_training](https://user-images.githubusercontent.com/67945103/124488902-5c44d500-ddeb-11eb-93fc-da9e37207fb2.gif)
![fashion_mnist_training](https://user-images.githubusercontent.com/67945103/124488565-f8baa780-ddea-11eb-89c8-8cd936862fa6.gif)

<br/>

**_Generated Image_**  
![generated_images_mnist](https://user-images.githubusercontent.com/67945103/124448990-f8f07e00-ddbd-11eb-8ffa-4e428d5ad689.png)
![Generated_images_fashion_mnist](https://user-images.githubusercontent.com/67945103/124488621-053f0000-ddeb-11eb-8186-bd43cb1a5c11.png)

**_Real Images_**

![real_images_mnist](https://user-images.githubusercontent.com/67945103/124446616-8c747f80-ddbb-11eb-8d91-ec89c329f70c.png)
![real_images_fashion_mnist](https://user-images.githubusercontent.com/67945103/124488594-ff491f00-ddea-11eb-8fbd-f88f263adc5a.png)

<br/>

## Graphs
![Learning Graphs](https://user-images.githubusercontent.com/67945103/124487902-32d77980-ddea-11eb-990c-8889556f91b6.png)

- **_Orange : MNIST_**
- **_Blue : Fashion MNIST_**

<br/>

## Dataset 

- [MNIST](https://drive.google.com/file/d/1-8vLwENbumOMHJD5ZS8hBmSIW_flT6tn/view?usp=sharing)
- [Fashion MNIST](https://drive.google.com/file/d/1-8vLwENbumOMHJD5ZS8hBmSIW_flT6tn/view?usp=sharing)
```text
   ├── data
       ├── mnist_images.npy
       └── fashion_mnist_images.npy 
```

```mnist_images.npy``` has 60000 MNIST images without labels   
```fashion_mnist_images.npy``` has 60000 Fashion MNIST images without labels

<br/>

## How to Run

Install required libraries.
```shell
pip install -r requirements.txt
```

<br/>

Train your model with ```mnist_images.npy``` dataset.
```shell
python main.py
```

<br/>

_Generated images, trained weights, tensorboard logs will be stored in ```checkpoints/```._  
_If you want to change model structure, revise ```config/model.yaml```._  
_If you want to change hyper-parameters, revise ```cofig/train.yaml```._
