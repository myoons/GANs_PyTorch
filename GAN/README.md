# _[Vanilla GAN](https://arxiv.org/pdf/1406.2661.pdf)_

<br/>

## Results

**_Training (200 Epochs)_**  
![training](https://user-images.githubusercontent.com/67945103/124447128-0ad12180-ddbc-11eb-986d-a531b6195d55.gif)

**_Generated Image_**

![generated_images](https://user-images.githubusercontent.com/67945103/124448990-f8f07e00-ddbd-11eb-8ffa-4e428d5ad689.png)

**_Real Images_**

![real_images](https://user-images.githubusercontent.com/67945103/124446616-8c747f80-ddbb-11eb-8d91-ec89c329f70c.png)

<br/>

## Graphs
![Probability](https://user-images.githubusercontent.com/67945103/124445751-c98c4200-ddba-11eb-9c0e-af778af1e214.png)
![Loss](https://user-images.githubusercontent.com/67945103/124445755-cabd6f00-ddba-11eb-965a-f9c2673c3346.png)
![Learning Rate](https://user-images.githubusercontent.com/67945103/124445761-cb560580-ddba-11eb-868c-adcb4afefbe0.png)

<br/>

## Dataset 

Download [Dataset](https://drive.google.com/file/d/1-8vLwENbumOMHJD5ZS8hBmSIW_flT6tn/view?usp=sharing) and locate in ```data/``` 
```text
   ├── data
       └── mnist_images.npy 
```

```mnist_images.npy``` has 60000 MNIST (hand-written digits) images

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
