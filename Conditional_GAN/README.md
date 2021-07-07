# _[Vanilla GAN](https://arxiv.org/pdf/1406.2661.pdf)_

<br/>

## Results

**_Training (200 Epochs)_**  
![mnist_training](https://user-images.githubusercontent.com/67945103/124746539-f3309f00-df5b-11eb-9c85-55414d953eef.gif)
![fashion_mnist_training](https://user-images.githubusercontent.com/67945103/124746201-92a16200-df5b-11eb-8220-90cfa644fc26.gif)

<br/>

**_Generated Image_**  
![generated_images_mnist](https://user-images.githubusercontent.com/67945103/124746486-e0b66580-df5b-11eb-9df4-398cef5fc91e.png)
![Generated_images_fashion_mnist](https://user-images.githubusercontent.com/67945103/124746112-7998b100-df5b-11eb-9d5e-1525f6f043a6.png)

<br/>

**_Real Images_**  
![real_images_mnist](https://user-images.githubusercontent.com/67945103/124746338-b49ae480-df5b-11eb-8018-be65a0b90db9.png)
![real_images_fashion_mnist](https://user-images.githubusercontent.com/67945103/124746027-65ed4a80-df5b-11eb-83ff-fa8296173789.png)

<br/>

## Graphs
![Learning Graphs](https://user-images.githubusercontent.com/67945103/124746903-5c181700-df5c-11eb-89f0-51c59adaaba7.png)

- **_Orange : MNIST_**
- **_Blue : Fashion MNIST_**

<br/>

## Dataset 

- [MNIST_Images](https://drive.google.com/file/d/1-8vLwENbumOMHJD5ZS8hBmSIW_flT6tn/view?usp=sharing)
- [MNIST_Labels](https://drive.google.com/file/d/1McLwUVnCb5P2yhzNVztNZtNJ6vYRm9ku/view?usp=sharing)
- [Fashion MNIST_Images](https://drive.google.com/file/d/1-8vLwENbumOMHJD5ZS8hBmSIW_flT6tn/view?usp=sharing)
- [Fashion_MNIST_Labels](https://drive.google.com/file/d/1K0TCWhb-S6VcjKtfjdL1YiA_hGcjWjaa/view?usp=sharing)
```text
   ├── data
       ├── mnist_images.npy
       ├── mnist_labels.npy
       ├── fashion_mnist_images.npy            
       └── fashion_mnist_labels.npy 
```

```mnist_images.npy``` has 60000 MNIST images without labels   
```mnist_labels.npy``` has 60000 MNIST labels which have same sequences with ```mnist_images.npy```  
```fashion_mnist_images.npy``` has 60000 Fashion MNIST images without labels
```fashion_mnist_labells.npy``` has 60000 Fashion MNIST labels which have same sequences with ```fashion_mnist.npy``` 

<br/>

## How to Run

Install required libraries.
```shell
pip install -r requirements.txt
```

<br/>

Train your model.
```shell
python main.py
```

<br/>

_Generated images, trained weights, tensorboard logs will be stored in ```checkpoints/```._  
_If you want to change model structure, revise ```config/model.yaml```._  
_If you want to change hyper-parameters and dataset, revise ```cofig/train.yaml```._
