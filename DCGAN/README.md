# _[Deep Convolution GAN](https://arxiv.org/pdf/1511.06434.pdf)_

<br/>

## Results

**_Training (200 Epochs)_**  
![furniture_training](https://user-images.githubusercontent.com/67945103/124582809-54d60800-de8d-11eb-9b5e-ee8f7067d0cb.gif)
![cats_training](https://user-images.githubusercontent.com/67945103/124606436-d89cee00-dea7-11eb-8edf-9621000e7ee5.gif)

<br/>

**_Generated Image_**  
![generated_images_furniture](https://user-images.githubusercontent.com/67945103/124636456-84a00280-dec3-11eb-8f4f-7b48906b7ce5.png)
![Generated_images_cats](https://user-images.githubusercontent.com/67945103/124636044-13604f80-dec3-11eb-97a0-ce6e0a1c071d.png)

<br/>

**_Real Images_**  
![real_images_furniture](https://user-images.githubusercontent.com/67945103/124636230-4571b180-dec3-11eb-9308-94a022aa6109.png)
![real_images_cats](https://user-images.githubusercontent.com/67945103/124636051-14917c80-dec3-11eb-9f36-77fc418247bf.png)

<br/>

## Graphs
![Learning Graphs](https://user-images.githubusercontent.com/67945103/124606279-b30fe480-dea7-11eb-89c4-b62e9e19fe16.png)

- **_Orange : Furniture_**
- **_Blue : Cats_**

<br/>

## Dataset 

- [Furniture](https://drive.google.com/file/d/15aSt3tgJQquWap0fzweWwYq0OEUkrR1G/view?usp=sharing)
- [Cats](https://drive.google.com/file/d/1fquqtosXXSaoMZJVXPWcJDVQdkKgIcBE/view?usp=sharing)
```text
   ├── data
       ├── furniture_images.npy
       └── cats_images.npy 
```

```furniture_images.npy``` has 13096 furniture images without labels   
```cats_images.npy``` has 15747 cat face images without labels

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
