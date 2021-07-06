# _[Deep Convolution GAN](https://arxiv.org/pdf/1511.06434.pdf)_

<br/>

## Results

**_Training (200 Epochs)_**  
![furniture_training](https://user-images.githubusercontent.com/67945103/124582809-54d60800-de8d-11eb-9b5e-ee8f7067d0cb.gif)
![cats_training](https://user-images.githubusercontent.com/67945103/124606436-d89cee00-dea7-11eb-8edf-9621000e7ee5.gif)

<br/>

**_Generated Image_**  
![generated_images_furniture](https://user-images.githubusercontent.com/67945103/124581600-34f21480-de8c-11eb-995f-f1a9a967db6d.png)
![Generated_images_cats](https://user-images.githubusercontent.com/67945103/124605893-590f1f00-dea7-11eb-9a4a-95f4563bfece.png)

**_Real Images_**  
![real_images_furniture](https://user-images.githubusercontent.com/67945103/124581396-0aa05700-de8c-11eb-9756-1c614de22779.png)
![real_images_cats](https://user-images.githubusercontent.com/67945103/124605902-5ad8e280-dea7-11eb-91a5-0f90972254a1.png)

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
