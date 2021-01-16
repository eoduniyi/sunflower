# 🏡 A Letter From Some Greenhouse. 
_By Erick O. Oduniyi_


---
### 💻 🧬 Computational Botany

**I would like to first say:** I am **not** a biologist (I just love biology). I have friends that are biologist. I don't know how to Garden...I have friends...

**This is what I do:** Build computational models of natural phenomena. And because I have friends who love gardening and botany, I also want to participate in their passions (which has mutual benefits for both humans and plants) by "advancing plant science through functional–structural plant modeling." For example, developing methods for automated species identification. 

#### 🔍 🌱 🌼 Classifying *[Iris](https://en.wikipedia.org/wiki/Iris_sibirica)*

**Then, let us begin:** I would like to classify [iris flowers](https://en.wikipedia.org/wiki/Iris_(plant)) based on the _length_ and _width_ measurements of their [sepals](https://en.wikipedia.org/wiki/Sepal) and [petals](https://en.wikipedia.org/wiki/Petal).

> The Iris genus entails about **300 species**, but our program will only classify the following three:
>
> * Iris setosa
> * Iris virginica
> * Iris versicolor

<table>
  <tr><td>
    <img src="https://www.tensorflow.org/images/iris_three_species.jpg"
         alt="Petal geometry compared for three iris species: Iris setosa, Iris virginica, and Iris versicolor">
  </td></tr>
  <tr><td align="center">
    <b>Figure 1.</b> <a href="https://commons.wikimedia.org/w/index.php?curid=170298">Iris setosa</a> (by <a href="https://commons.wikimedia.org/wiki/User:Radomil">Radomil</a>, CC BY-SA 3.0), <a href="https://commons.wikimedia.org/w/index.php?curid=248095">Iris versicolor</a>, (by <a href="https://commons.wikimedia.org/wiki/User:Dlanglois">Dlanglois</a>, CC BY-SA 3.0), and <a href="https://www.flickr.com/photos/33397993@N05/3352169862">Iris virginica</a> (by <a href="https://www.flickr.com/photos/33397993@N05">Frank Mayfield</a>, CC BY-SA 2.0).<br/>
  </td></tr>
</table>

#### 🧠 Creating a neural network model to classify flowers 

<table>
  <tr><td>
    <img src="https://www.tensorflow.org/images/custom_estimators/full_network.png"
         alt="A diagram of the network architecture: Inputs, 2 hidden layers, and outputs">
  </td></tr>
  <tr><td align="center">
    <b>Figure 2.</b> A fully-connected neural network consisting of an input layer (features), two hidden layers, and an output layer (predictions).<br/>
  </td></tr>
</table>

##### 💾 Implementing the neural network in `Swift`

```swift
// Import Swift for TensorFlow Deep Learning Library
import TensorFlow

// Set the # of artificial neurons
let hiddenSize: Int = 10

// Create a model: fully-connected 3 by 10 layered Neural Network (NN)
struct lightIrisModel: Layer {
    var layer1 = Dense<Float>(inputSize: 4, outputSize: hiddenSize, activation: relu)
    var layer2 = Dense<Float>(inputSize: hiddenSize, outputSize: hiddenSize, activation: relu)
    var layer3 = Dense<Float>(inputSize: hiddenSize, outputSize: 3)
		
    // Differentiable Programming! 
    @differentiable
    func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        return input.sequenced(through: layer1, layer2, layer3)
    }
}

// Initialize the NN
var model = lightIrisModel()

// Train the NN
let firstTrainPredictions = model(firstTrainFeatures)
firstTrainPredictions[0..<5]
```
> **Note:** The "iris classification problem" is one of the fundamental machine learning classification excercises. 

> To convert these logits to a probability for each class, use the [softmax](https://developers.google.com/machine-learning/crash-course/glossary#softmax) function:

```swift
softmax(firstTrainPredictions[0..<5])
```

```
[[  0.7787177,  0.19956581, 0.021716468],
 [ 0.78695095,  0.19159174, 0.021457288],
 [ 0.76027817,  0.19241588, 0.047306072],
 [   0.794681,  0.17019537, 0.035123594],
 [  0.8038667,   0.1630488, 0.033084426]]
```

> See this [link](https://deepai.org/machine-learning-glossary-and-terms/softmax-layer#:~:text=The%20softmax%20function%20is%20a,can%20be%20interpreted%20as%20probabilities.) for more information about softmax

.

.

.

**To Be Continued**
