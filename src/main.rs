use core::error;

use mnist::{Mnist, MnistBuilder};
use ndarray::prelude::*;

fn main() {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .base_path("data")
        .finalize();

    let training_images = mnist.trn_img;
    let training_labels = mnist.trn_lbl;
    let train_images_array = Array2::from_shape_vec(
        (28 * 28, 60_000),
        training_images
            .into_iter()
            .map(|x| x as f64 / 255.0)
            .collect(),
    )
    .unwrap();
    let train_labels_array = Array1::from_shape_vec(60_000, training_labels).unwrap();
    let mut nn = NeuralNetwork::new(28 * 28, 10, vec![16, 16]);
    let first_hidden_layer = &nn.layers[0];
    println!("Training...");
    for epoch in 0..2 {
        for i in 0..1000 {
            let img = train_images_array.column(i).to_owned();
            let label = train_labels_array[i];
            nn.train(&img, label);
        }
    }
    let test_images = mnist.tst_img;
    let test_labels = mnist.tst_lbl;
    let test_images_array = Array2::from_shape_vec(
        (28 * 28, 10_000),
        test_images.into_iter().map(|x| x as f64 / 255.0).collect(),
    )
    .unwrap();
    let test_labels_array = Array1::from_shape_vec(10_000, test_labels).unwrap();
    let mut correct = 0;
    println!("Testing...");
    for i in 0..10_000 {
        let img = test_images_array.column(i).to_owned();
        let label = test_labels_array[i];
        let predicted = nn.predict(&img);
        let max = predicted.iter().enumerate().max_by(|x, y| x.1.partial_cmp(y.1).unwrap()).unwrap().0;
        if max == label as usize {
            correct += 1;
        }
    }
    println!("Accuracy: {}", correct as f64 / 10_000.0);
}
#[derive(Debug)]
pub enum Activator {
    Sigmoid,
    ReLU,
    Softmax,
}

impl Activator {
    fn activate(&self, data: &Array1<f64>) -> Array1<f64> {
        match self {
            Activator::Sigmoid => {
                data.map(|x| sigmoid(*x))
            }
            Activator::ReLU => {
                data.map(|x| relu(*x))
            }
            Activator::Softmax => {
                let sum = data.iter().map(|x| x.exp()).sum::<f64>();
                data.map(|x| x.exp() / sum)
            }
        }
    }

    fn derivative(&self, data: &Array1<f64>) -> Array1<f64> {
        match self {
            Activator::Sigmoid => {
                data.map(|x| sigmoid_derivative(*x))
            }
            Activator::ReLU => {
                data.map(|x| relu_derivative(*x))
            }
            Activator::Softmax => {
                unimplemented!()
            }
        }
    }
}
#[derive(Debug)]
pub enum LayerKind {
    Activation {
        activator: Activator,
    },
    Dense {
        weights: Array2<f64>,
        biases: Array1<f64>,
    }
}

impl From<Activator> for LayerKind {
    fn from(activator: Activator) -> LayerKind {
        LayerKind::new_activation(activator)
    }
}

impl From<(usize, usize)> for LayerKind {
    fn from((input_nodes, output_nodes): (usize, usize)) -> LayerKind {
        LayerKind::new_dense(input_nodes, output_nodes)
    }
}

impl LayerKind {
    fn new_dense(input_nodes: usize, output_nodes: usize) -> LayerKind {
        let weights_array = (0..input_nodes * output_nodes)
            .map(|_| rand::random::<f64>())
            .collect::<Vec<f64>>();
        let biases_array = (0..output_nodes)
            .map(|_| rand::random::<f64>())
            .collect::<Vec<f64>>();
        LayerKind::Dense {
            weights: Array2::from_shape_vec((output_nodes, input_nodes), weights_array).unwrap(),
            biases: Array1::from_shape_vec(output_nodes, biases_array).unwrap(),
        }
    }

    fn new_activation(activator: Activator) -> LayerKind {
        LayerKind::Activation { activator }
    }

    fn forward(&self, inputs: &Array1<f64>) -> Array1<f64> {
        match self {
            LayerKind::Activation { activator } => {
                activator.activate(inputs)
            }
            LayerKind::Dense { weights, biases } => {
                weights.dot(inputs) + biases
            }
        }
    }
}

pub struct Layer {
    inputs: Array1<f64>,
    outputs: Array1<f64>,
    kind: LayerKind,
}

impl Layer {
    fn new<K>(k: K) -> Layer
    where
        K: Into<LayerKind>,
    {
        Layer {
            inputs: Array1::zeros(0),
            outputs: Array1::zeros(0),
            kind: k.into(),
        }
    }

    fn forward(&self, inputs: &Array1<f64>) -> Array1<f64> {
        self.kind.forward(inputs)
    }

    fn forward_train(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        //println!("Kind: {:?}", self.kind);
        self.inputs = inputs.clone();
        //println!("Input dims: {:?}", self.inputs.dim());
        self.outputs = self.forward(inputs);
        //println!("Output dims: {:?}", self.outputs.dim());
        //println!("___");
        self.outputs.clone()
    }

    fn backward(&mut self, errors: Array1<f64>, learning_rate: f64) -> Array1<f64> {
        match &mut self.kind {
            LayerKind::Activation { activator } => {
                //println!("====");
                //println!("Error dim: {:?}", errors.dim());
                //println!("Inputs dim: {:?}", self.inputs.dim());
                //println!("Outputs dim: {:?}", self.outputs.dim());
                //println!("====");
                let derivative = activator.derivative(&errors);
                derivative
            }
            LayerKind::Dense { weights, biases } => {
                // Convert inputs to 2D array
                //println!("pre_inputs_dim: {:?}", self.inputs.dim());
                //println!("pre_errors_dim: {:?}", errors.dim());
                let inputs = self.inputs.to_shape((self.inputs.len(), 1)).unwrap();
                let errors = errors.to_shape((errors.len(), 1)).unwrap();
                let d_w = errors.dot(&inputs.t());
                let d_x = errors.t().dot(weights);
                //println!("weights_dim: {:?}", weights.dim());
                //println!("d_w_dim: {:?}", d_w.dim());
                //println!("inputs_dim: {:?}", inputs.dim());
                //println!("errors_dim: {:?}", errors.dim());
                let new_weights = weights.clone() - learning_rate * d_w;
                *weights = new_weights;
                let d_x = d_x.to_shape(d_x.len()).unwrap().to_owned();
                //println!("d_x_dim: {:?}", d_x.dim());
                //println!("___");
                d_x
            }
        }
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new<L>(inputs: usize, outputs: usize, hidden_layers: L) -> NeuralNetwork
    where
        L: IntoIterator<Item = usize>,
    {
        let mut last_layer_size = inputs;
        let mut layers = Vec::new();
        // Create hidden layers
        for layer_size in hidden_layers {
            layers.push(Layer::new((last_layer_size, layer_size)));
            layers.push(Layer::new(Activator::Sigmoid));
            last_layer_size = layer_size;
        }
        
        // Create output layer
        layers.push(Layer::new((last_layer_size, outputs)));
        layers.push(Layer::new(Activator::Sigmoid));
        NeuralNetwork { layers }
    }

    fn predict(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut outputs = inputs.clone();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    fn forward(&mut self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut outputs = vec![inputs.clone()];
        for layer in &mut self.layers {
            outputs.push(layer.forward_train(&outputs.last().unwrap()));
        }
        outputs.pop().unwrap()
    }

    fn backward(&mut self, errors: &Array1<f64>, learning_rate: f64) {
        let mut errors = errors.clone();
        for layer in self.layers.iter_mut().rev() {
            errors = layer.backward(errors, learning_rate);
        }
    }

    fn train(&mut self, inputs: &Array1<f64>, actual: u8) {
        let predicted = self.forward(inputs);
        let errors = cost_derivative(&predicted, actual);
        self.backward(&errors, 0.1);
    }

    fn train_batch(&mut self, inputs: &Array2<f64>, actuals: &Array1<u8>) {
    }

}

/// sigmoid(x) = 1 / (1 + e^(-x))
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
fn sigmoid_derivative(x: f64) -> f64 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

/// ReLU(x) = max(0, x)
fn relu(x: f64) -> f64 {
    x.max(0.0)
}

/// ReLU'(x) = 1 if x > 0, 0 otherwise
fn relu_derivative(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// cost(a, b) = sum((a - b)^2)
fn cost(predicted: &Array1<f64>, actual: u8) -> f64 {
    let mut cost = Array1::zeros(10);
    cost[actual as usize] = 1.0;
    (predicted - cost).map(|x| x.powi(2)).sum()
}

/// cost'(a, b) = a - b
fn cost_derivative(predicted: &Array1<f64>, actual: u8) -> Array1<f64> {
    let mut cost = Array1::zeros(10);
    cost[actual as usize] = 1.0;
    predicted - cost
}