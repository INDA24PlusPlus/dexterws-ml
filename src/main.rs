use ndarray::prelude::*;
use mnist::{Mnist, MnistBuilder};

fn main() {
    let mnist = MnistBuilder::new()
        .label_format_digit()
        .training_set_length(60_000)
        .base_path("data")
        .finalize();

    let training_images = mnist.trn_img;
    let training_labels = mnist.trn_lbl;
    let train_images_array = Array2::from_shape_vec((28 * 28, 60_000), training_images.into_iter().map(|x| x as f64 / 255.0).collect()).unwrap();
    let train_labels_array = Array1::from_shape_vec(60_000, training_labels).unwrap();
    let nn = NeuralNetwork::new(28 * 28, 10, vec![100, 50]);
    let mut sum = 0.;
    for i in 0..100 {
        let image = train_images_array.column(i).to_owned();
        let label = train_labels_array[i];
        let prediction = nn.predict(&image);
        let result = prediction.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap();
        if label == result.0 as u8 {
            sum += 1.;
        }
    }

    


}


struct Layer {
    weights: Array2<f64>,
    biases: Array1<f64>,
}

impl Layer {
    fn new(input_nodes: usize, output_nodes: usize) -> Layer {
        let weights_array = (0..input_nodes * output_nodes).map(|_| rand::random::<f64>()).collect::<Vec<f64>>();
        let biases_array = (0..output_nodes).map(|_| rand::random::<f64>()).collect::<Vec<f64>>();
        Layer {
            weights: Array2::from_shape_vec((input_nodes, output_nodes), weights_array).unwrap(),
            biases: Array1::from_shape_vec(output_nodes, biases_array).unwrap(),
        }
    }

    fn forward(&self, inputs: &Array1<f64>) -> Array1<f64> {
        // Weights are shape (784 x 16), inputs are shape (784 x 1), so we need to transpose the weights
        let outputs = self.weights.t().dot(inputs) + &self.biases;
        outputs.map(|&x| sigmoid(x))
    }
}

struct NeuralNetwork {
    layers: Vec<Layer>,
}

impl NeuralNetwork {
    fn new<L>(inputs: usize, outputs: usize, hidden_layers: L) -> NeuralNetwork
    where L: IntoIterator<Item=usize>
    {
        let mut last_layer_size = inputs;
        let mut layers = Vec::new();
        for layer_size in hidden_layers {
            layers.push(Layer::new(last_layer_size, layer_size));
            last_layer_size = layer_size;
        }
        layers.push(Layer::new(last_layer_size, outputs));
        NeuralNetwork { layers }
    }

    fn predict(&self, inputs: &Array1<f64>) -> Array1<f64> {
        let mut outputs = inputs.clone();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

fn mse(true_values: &Array1<f64>, predicted_values: &Array1<f64>) -> f64 {
    // Mean squared error
    let diff = true_values - predicted_values;
    let squared_diff = diff.map(|&x| x * x);
    let sum = squared_diff.sum();
    sum / (true_values.len() as f64)
}