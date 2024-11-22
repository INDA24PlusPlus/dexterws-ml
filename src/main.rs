use ndarray::prelude::*;

mod data;


fn main() {
    let data = data::MnistData::new();

    let mut nn = NeuralNetwork::new(28 * 28, 10, vec![100, 50]);
    let batch_size = 64;
    println!("Training...");
    nn.train_batch(&data.trn_img, &data.trn_lbl, batch_size, 10);
    println!("Testing...");
    let accuracy = nn.test(&data.tst_img, &data.tst_lbl);
    println!("Accuracy: {}%", accuracy * 100.);
}

fn create_batch(inputs: &Array2<f64>, labels: &Array2<f64>, batch_size: usize) -> Vec<(Array2<f64>, Array2<f64>)> {
    let mut batches = Vec::new();
    for i in 0..inputs.len_of(Axis(0)) / batch_size {
        let start = i * batch_size;
        let end = (i + 1) * batch_size;
        let inputs = inputs.slice(s![start..end, ..]).to_owned();
        let labels = labels.slice(s![start..end, ..]).to_owned();
        batches.push((inputs, labels));
    }
    batches
}

#[derive(Debug)]
pub enum Activator {
    Sigmoid,
    ReLU,
    Softmax,
}

impl Activator {
    fn activate(&self, data: &Array2<f64>) -> Array2<f64> {
        match self {
            Activator::Sigmoid => {
                data.map(|x| sigmoid(*x))
            }
            Activator::ReLU => {
                data.map(|x| relu(*x))
            }
            Activator::Softmax => {
                let mut res = Array2::zeros(data.dim());
                for (i, row) in data.outer_iter().enumerate() {
                    res.row_mut(i).assign({
                        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exp = row.map(|x| (x - max).exp());
                        let sum = exp.sum();
                        &(exp / sum)
                    });
                }
                res
            }
        }
    }

    fn derivative(&self, errors: &Array2<f64>, data: &Array2<f64>) -> Array2<f64> {
        match self {
            Activator::Sigmoid => {
                errors * data.map(|x| sigmoid_derivative(*x))
            }
            Activator::ReLU => {
                let res = errors * data.map(|x| relu_derivative(*x));
                res
            }
            Activator::Softmax => {
                errors.clone()
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
            .map(|_| 2. * rand::random::<f64>() - 1.)
            .collect::<Vec<f64>>();
        let biases_array = (0..output_nodes)
            .map(|_| 2. * rand::random::<f64>() - 1.)
            .collect::<Vec<f64>>();
        LayerKind::Dense {
            weights: Array2::from_shape_vec((input_nodes, output_nodes), weights_array).unwrap(),
            biases: Array1::from_shape_vec(output_nodes, biases_array).unwrap(),
        }
    }

    fn new_activation(activator: Activator) -> LayerKind {
        LayerKind::Activation { activator }
    }

    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        match self {
            LayerKind::Activation { activator } => {
                activator.activate(inputs)
            }
            LayerKind::Dense { weights, biases } => {
                let res = inputs.dot(weights) + biases;
                res
            }
        }
    }
}

pub struct Layer {
    inputs: Array2<f64>,
    kind: LayerKind,
}

impl Layer {
    fn new<K>(k: K) -> Layer
    where
        K: Into<LayerKind>,
    {
        Layer {
            inputs: Array2::zeros((0, 0)),
            kind: k.into(),
        }
    }

    fn forward(&self, inputs: &Array2<f64>) -> Array2<f64> {
        self.kind.forward(inputs)
    }

    fn forward_train(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        //println!("Kind: {:?}", self.kind);
        self.inputs = inputs.clone();
        //println!("Input dims: {:?}", self.inputs.dim());
        self.forward(inputs)
    }

    fn backward(&mut self, errors: Array2<f64>, learning_rate: f64) -> Array2<f64> {
        match &mut self.kind {
            LayerKind::Activation { activator } => {
                //println!("====");
                //println!("Error dim: {:?}", errors.dim());
                //println!("Inputs dim: {:?}", self.inputs.dim());
                //println!("Outputs dim: {:?}", self.outputs.dim());
                //println!("====");
                let d_x = activator.derivative(&errors, &self.inputs);
                d_x
            }
            LayerKind::Dense { weights, biases } => {
                let d_w = self.inputs.t().dot(&errors);
                let d_b = errors.sum_axis(Axis(0));
                let d_x = errors.dot(&weights.t());
                //println!("d_w: {:?}", d_w);
                //std::thread::sleep(std::time::Duration::from_secs(1));
                *weights = weights.clone() - learning_rate * d_w;
                *biases = biases.clone() - learning_rate * d_b;
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
            layers.push(Layer::new(Activator::ReLU));
            last_layer_size = layer_size;
        }
        
        // Create output layer
        layers.push(Layer::new((last_layer_size, outputs)));
        layers.push(Layer::new(Activator::Softmax));
        NeuralNetwork { layers }
    }

    fn predict(&self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = inputs.clone();
        for layer in &self.layers {
            outputs = layer.forward(&outputs);
        }
        outputs
    }

    fn forward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        let mut outputs = inputs.clone();
        for layer in &mut self.layers {
            outputs = layer.forward_train(&outputs);
        }
        outputs
    }

    fn backward(&mut self, errors: &Array2<f64>, learning_rate: f64) {
        let mut errors = errors.clone();
        let mut i = 0;
        for layer in self.layers.iter_mut().rev() {
            i += 1;
            errors = layer.backward(errors, learning_rate);
        }
    }

    fn train(&mut self, inputs: &Array2<f64>, actual: u8) {
        let predicted = self.forward(inputs);
        self.backward(&predicted, 0.1);
    }

    fn train_batch(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>, batch_size: usize, epochs: usize) {
        for e in 0..epochs {
            println!("Epoch: {}", e);
            let batches = create_batch(inputs, labels, batch_size);
            for (inputs, labels) in &batches {
                let predicted = self.forward(inputs);
                let err = predicted - labels.to_owned();
                self.backward(&err, 0.001);
            }
        }
    }

    fn test(&mut self, inputs: &Array2<f64>, labels: &Array2<f64>) -> f64 {
        // Could just put the entire thing in, but that takes 1 billion memory :p
        let test_batches = create_batch(inputs, labels, 500);
        let mut correct = 0;
        let mut total = 0;
        for (inputs, labels) in &test_batches {
            let predicted = self.predict(inputs);
            for (p, l) in predicted.outer_iter().zip(labels.outer_iter()) {
                let p = p.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                let l = l.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
                if p == l {
                    correct += 1;
                }
                total += 1;
            }
        }
        let accuracy = correct as f64 / total as f64;
        return accuracy;
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
fn cost(predicted: &Array2<f64>, actual: u8) -> f64 {
    let mut cost = Array1::zeros(10);
    cost[actual as usize] = 1.0;
    (predicted - cost).map(|x| x.powi(2)).sum()
}

/// cost'(a, b) = a - b
fn cost_derivative(predicted: &Array2<f64>, actual: u8) -> Array2<f64> {
    let mut cost = Array1::zeros(10);
    cost[actual as usize] = 1.0;
    predicted - cost
}