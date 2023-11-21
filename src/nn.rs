use std::iter::zip;

use crate::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool
}

impl Neuron {
    pub fn new(input_size: usize, nonlin: bool) -> Neuron {
        let mut weights = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            weights.push(Value::new(0.0));
        }
        let bias = Value::new(0.0);

        Neuron {
            weights,
            bias,
            nonlin,
        }
    }

    pub fn forward(&self, input: &[Value]) -> Value {
        zip(self.weights.iter(), input.iter())
            .map(|(w, i)| w.mul(i))
            .sum::<Value>()
            .add(&self.bias)
            .relu()
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize, nonlin: bool) -> Layer {
        let mut neurons = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            neurons.push(Neuron::new(input_size, nonlin));
        }

        Layer {
            neurons,
        }
    }

    pub fn forward(&mut self, input: Vec<Value>) -> Vec<Value> {
        self.neurons.iter()
            .map(|n| n.forward(&input))
            .collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layer_dims: &[usize]) -> MLP {
        let mut layers = Vec::with_capacity(layer_dims.len() - 1);
        for i in 0..layer_dims.len() - 1 {
            layers.push(Layer::new(layer_dims[i], layer_dims[i+1], i != layer_dims.len() - 2));
        }

        MLP {
            layers,
        }
    }

    pub fn forward(&mut self, input: Vec<Value>) -> Vec<Value> {
        let mut new_x = input;

        for layer in self.layers.iter_mut() {
            new_x = layer.forward(new_x);
        }

        new_x
    }
}