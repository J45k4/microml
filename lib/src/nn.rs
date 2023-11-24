use std::iter::zip;

use crate::Value;
use crate::create_random_floats;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlin: bool
}

impl Neuron {
    pub fn new(input_size: usize, nonlin: bool) -> Neuron {
        let weights = create_random_floats(input_size).iter().map(|x| Value::new(*x)).collect::<Vec<Value>>();
        let bias = Value::new(0.0);

        Neuron {
            weights,
            bias,
            nonlin,
        }
    }

    pub fn weights(&self) -> &Vec<Value> {
        &self.weights
    }

    pub fn bias(&self) -> &Value {
        &self.bias
    }

    pub fn forward(&self, input: &[Value]) -> Value {
        zip(self.weights.iter(), input.iter())
            .map(|(w, i)| w.mul(i))
            .sum::<Value>()
            .add(&self.bias)
            // .relu()
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

    pub fn neurons(&self) -> &Vec<Neuron> {
        &self.neurons
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
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

    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn forward(&self, input: Vec<Value>) -> Vec<Value> {
        let mut new_x = input;

        for layer in self.layers.iter() {
            new_x = layer.forward(new_x);
        }

        new_x
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::new();

        for layer in self.layers.iter() {
            for neuron in layer.neurons.iter() {
                params.push(neuron.bias());
                for weight in neuron.weights().iter() {
                    params.push(weight);
                }
            }
        }

        params
    }

    pub fn zero_grad(&self) {
        for layer in self.layers.iter() {
            for neuron in layer.neurons.iter() {
                neuron.bias().zero_grad();
                for weight in neuron.weights().iter() {
                    weight.zero_grad();
                }
            }
        }
    }
}