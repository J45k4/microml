use std::iter::zip;

use crate::Value;
use crate::create_random_floats;

#[derive(Debug)]
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
        let o = zip(self.weights.iter(), input.iter())
            .map(|(w, i)| w.mul(i))
            .sum::<Value>()
            .add(&self.bias);
        
        if self.nonlin {
            o.relu()
        } else {
            o
        }
    }
}

#[derive(Debug)]
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

#[derive(Debug)]
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(layer_dims: &[usize]) -> MLP {
        println!("layer_dims: {}", layer_dims.len());
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

#[cfg(test)]
mod tests {
    use crate::cross_entropy_loss;
    use crate::one_hot_encode;
    use crate::softmax;

    use super::*;

    #[test]
    fn test_forward() {
        let mlp = MLP::new(&[1, 1]);

        let input = vec![Value::new(1.0)];
        let output = mlp.forward(input);
        let output_softmax = softmax(&output);
        println!("{:#?}", output[0]);
        let label = one_hot_encode(0, 1);
        let loss = cross_entropy_loss(&output_softmax, &label);
        //    &vec![Value::new(1.0)]);
        loss.backward();


        println!("LOSS: {:#?}", loss);
        println!("{:#?}", output[0]);
        println!("{:#?}", mlp);
        println!("{:#?}", mlp.parameters());
    }
}