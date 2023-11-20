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

    pub fn forward(&mut self, input: Vec<f64>) -> f64 {
        // let mut sum = Value::new(0.0);
        // for (i, value) in input.iter().enumerate() {
        //     sum += self.weights[i].mul(value);
        // }
        // sum += self.bias.data();
        // sum
        0.0
    }
}

pub struct MLP {
    layers: Vec<Vec<Value>> 
}

impl MLP {
    pub fn new(layer_dims: &[usize]) -> MLP {
        let mut layers = Vec::with_capacity(layer_dims.len());

        for dim in layer_dims {
            let mut layer = Vec::with_capacity(*dim);
            for _ in 0..*dim {
                layer.push(Value::new(0.0));
            }
            layers.push(layer);
        }

        MLP {
            layers,
        }
    }

    pub fn forward(&mut self, input: Vec<f64>) -> Vec<f64> {
        // for (i, value) in input.iter().enumerate() {
        //     self.layers[0][i].set_data(*value);
        // }

        // for i in 1..self.layers.len() {
        //     for j in 0..self.layers[i].len() {
        //         let mut sum = 0.0;
        //         for k in 0..self.layers[i-1].len() {
        //             sum += self.layers[i-1][k].get_data() * 0.5;
        //         }
        //         self.layers[i][j].set_data(sum);
        //     }
        // }

        let mut output = Vec::new();
        // for value in self.layers[self.layers.len()-1].iter() {
        //     output.push(value.get_data());
        // }
        output
    }
}