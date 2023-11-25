
mod nn;
mod loss;
mod value;

use std::iter::zip;

pub use nn::*;
pub use loss::*;
use rand::Rng;
use rand::distributions::Uniform;
pub use value::*;

pub fn create_random_floats(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut floats = Vec::with_capacity(n);

    for _ in 0..n {
        let random_float: f64 = rng.gen(); // Generate a random f64.
        floats.push(random_float);
    }

    floats
}

pub fn softmax(values: &[Value]) -> Vec<Value> {
    let max = values.iter().fold(Value::new(0.0), |a, b| a.max(b));
    let exps = values.iter().map(|v| v.sub(&max).exp()).collect::<Vec<_>>();
    let sum = exps.iter().sum::<Value>();
    exps.iter().map(|v| v.div(&sum)).collect::<Vec<Value>>()
}

pub fn one_hot_encode(label: usize, size: usize) -> Vec<Value> {
    let mut vec = vec![Value::new(0.0); size];
    vec[label] = Value::new(1.0);
    vec
}

pub fn get_predicted_label(softmax_output: &[Value]) -> usize {
    let mut max_value = f64::MIN;
    let mut max_index = 0;

    for (index, value) in softmax_output.iter().enumerate() {
        if value.data() > max_value {
            max_value = value.data();
            max_index = index;
        }
    }

    max_index
}

pub fn calculate_accuracy(y: &[u32], y_hat: &[u32]) -> f64 {
    let mut correct = 0;

    for (y, y_hat) in zip(y, y_hat) {
        if y == y_hat {
            correct += 1;
        }
    }

    correct as f64 / y.len() as f64
}

pub fn he_initialization(size: usize, fan_in: usize) -> Vec<f64> {
    let std_dev = (2.0 / fan_in as f64).sqrt();
    let normal = Uniform::new(-std_dev, std_dev);
    let mut rng = rand::thread_rng();

    (0..size).map(|_| rng.sample(&normal)).collect()
}