
mod nn;
mod loss;
mod value;

pub use nn::*;
pub use loss::*;
use rand::Rng;
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