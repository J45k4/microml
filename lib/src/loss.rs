use std::iter::zip;

use crate::Value;

pub fn cross_entropy_loss(y: &[Value], y_hat: &[Value]) -> Value {
    zip(y.iter(), y_hat.iter())
        .map(|(y, y_hat)| {
            let clipped_y_hat = y_hat.max(&Value::new(1e-15)).min(&Value::new(1.0 - 1e-15));
            y.mul(&clipped_y_hat.log())
        })
        .sum::<Value>()
        .mul(&Value::new(-1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test() {
        let y = [Value::new(0.5), Value::new(0.1)];
        let y_hat = [Value::new(0.4), Value::new(0.2)];
        let loss = super::cross_entropy_loss(&y, &y_hat);
        println!("{:#?}", loss.data())
    }

    #[test]
    fn test2() {
        let y = [Value::new(0.5), Value::new(0.5)];
        let y_hat = [Value::new(0.0), Value::new(1.0)];
        let loss = super::cross_entropy_loss(&y, &y_hat);
        println!("{:#?}", loss.data())
    }
}