use std::iter::zip;

use crate::Value;


pub fn cross_entropy_loss(y: &[Value], y_hat: &[Value]) -> Value {
    zip(y.iter(), y_hat.iter())
        .map(|(y, y_hat)| y.mul(&y_hat.log()))
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
}