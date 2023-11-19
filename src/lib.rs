use std::{cell::RefCell, rc::Rc};

#[derive(Debug)]
enum Op {
    Mul,
    Add,
    Pow,
    Relu,
}

#[derive(Debug)]
enum Parent {
    None,
    BinOp {
        op: Op,
        left: Rc<RefCell<Inner>>,
        right: Rc<RefCell<Inner>>,
    },
    UnaryOp {
        op: Op,
        inner: Rc<RefCell<Inner>>,
    },
}

#[derive(Debug)]
struct Inner {
    data: f64,
    grad: f64,
    parent: Parent,
}

impl Inner {
    fn backward_with_grad(&mut self, grad: f64) {
        self.grad = grad;
        
        match &self.parent {
            Parent::None => {},
            Parent::BinOp { op, left, right } => {
                let mut left = left.borrow_mut();
                let mut right = right.borrow_mut();

                match op {
                    Op::Mul => {
                        let left_grad = right.data * self.grad;
                        let right_grad = left.data * self.grad;
                        left.backward_with_grad(left_grad);
                        right.backward_with_grad(right_grad);
                    },
                    Op::Add => {
                        left.grad += self.grad;
                        right.grad += self.grad;
                    },
                    Op::Pow => {
                        left.grad += right.data * left.data.powf(right.data - 1.0);
                        right.grad += left.data.powf(right.data) * (right.data * left.data.ln());
                    },
                    Op::Relu => {
                        left.grad += if left.data > 0.0 { 1.0 } else { 0.0 };
                        right.grad += if right.data > 0.0 { 1.0 } else { 0.0 };
                    },
                }
            },
            Parent::UnaryOp { op, inner } => {
                let mut inner = inner.borrow_mut();

                match op {
                    Op::Mul => {
                        inner.grad += inner.data;
                    },
                    Op::Add => {
                        inner.grad += 1.0;
                    },
                    Op::Pow => {
                        inner.grad += inner.data.powf(inner.data - 1.0);
                    },
                    Op::Relu => {
                        inner.grad += if inner.data > 0.0 { 1.0 } else { 0.0 };
                    },
                }
            },
        }
    }
}

#[derive(Debug)]
pub struct Value {
    inner: Rc<RefCell<Inner>>,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value {
            inner: Rc::new(RefCell::new(Inner {
                data,
                grad: 0.0,
                parent: Parent::None,
            })),         
        }
    }

    pub fn mul(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data * other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::BinOp {
                    op: Op::Mul,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            })), 
        }
    }

    // pub fn add(&self, other: &Value) -> Value {
    //     let new_data = self.data + other.data;

    //     Value { 
    //         parent: Parent::BinOp {
    //             left: self,
    //             right: other,
    //         },
    //         grad: 0.0,
    //         data: new_data,
    //     }
    // }

    // pub fn pow (&self, other: &Value) -> Value {
    //     let new_data = self.data.powf(other.data);

    //     Value { 
    //         data: new_data,
    //         parent: Parent::BinOp {
    //             left: self,
    //             right: other,
    //         },
    //         grad: 0.0
    //     }
    // }

    // pub fn relu(&self) -> Value {
    //     let new_data = if self.data > 0.0 {
    //         self.data
    //     } else {
    //         0.0
    //     };

    //     Value { 
    //         data: new_data,
    //         parent: Parent::None,
    //         grad: 0.0
    //     }
    // }

    

    pub fn backward(&self) {
        self.inner.borrow_mut().backward_with_grad(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mul() {
        let a = Value::new(2.0);
        let b = Value::new(2.0);
        let c = a.mul(&b);
        c.backward();

        println!("{:?}", c);

        // assert_eq!(c.data, 4.0);
    }
}