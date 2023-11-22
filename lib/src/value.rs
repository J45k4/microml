use std::cell::RefCell;
use std::iter::Sum;
use std::ops::Div;
use std::ops::Sub;
use std::rc::Rc;

#[derive(Debug)]
enum Op {
    Mul,
    Add,
    Pow,
    Relu,
    Log,
    Exp,
    Max
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
                    _ => panic!("Not a bin OP")
                }
            },
            Parent::UnaryOp { op, inner } => {
                let mut inner = inner.borrow_mut();

                match op {
                    Op::Pow => {
                        let inner_grad = self.grad * self.data.powf(self.data - 1.0);
                        inner.backward_with_grad(inner_grad);
                    },
                    Op::Relu => {
                        let inner_grad = if self.data > 0.0 {
                            self.grad
                        } else {
                            0.0
                        };
                        inner.backward_with_grad(inner_grad);
                    },
                    _ => panic!("Not a unary OP")
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

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
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

    pub fn add(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data + other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::BinOp {
                    op: Op::Add,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            })),
        }
    }

    pub fn pow(&self, other: f64) -> Value {
        let new_data = self.inner.borrow().data.powf(other);

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::UnaryOp {
                    op: Op::Pow,
                    inner: self.inner.clone(),
                }
            })),
        }
    }

    pub fn relu(&self) -> Value {
        let new_data = if self.inner.borrow().data > 0.0 {
            self.inner.borrow().data
        } else {
            0.0
        };

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::UnaryOp {
                    op: Op::Relu,
                    inner: self.inner.clone(),
                }
            })),
        }
    }

    pub fn log(&self) -> Value {
        let new_data = self.inner.borrow().data.log2();

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::UnaryOp {
                    op: Op::Log,
                    inner: self.inner.clone(),
                }
            })),
        }
    }

    pub fn exp(&self) -> Value {
        let new_data = self.inner.borrow().data.exp();

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::UnaryOp {
                    op: Op::Log,
                    inner: self.inner.clone(),
                }
            })),
        }
    }

    pub fn max(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data.max(other.inner.borrow().data);

        Value { 
            inner: Rc::new(RefCell::new(Inner {
                data: new_data,
                grad: 0.0,
                parent: Parent::BinOp {
                    op: Op::Max,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            })),
        }
    }

    pub fn backward(&self) {
        self.inner.borrow_mut().backward_with_grad(1.0);
    }
}

impl Sub for Value {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        self.add(&other.mul(&Value::new(-1.0)))
    }
}

impl Sub for &Value {
    type Output = Value;

    fn sub(self, other: Self) -> Value {
        self.add(&other.mul(&Value::new(-1.0)))
    }
}

impl Div for Value {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        self.mul(&other.pow(-1.0))
    }
}

impl Div for &Value {
    type Output = Value;

    fn div(self, other: Self) -> Value {
        self.mul(&other.pow(-1.0))
    }
}

impl Div<&Value> for Value {
    type Output = Value;

    fn div(self, other: &Self) -> Self::Output {
        self.mul(&other.pow(-1.0))
    }
}

impl Sum for Value {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Value::new(0.0), |acc, x| acc.add(&x))
    }
}

// impl Sum for &Value {
//     fn sum<I>(iter: I) -> Self
//     where
//         I: Iterator<Item = Self>,
//     {
//         iter.fold(&Value::new(0.0), |acc, x| &acc.add(&x))
//     }
// }

impl<'a> Sum<&'a Value> for Value {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        iter.fold(Value::new(0.0), |acc, x| acc.add(x))
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

        println!("{:#?}", c);

        // assert_eq!(c.data, 4.0);
    }
}