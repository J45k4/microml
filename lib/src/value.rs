use std::cell::RefCell;
use std::collections::HashSet;
use std::hash::Hasher;
use std::iter::Sum;
use std::ops::Div;
use std::ops::Sub;
use std::hash::Hash;
use std::rc::Rc;
use std::sync::atomic::AtomicUsize;

static ID_COUNTER: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug)]
enum BinOPType {
    Mul,
    Div,
    Add,
    Sub,
    Max,
    Min
}

#[derive(Debug)]
enum UnaryOPType {
    Log,
    Exp
}

#[derive(Debug)]
enum Parent {
    None,
    BinOp {
        op: BinOPType,
        left: Rc<RefCell<Inner>>,
        right: Rc<RefCell<Inner>>,
    },
    UnaryOp {
        op: UnaryOPType,
        inner: Rc<RefCell<Inner>>,
    },
}

#[derive(Debug)]
struct Inner {
    id: usize,
    data: f64,
    grad: f64,
    parent: Parent,
}

impl Inner {
    fn new(data: f64, parent: Parent) -> Inner {
        Inner {
            id: ID_COUNTER.fetch_add(1, std::sync::atomic::Ordering::SeqCst),
            data,
            grad: 0.0,
            parent: parent,
        }
    }

    fn backward(&self) {
        match &self.parent {
            Parent::None => {
                log::debug!("no parent")
            },
            Parent::BinOp { op, left, right } => {
                let mut left = left.borrow_mut();
                let mut right = right.borrow_mut();

                match op {
                    BinOPType::Mul => {
                        log::debug!("mul");
                        left.grad += right.data * self.grad;
                        right.grad += left.data * self.grad;
                    },
                    BinOPType::Add => {
                        log::debug!("add");
                        left.grad += self.grad;
                        right.grad += self.grad;
                    },
                    BinOPType::Sub => {
                        log::debug!("sub");
                        left.grad += self.grad;
                        right.grad += -self.grad;
                    },
                    BinOPType::Max => {
                        log::debug!("max");
                        left.grad += if left.data > right.data {
                            self.grad
                        } else {
                            0.0
                        };
                        right.grad += if right.data > left.data {
                            self.grad
                        } else {
                            0.0
                        };
                    },
                    BinOPType::Min => {
                        log::debug!("min");
                        left.grad += if left.data < right.data {
                            self.grad
                        } else {
                            0.0
                        };
                        right.grad += if right.data < left.data {
                            self.grad
                        } else {
                            0.0
                        };
                    },
                    BinOPType::Div => {
                        log::debug!("div");
                        left.grad += self.grad / right.data;
                        right.grad += -(left.data / right.data.powi(2)) * self.grad;
                    },
                }
            },
            Parent::UnaryOp { op, inner } => {
                log::debug!("unaryop");
                let mut inner = inner.borrow_mut();

                match op {
                    UnaryOPType::Log => {
                        inner.grad += self.grad / inner.data;
                    },
                    UnaryOPType::Exp => {
                        inner.grad += self.grad * self.data.exp();
                    }
                }
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct Value {
    inner: Rc<RefCell<Inner>>,
}

impl Value {
    pub fn new(data: f64) -> Value {
        Value {
            inner: Rc::new(RefCell::new(Inner::new(data, Parent::None))),        
        }
    }

    pub fn data(&self) -> f64 {
        self.inner.borrow().data
    }

    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    pub fn sub_assign(&self, v: f64) {
        //println!("update value: {}", v);
        self.inner.borrow_mut().data -= v
    }

    pub fn mul(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data * other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Mul,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    pub fn div(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data / other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Div,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    pub fn add(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data + other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Add,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    pub fn sub(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data - other.inner.borrow().data;

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Sub,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    // pub fn pow(&self, other: f64) -> Value {
        
    // }

    pub fn relu(&self) -> Value {
        self.max(&Value::new(0.0))
    }

    pub fn log(&self) -> Value {
        let new_data = self.inner.borrow().data.log2();

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::UnaryOp {
                    op: UnaryOPType::Log,
                    inner: self.inner.clone(),
                }
            ))), 
        }
    }

    pub fn exp(&self) -> Value {
        let new_data = self.inner.borrow().data.exp();

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::UnaryOp {
                    op: UnaryOPType::Exp,
                    inner: self.inner.clone(),
                }
            ))), 
        }
    }

    pub fn max(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data.max(other.inner.borrow().data);

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Max,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    pub fn min(&self, other: &Value) -> Value {
        let new_data = self.inner.borrow().data.min(other.inner.borrow().data);

        Value { 
            inner: Rc::new(RefCell::new(Inner::new(
                new_data, 
                Parent::BinOp {
                    op: BinOPType::Min,
                    left: self.inner.clone(),
                    right: other.inner.clone(),
                }
            ))), 
        }
    }

    pub fn zero_grad(&self) {
        self.inner.borrow_mut().grad = 0.0;
    }

    pub fn backward(&self) {
        self.inner.borrow_mut().grad = 1.0;
        let mut toppo = vec![];
        let mut stack = vec![self.inner.clone()];
        let mut visited = HashSet::new();
        loop {
            let inner = match stack.pop() {
                Some(inner) => inner,
                None => break,
            };
            {
                let inner: std::cell::Ref<'_, Inner> = inner.borrow();

                if visited.contains(&inner.id) {
                    continue;
                }

                visited.insert(inner.id);

                match &inner.parent {
                    Parent::None => {},
                    Parent::BinOp { left, right, .. } => {
                        stack.push(left.clone());
                        stack.push(right.clone());
                    },
                    Parent::UnaryOp { inner, .. } => {
                        stack.push(inner.clone());
                    }
                }
            } 

            toppo.push(inner.clone());
        }

        log::info!("backward {} values", toppo.len());

        for inner in toppo {
            inner.borrow().backward();
        }
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