#![allow(dead_code, unused_variables, unused_imports)]

use tch::{manual_seed, Device, Kind, Tensor};

macro_rules! dim {
    ($n : expr) => {
        Some([$n].as_slice())
    };
}

#[derive(Debug)]
struct Linear {
    weight: Tensor,
}

impl Linear {
    fn new(fan_in: i64, fan_out: i64) -> Self {
        Self {
            weight: Tensor::randn(&[fan_in, fan_out], (Kind::Float, Device::Cpu)),
        }
    }

    fn parameters(self) -> Vec<Tensor> {
        vec![self.weight]
    }

    fn apply(&self, x: Tensor) -> Tensor {
        x.matmul(&self.weight)
    }
}

#[derive(Debug)]
struct Batch {
    eps: f64,
    momentum: f64,
    training: bool,
    gamma: Tensor,
    beta: Tensor,
    running_var: Tensor,
    running_mean: Tensor,
}

impl Batch {
    fn new(dim: i64) -> Self {
        let eps = 0.00001;
        let momentum = 0.1;
        let training = true;
        let running_mean = Tensor::zeros(&[dim], (Kind::Float, Device::Cpu));
        let running_var = Tensor::ones(&[dim], (Kind::Float, Device::Cpu));
        let beta = Tensor::zeros(&[dim], (Kind::Float, Device::Cpu));
        let gamma = Tensor::ones(&[dim], (Kind::Float, Device::Cpu));

        Self {
            eps,
            momentum,
            training,
            gamma,
            beta,
            running_mean,
            running_var,
        }
    }

    fn set_momentum(&mut self, momentum: f64) -> &Self {
        self.momentum = momentum;
        self
    }

    fn set_eps(&mut self, eps: f64) -> &Self {
        self.eps = eps;
        self
    }

    fn set_training(&mut self, training: bool) -> &Self {
        self.training = training;
        self
    }

    fn parameters(self) -> Vec<Tensor> {
        vec![self.gamma, self.beta]
    }

    fn apply(&mut self, x: Tensor) -> Tensor {
        let (xmean, xvar) = if self.training {
            (
                x.mean_dim(dim!(0), true, Kind::Float),
                x.var_dim(dim!(0), false, true),
            )
        } else {
            (
                self.running_mean.shallow_clone(),
                self.running_var.shallow_clone(),
            )
        };
        let xhat = (x - &xmean).divide(&Tensor::sqrt(&(xvar.shallow_clone() + self.eps)));
        let out = &self.gamma * xhat + &self.beta;
        if self.training {
            tch::no_grad(|| {
                self.running_mean =
                    (1f64 - self.momentum) * &self.running_mean + xmean * self.momentum;
                self.running_var =
                    (1f64 - self.momentum) * &self.running_var + xvar * self.momentum;
            })
        }
        out
    }
}

#[derive(Debug)]
struct Tanh {
    fun: fn(Tensor) -> Tensor,
}

impl Tanh {
    fn new() -> Self {
        Self {
            fun: move |x| Tensor::tanh(&x),
        }
    }

    // fn apply(&mut self, x: Tensor) {
    //     self.out = Tensor::tanh(&x);
    // }
    //
    // fn parameters(&self) -> Vec<Tensor> {
    //     Vec::<Tensor>::new()
    // }
}

enum Layer {
    Neuron(Linear),
    Norm(Batch),
    Activation(Tanh),
}

fn main() {
    manual_seed(2147483647);

    let n_embd = 10;
    let block_size = 3;
    let n_hidden = 100;
    let vocab_size = 27;

    let c = Tensor::randn(&[vocab_size], (Kind::Float, Device::Cpu));

    let layers: Vec<Layer> = vec![
        Layer::Neuron(Linear::new(n_embd * block_size, n_hidden)),
        Layer::Norm(Batch::new(n_hidden)),
        Layer::Activation(Tanh::new()),
    ];
    println!("hello")
}
