use tch::{Device, Kind, Tensor};

fn tensor_from() {
    let t1 = Tensor::from(1).to_kind(Kind::Int64).to_device(Device::Cpu);
    println!("t1: {:?}", t1);
    println!("t1.size: {:?}", t1.size());
    println!("t1.dim: {:?}", t1.dim());
    t1.print();
    println!();

    let t2 = Tensor::of_slice(&[1])
        .to_kind(Kind::Int64)
        .to_device(Device::Cpu);
    println!("t2: {:?}", t2);
    println!("t2.size: {:?}", t2.size());
    println!("t2.dim: {:?}", t2.dim());
    t2.print();
    println!();

    let t3 = Tensor::of_slice(&[1, 2, 3, 4])
        .to_kind(Kind::Int64)
        .to_device(Device::Cpu);
    println!("t3: {:?}", t3);
    println!("t3.size: {:?}", t3.size());
    println!("t3.dim: {:?}", t3.dim());
    t3.print();
    println!();

    let t4 = t3.view((2, 2));
    println!("t4: {:?}", t4);
    println!("t4.size: {:?}", t4.size());
    println!("t4.dim: {:?}", t4.dim());
    println!("t4.numel: {:?}", t4.numel());
    println!("t4.data: {:?}", t4.data());
    t4.print();
    println!();
}

fn tensor_create() {
    let zeros_tensor = Tensor::zeros(&[2, 3], (Kind::Int64, Device::Cpu));
    zeros_tensor.print();

    let ones_tensor = Tensor::ones(&[2, 3], (Kind::Int64, Device::Cpu));
    ones_tensor.print();

    let randn_tensor = Tensor::randn(&[2, 3], (Kind::Float, Device::Cpu));
    randn_tensor.print();
}

fn main() {
    tensor_from();
    tensor_create();
}
