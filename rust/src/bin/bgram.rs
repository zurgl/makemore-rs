use std::collections::{BTreeSet, HashMap};
use std::fs::read_to_string;
use std::iter::zip;

use tch::{no_grad, Device, Kind, Tensor};

fn extract_lables(data: &str) -> BTreeSet<char> {
    let mut labels = BTreeSet::<char>::new();
    data.chars().filter(|&char| char != '\n').for_each(|char| {
        labels.insert(char);
    });
    labels
}

fn char_to_int(labels: &BTreeSet<char>) -> impl Fn(char) -> u8 {
    let mut stoi = HashMap::from([('.', 0)]);
    labels.iter().enumerate().for_each(|(index, &label)| {
        stoi.insert(label, (index as u8) + 1u8);
    });
    move |x| *stoi.get(&x).unwrap()
}

fn int_to_char(labels: &BTreeSet<char>) -> impl Fn(u8) -> char {
    let mut itos = HashMap::from([(0, '.')]);
    labels.iter().enumerate().for_each(|(index, &label)| {
        itos.insert((index as u8) + 1u8, label);
    });
    move |x| *itos.get(&x).unwrap()
}

macro_rules! dim {
    ($n : expr) => {
        Some([$n].as_slice())
    };
}

fn create_dataset(words: &Vec<String>, labels: &BTreeSet<char>) -> (Tensor, Tensor, i64) {
    let mut xs0 = Vec::new();
    let mut ys0 = Vec::new();
    let stoi = char_to_int(labels);
    for word in words {
        let chs = format!(".{}.", word);
        for (ch1, ch2) in zip(chs.chars(), chs.chars().skip(1)) {
            let ix1 = stoi(ch1);
            let ix2 = stoi(ch2);
            xs0.push(ix1);
            ys0.push(ix2);
        }
    }
    let xs = tch::Tensor::from_slice(&xs0).to_kind(Kind::Float);
    let ys = tch::Tensor::from_slice(&ys0).to_kind(Kind::Int64);

    let num = xs.size();
    println!("number of examples: {:?}", num);

    (xs, ys, num[0])
}

fn main() -> Result<(), String> {
    let device = Device::cuda_if_available();
    //let device = Device::Cpu;
    println!("{:?}", device);
    let data = read_to_string("data/names.txt").expect("File not Found.");
    let labels = extract_lables(&data);
    println!("{:?}", labels);

    let words: Vec<_> = data.lines().map(|word| String::from(word.trim())).collect();

    let (xs, ys, num) = create_dataset(&words, &labels);

    tch::manual_seed(2147483647);
    let mut ws = Tensor::randn(&[27, 27], (Kind::Float, device)).set_requires_grad(true);
    for k in 0..1200 {
        let xenc = xs.onehot(27).to_kind(Kind::Float);
        let logits = xenc
            .to_device(device)
            .matmul(&ws.to_device(device))
            .to_kind(Kind::Float)
            .to_device(device);
        let counts = logits.exp();
        let probs = counts.divide(&counts.sum_dim_intlist(dim!(1), true, Kind::Float));

        let gravity = ws.pow_tensor_scalar(2).mean(Kind::Float) * (0.01);
        let loss = -probs
            .index(&[
                Some(Tensor::arange(num, (Kind::Int64, device))),
                Some(ys.shallow_clone()),
            ])
            .log()
            .mean(Kind::Float)
            + gravity;
        if (k + 1) % 20 == 0 {
            println!("iterations: {}, loss : {:?}", k, loss.data());
        }

        // backward pass
        ws.zero_grad();
        loss.backward();

        // update
        no_grad(|| {
            ws += ws.grad() * (-50);
        });
    }

    tch::manual_seed(2147483647);
    let itos = int_to_char(&labels);
    let mut ix = 0;
    for _ in 0..5 {
        let mut pred = String::new();
        loop {
            let xenc = Tensor::from(ix).onehot(27).to_kind(Kind::Float);
            let logits = xenc
                .to_device(device)
                .matmul(&ws.to_device(device))
                .to_device(device);
            let counts = logits.exp();
            let probs =
                counts.divide(&counts.sum_dim_intlist(Some([0].as_slice()), true, Kind::Float));
            ix = u8::try_from(probs.multinomial(1, true)).expect("cannot cast tensor to u8");
            if ix == 0 {
                break;
            }
            pred.push(itos(ix));
        }
        println!("{pred}");
    }

    Ok(())
}
