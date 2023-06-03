// #![allow(unused_macros, unused_imports, unused_variables, dead_code)]

use std::collections::{BTreeSet, HashMap};
// use std::ops::{Div, Mul};

use tch::{no_grad, Device, IndexOp, Kind, Tensor};

const BLOCK_SIZE: usize = 3;
const BATCH_SIZE: i64 = 32;
const TOKEN: char = '.';

type Set = BTreeSet<char>;

fn extract_labels(data: &str) -> Set {
    let mut labels = Set::new();
    data.chars()
        .filter(|&char| char != TOKEN && char != '\n')
        .for_each(|char| {
            labels.insert(char);
        });
    labels
}

fn make_lens(labels: &Set) -> (impl Fn(char) -> i64, impl Fn(i64) -> char) {
    let mut index = 0;
    let mut stoi = HashMap::from([(TOKEN, index)]);
    let mut itos = HashMap::from([(index, TOKEN)]);
    labels.iter().for_each(|&label| {
        index += 1;
        stoi.insert(label, index);
        itos.insert(index, label);
    });

    (
        move |x| *stoi.get(&x).unwrap(),
        move |x| *itos.get(&x).unwrap(),
    )
}

fn create_dataset(
    words: &[String],
    stoi: &impl Fn(char) -> i64,
    device: Device,
) -> (Tensor, Tensor) {
    let (mut xs, mut ys) = (vec![], vec![]);
    for word in words.iter() {
        let chars = word.chars().map(stoi).collect::<Vec<i64>>();
        for chunk in chars.windows(BLOCK_SIZE + 1) {
            xs.extend_from_slice(&chunk[..BLOCK_SIZE]);
            ys.extend_from_slice(&chunk[BLOCK_SIZE..]);
        }
    }

    (
        Tensor::from_slice(&xs)
            .view([-1, (BLOCK_SIZE as i64)])
            .to_device(device),
        Tensor::from_slice(&ys).to_device(device),
    )
}

fn run(device: Device) -> Result<(), String> {
    println!("{:?}", device);

    let data = include_str!("../../data/names.txt");
    let labels = extract_labels(data);
    println!("{:?}", labels);

    let words: Vec<_> = data
        .lines()
        .map(|word| format!("{}{}{}", TOKEN.to_string().repeat(BLOCK_SIZE), word, TOKEN))
        .collect();

    let (stoi, itos) = make_lens(&labels);

    let (xs, ys) = create_dataset(&words, &stoi, device);
    //let (xs_dev, ys_dev) = create_dataset(&words[n1..n2], &stoi);
    //let (xs_test, ys_test) = create_dataset(&words[n2..], &stoi);

    tch::manual_seed(2147483647);
    let size = xs.size()[0];
    let options = (Kind::Float, device);
    let mut c = Tensor::randn(&[27, 10], options).set_requires_grad(true);
    let mut w1 = Tensor::randn(&[30, 200], options).set_requires_grad(true);
    let mut b1 = Tensor::randn(&[200], options).set_requires_grad(true);
    let mut w2 = Tensor::randn(&[200, 27], options).set_requires_grad(true);
    let mut b2 = Tensor::randn(&[27], options).set_requires_grad(true);

    for n in 0..200_000 {
        let ix = Tensor::randint(size, &[BATCH_SIZE], (Kind::Int64, device));
        let emb = c.index(&[Some(xs.i(&ix))]);
        let h = Tensor::tanh(&(emb.view((-1, 30)).matmul(&w1) + b1.shallow_clone()));
        let logits = h.matmul(&w2) + b2.shallow_clone();
        let loss = logits.cross_entropy_for_logits(&ys.i(&ix));

        if n % 20_000 == 0 {
            println!("iterations: {}, loss : {:?}", n, loss.data());
        }

        c.zero_grad();
        w1.zero_grad();
        b1.zero_grad();
        w2.zero_grad();
        b2.zero_grad();
        loss.backward();

        let lr = if n < 100_000 { 0.1 } else { 0.01 };

        // update
        no_grad(|| {
            c += c.grad() * (-lr);
            w1 += w1.grad() * (-lr);
            b1 += b1.grad() * (-lr);
            w2 += w2.grad() * (-lr);
            b2 += b2.grad() * (-lr);
        });
    }

    tch::manual_seed(2147483647 + 10);
    for _ in 0..10 {
        let mut output = String::new();
        let mut context = Tensor::from_slice(&[0i64; BLOCK_SIZE]);
        loop {
            let emb = c.index(&[Some(context.shallow_clone())]);
            let h = Tensor::tanh(&(emb.view((1, -1)).matmul(&w1) + b1.shallow_clone()));
            let logits = h.matmul(&w2) + b2.shallow_clone();
            let probs = logits.softmax(1, Kind::Float);
            let ix = i64::try_from(probs.multinomial(1, true)).expect("cannot cast tensor to i64");
            if ix == 0i64 {
                break;
            }
            context = Tensor::cat(&[context.i(1..), Tensor::from_slice(&[ix])], 0);
            output.push(itos(ix));
        }
        println!("{output}");
    }

    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    match a.iter().map(|x| x.as_str()).collect::<Vec<_>>().as_slice() {
        [_, "cpu"] => run(Device::Cpu).ok(),
        [_, "gpu"] => run(Device::cuda_if_available()).ok(),
        _ => panic!("usage: main cpu|gpu"),
    };
}

#[test]
fn test_tensor() {
    let mut context = Tensor::from_slice(&[0i64; BLOCK_SIZE]);
    context.print();

    let x = Tensor::from_slice(&[1i64]);
    context = Tensor::cat(&[context.i(1..), x], 0);
    context.print();

    let x = Tensor::from_slice(&[2i64]);
    context = Tensor::cat(&[context.i(1..), x], 0);
    context.print();
}

#[test]
fn test_labels() {
    let alphabet = "abcdefghijklmnopqrstuvwxyz";
    let labels = extract_labels(alphabet);
    assert_eq!(labels.len(), 26);

    let name = "emma\n";
    let labels = extract_labels(name);
    assert_eq!(labels.len(), 3)
}

#[test]
fn test_lens() {
    let alphabet = "abcdefghijklmnopqrstuvwxyz";
    let labels = extract_labels(alphabet);
    let (stoi, itos) = make_lens(&labels);

    assert_eq!(stoi('.'), 0);
    assert_eq!(stoi('a'), 1);

    assert_eq!(itos(0), '.');
    assert_eq!(itos(1), 'a');
}

#[test]
fn test_windows() {
    let slice = ['r', 'u', 's', 't'];
    let mut iter = slice.windows(2);
    assert_eq!(iter.next().unwrap(), &['r', 'u']);
    assert_eq!(iter.next().unwrap(), &['u', 's']);
    assert_eq!(iter.next().unwrap(), &['s', 't']);
    assert!(iter.next().is_none());
}
