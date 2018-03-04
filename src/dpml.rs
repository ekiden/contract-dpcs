use ndarray::{Array, Array2, Ix2};
use rand;
use rand::distributions::{Normal, Sample};
use std::f64::consts::E;

pub fn scalar_add(x: &Array2<f64>, s: f64) -> Array2<f64> {
    let v: Vec<f64> = x.iter().map(|i| i.clone() + s).collect();
    let result = Array::from_shape_vec(x.shape(), v).unwrap();
    return result.into_dimensionality::<Ix2>().unwrap();
}

pub fn scalar_multiply(x: &Array2<f64>, s: f64) -> Array2<f64> {
    let v: Vec<f64> = x.iter().map(|i| i.clone() * s).collect();
    let result = Array::from_shape_vec(x.shape(), v).unwrap();
    return result.into_dimensionality::<Ix2>().unwrap();
}

pub fn scalar_divide(x: &Array2<f64>, s: f64) -> Array2<f64> {
    let v: Vec<f64> = x.iter().map(|i| i.clone() / s).collect();
    let result = Array::from_shape_vec(x.shape(), v).unwrap();
    return result.into_dimensionality::<Ix2>().unwrap();
}

pub fn pointwise_exp(x: &Array2<f64>) -> Array2<f64> {
    let v: Vec<f64> = x.iter().map(|i| E.powf(i.clone())).collect();
    let result = Array::from_shape_vec(x.shape(), v).unwrap();
    return result.into_dimensionality::<Ix2>().unwrap();
}

pub fn lr_gradient(
    features: &Array2<f64>,
    labels: &Array2<f64>,
    theta: &Array2<f64>,
    lambda: f64,
) -> Array2<f64> {
    let height = features.shape()[0] as f64;
    let exponent = labels * &(features.dot(theta));
    let gradient_loss = scalar_divide(
        &(-((features.t()).dot(&(labels / &(scalar_add(&pointwise_exp(&exponent), 1.0_f64)))))),
        height,
    );
    let regularization = scalar_multiply(theta, lambda);
    let result = gradient_loss + regularization;
    return result;
}

pub fn dp_logistic_regression(
    features: &Array2<f64>,
    labels: &Array2<f64>,
    lambda: f64,
    learning_rate: f64,
    eps: f64,
    delta: f64,
) -> Array2<f64> {
    let n = features.shape()[0] as f64;
    let l: f64 = 1.0;
    let num_iters = 5 as usize;
    let mut theta = Array2::<f64>::zeros((features.shape()[1], 1));
    let std_dev: f64 = 4.0 * l * ((num_iters as f64) * (1.0 / delta).ln()).sqrt() / (n * eps);
    for _i in 1..num_iters {
        let gradient = lr_gradient(features, labels, &theta, lambda);
        let mut noise_distr = Normal::new(0., std_dev);
        let mut rng = rand::thread_rng();
        let noise_vec = (0..gradient.len())
            .map(|_| noise_distr.sample(&mut rng))
            .collect::<Vec<f64>>();
        let noise = Array::from_shape_vec(gradient.shape(), noise_vec).unwrap();
        theta = theta - learning_rate * (gradient + noise);
    }
    return theta;
}
