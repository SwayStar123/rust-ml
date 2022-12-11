use rand::{thread_rng, Rng};
use rand::distributions::Uniform;

pub mod nn;

use nn::{activations::SIGMOID, network::{loss, Network},};

// // Makes a dataset of random points in a circle with radius r, and returns a vector of tuples of the form (vector of x and y coordinates, 0 or 1 depending on whether the point is inside the circle or not)
// pub fn circle_dataset(r: f64, num_samples: u64) -> Vec<(Vec<f64>, Vec<f64>)> {
//     let mut rng = thread_rng();
//     let range = Uniform::new(r * -2.0, r * 2.0);

//     let random_samples: Vec<Vec<f64>> = (0..num_samples).map(|_| vec![rng.sample(range), rng.sample(range)]).collect();

//     let output_samples: Vec<(Vec<f64>, Vec<f64>)> = random_samples.iter().map(|vec| (vec.to_owned(), vec![(vec[0].powf(2.0) + vec[1].powf(2.0) <= r.powf(2.0)) as usize as f64])).collect();
//     output_samples
// }

// The above function but with the targets as a one hot encoded vector
pub fn circle_dataset(r: f64, num_samples: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = thread_rng();
    let range = Uniform::new(r * -2.0, r * 2.0);

    let random_samples: Vec<Vec<f64>> = (0..num_samples).map(|_| vec![rng.sample(range), rng.sample(range)]).collect();

    let inside_circle = [1.0, 0.0];
    let outside_circle = [0.0, 1.0];

    let output_samples: Vec<(Vec<f64>, Vec<f64>)> = random_samples.iter().map(|vec| (vec.to_owned(), if vec[0].powf(2.0) + vec[1].powf(2.0) <= r.powf(2.0) { inside_circle.to_vec() } else { outside_circle.to_vec() })).collect();
    let (inputs, targets): (Vec<Vec<f64>>, Vec<Vec<f64>>) = output_samples.into_iter().unzip();
    (inputs, targets)
}

// OCR MNIST dataset
// pub fn mnist_dataset() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
//     // obtain the images and labels from the mnist dataset


fn main() {
    let (inputs, targets) = circle_dataset(5.0, 100000);
    let (test_inputs, test_targets) = circle_dataset(5.0, 1000);

    let mut network = Network::new(vec![2, 20, 20, 2], 0.5, SIGMOID);

    // print the accuracy before trainig
    println!("Loss before training: {}", loss(&mut network, test_inputs.to_owned(), test_targets.to_owned()));

    network.train(inputs, targets, 100);

    // print the accuracy after training    
    println!("Loss after training: {}", loss(&mut network, test_inputs.to_owned(), test_targets.to_owned()));
}

