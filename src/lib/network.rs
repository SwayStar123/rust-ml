use std::{
	fs::File,
	io::{Read, Write},
};

use serde::{Deserialize, Serialize};
use serde_json::{from_str, json};

use super::{activations::Activation, matrix::Matrix};

pub struct Network<'a> {
	layers: Vec<usize>,
	weights: Vec<Matrix>,
	biases: Vec<Matrix>,
	data: Vec<Matrix>,
	learning_rate: f64,
	activation: Activation<'a>,
}

#[derive(Serialize, Deserialize)]
struct SaveData {
	weights: Vec<Vec<Vec<f64>>>,
	biases: Vec<Vec<Vec<f64>>>,
}

impl Network<'_> {
	pub fn new<'a>(
		layers: Vec<usize>,
		learning_rate: f64,
		activation: Activation<'a>,
	) -> Network<'a> {
		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..layers.len() - 1 {
			weights.push(Matrix::random(layers[i + 1], layers[i]));
			biases.push(Matrix::random(layers[i + 1], 1));
		}

		Network {
			layers,
			weights,
			biases,
			data: vec![],
			learning_rate,
			activation,
		}
	}

	pub fn feed_forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
		if inputs.len() != self.layers[0] {
			panic!("Invalid inputs length");
		}

		let mut current = Matrix::from(vec![inputs]).transpose();
		self.data = vec![current.clone()];

		for i in 0..self.layers.len() - 1 {
			current = self.weights[i]
				.multiply(&current)
				.add(&self.biases[i])
				.map(self.activation.function);
			self.data.push(current.clone());
		}

		current.transpose().data[0].to_owned()
	}

	pub fn back_propogate(&mut self, outputs: Vec<f64>, targets: Vec<f64>) {
		if targets.len() != self.layers[self.layers.len() - 1] {
			panic!("Invalid targets length");
		}

		let mut parsed = Matrix::from(vec![outputs.clone()]);
		// let mut errors = Matrix::from(vec![targets]).subtract(&parsed).transpose();
		let mut errors = Matrix::from(vec![loss_single(outputs, targets)]).transpose();
		let mut gradients = parsed.map(self.activation.derivative).transpose();

		for i in (0..self.layers.len() - 1).rev() {
			gradients = gradients
				.dot_multiply(&errors)
				.map(&|x| x * self.learning_rate);

			self.weights[i] = self.weights[i].add(&gradients.multiply(&self.data[i].transpose()));
			self.biases[i] = self.biases[i].add(&gradients);

			errors = self.weights[i].transpose().multiply(&errors);
			gradients = self.data[i].map(self.activation.derivative);
		}
	}

	pub fn train(&mut self, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>, epochs: u16) {
		for i in 1..=epochs {
			if epochs < 100 || i % (epochs / 100) == 0 {
				println!("Epoch {} of {}", i, epochs);
				// print accuracy
				let mut i = inputs.clone();
				i.truncate(50);
				let mut t = targets.clone();
				t.truncate(50);
				println!("Loss: {}", loss(self, i, t))
			}
			for j in 0..inputs.len() {
				let outputs = self.feed_forward(inputs[j].clone());
				self.back_propogate(outputs, targets[j].clone());
			}
		}
	}

	pub fn save(&mut self, file: String) {
		let mut file = File::create(file).expect("Unable to touch save file");

		file.write_all(
			json!({
				"weights": self.weights.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>(),
				"biases": self.biases.clone().into_iter().map(|matrix| matrix.data).collect::<Vec<Vec<Vec<f64>>>>()
			}).to_string().as_bytes(),
		).expect("Unable to write to save file");
	}

	pub fn load(&mut self, file: String) {
		let mut file = File::open(file).expect("Unable to open save file");
		let mut buffer = String::new();

		file.read_to_string(&mut buffer)
			.expect("Unable to read save file");

		let save_data: SaveData = from_str(&buffer).expect("Unable to serialize save data");

		let mut weights = vec![];
		let mut biases = vec![];

		for i in 0..self.layers.len() - 1 {
			weights.push(Matrix::from(save_data.weights[i].clone()));
			biases.push(Matrix::from(save_data.biases[i].clone()));
		}

		self.weights = weights;
		self.biases = biases;
	}
}

// // find accuracy as within 0.05 of the target
// pub fn accuracy(network: &mut Network, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
//     let mut correct = 0;
//     for i in 0..inputs.len() {
//         let output = network.feed_forward(inputs[i].to_owned());
//         if (output[0] - targets[i][0]).abs() < 0.05 {
//             correct += 1;
//         }
//     }
//     correct as f64 / inputs.len() as f64
// }

//Categorical Cross-Entropy Loss
// pub fn loss(network: &mut Network, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
// 	let mut loss = 0.0;
// 	for i in 0..inputs.len() {
// 		let output = network.feed_forward(inputs[i].to_owned());
// 		for j in 0..output.len() {
// 			loss += targets[i][j] * output[j].ln();
// 		}
// 	}
// 	-loss / inputs.len() as f64
// }

// Catagorical Cross-Entropy Loss which accounts for 0 probabilities (avoiding a multiply by 0 error)
// clip the output to be between 0.000000001 and 0.999999999
pub fn loss(network: &mut Network, inputs: Vec<Vec<f64>>, targets: Vec<Vec<f64>>) -> f64 {
	let mut loss = 0.0;
	for i in 0..inputs.len() {
		let output = network.feed_forward(inputs[i].to_owned());
		for j in 0..output.len() {
			let clipped = output[j].max(0.000000001).min(0.999999999);
			loss += targets[i][j] * clipped.ln();
		}
	}
	-loss / inputs.len() as f64
}

// single loss for each individual output neuron given Vec<output> and Vec<target>
pub fn loss_single(outputs: Vec<f64>, targets: Vec<f64>) -> Vec<f64> {
	let mut loss = vec![];
	for i in 0..outputs.len() {
		let clipped = outputs[i].max(0.000000001).min(0.999999999);
		loss.push(-(targets[i] * clipped.ln()));
	}
	loss
}
