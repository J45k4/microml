use microml::MLP;
use microml::Value;
use microml::create_random_floats;
use microml::cross_entropy_loss;
use microml::softmax;

#[tokio::main]
async fn main() {
    let mlp = MLP::new(&[2, 2]);

    let input = create_random_floats(2);
    println!("input: {:?}", input);
    let input = input.iter().map(|x| Value::new(*x)).collect();

    let out = mlp.forward(input);
    
    for (inx, layer) in mlp.layers().iter().enumerate() {
        for (ninx, neuron) in layer.neurons().iter().enumerate() {
            let weight_values = neuron.weights().iter().map(|x| x.data()).collect::<Vec<f64>>();
            println!("layer: {} neuron: {} weights: {:?}", inx, ninx, weight_values);
            println!("layer: {} neuron: {} bias: {:?}", inx, ninx, neuron.bias().data());
        }
    }
    let out =  softmax(&out);
    //let out = out.iter().map(|x| x.data()).collect::<Vec<f64>>();
    let loss = cross_entropy_loss(&out, &[Value::new(0.0), Value::new(1.0)]);
    loss.backward();
    println!("loss: {:?}", loss.data());

    // println!("output: {:?}", out);
}
