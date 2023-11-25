use std::collections::HashMap;
use std::iter::zip;
use dataset::Mnist;
use microml::MLP;
use microml::Value;
use microml::calculate_accuracy;
use microml::cross_entropy_loss;
use microml::get_predicted_label;
use microml::one_hot_encode;
use microml::softmax;
use simple_logger::SimpleLogger;

#[tokio::main]
async fn main() {
    SimpleLogger::new().with_level(log::LevelFilter::Info).init().unwrap();

    let mnist = Mnist::load().await.unwrap();

    let mlp = MLP::new(&[784, 128, 64, 10]);
    log::info!("parameter count: {}", mlp.parameters().len());

    let learning_rate = 0.001;
    let batch_size = 64;
    let num_batches = mnist.train_images.count() / batch_size;
    let mut label_losses: HashMap<u8, f64> = HashMap::new();
    let mut last_loss = 0.0;
    let mut last_accuracy = 0.0;

    log::info!("image count: {}", mnist.train_images.count());
    log::info!("learning_rate: {}", learning_rate);
    log::info!("batch_size: {}", batch_size);
    log::info!("num_batches: {}", num_batches);

    let mut i = 0;

    for epoch in 0..5 {
        let mut predicted_labels: Vec<u32> = vec![];
        let mut actual_labels: Vec<u32> = vec![];

        for batch in 0..num_batches - 1 {
            let inputs = mnist.train_images.get_batch(batch, batch_size).unwrap()    
                .chunks(batch_size as usize);
            let labels = mnist.train_labels.get_batch(batch, batch_size).unwrap();

            let mut batch_loss = 0.0;
            for (input, label) in zip(inputs, labels) {
                actual_labels.push(*label as u32);
                let input = input.iter().map(|p| Value::new(*p as f64 / 255.0)).collect::<Vec<Value>>();
                let out = mlp.forward(input);
                let out =  softmax(&out);
                let predicted_label = get_predicted_label(&out);
                predicted_labels.push(predicted_label as u32);
                let label = one_hot_encode(*label as usize, 10);
                let loss = cross_entropy_loss(&out, &label);
                batch_loss += loss.data();
                loss.backward();

                if i % 10_000 == 0 { 
                    let out = out.iter().map(|v| v.data()).collect::<Vec<f64>>();
                    let label = label.iter().map(|v| v.data()).collect::<Vec<f64>>();
                    log::info!("out: {:.4?} label: {:.4?}", out, label);
                }

                i += 1;
            }

            batch_loss /= batch_size as f64;
            
            for p in mlp.parameters() {
                let grad = p.grad() / batch_size as f64;
                p.sub_assign(grad * learning_rate);
            }

            mlp.zero_grad();

            if batch % 50 == 0 {
                let loss_str = if batch_loss > last_loss {
                    "↑"
                } else {
                    "↓"
                };
                last_loss = batch_loss;

                let accuracy = calculate_accuracy(&actual_labels, &predicted_labels);

                let acc_str = if last_accuracy > accuracy {
                    "↓"
                } else {
                    "↑"
                };
                last_accuracy = accuracy;

                log::info!("epoch: {}, batch: {}, loss: {} {} accuracy: {} {}", epoch, batch, batch_loss, loss_str, accuracy, acc_str);

                actual_labels.clear();
                predicted_labels.clear();
            }
        }
    }
}