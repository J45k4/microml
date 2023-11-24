use std::collections::HashSet;

use dataset::Mnist;
use microml::MLP;
use microml::Value;
use microml::create_random_floats;
use microml::cross_entropy_loss;
use microml::one_hot_encode;
use microml::softmax;
use simple_logger::SimpleLogger;

#[tokio::main]
async fn main() {
    SimpleLogger::new().with_level(log::LevelFilter::Info).init().unwrap();

    let mnist = Mnist::load().await.unwrap();

    let mlp = MLP::new(&[1, 1]);

    log::info!("parameter count: {}", mlp.parameters().len());

    for epoch in 0..1 {
        for i in 0..1 {
            let input = mnist.train_images.get(i).unwrap().iter()
                .map(|x| Value::new(*x as f64)).collect::<Vec<Value>>();
            let out = mlp.forward(input);
            let out =  softmax(&out);
            let label_num = mnist.train_labels.get(i).unwrap();
            let label = one_hot_encode(label_num as usize, 10);
            let loss = cross_entropy_loss(&out, &label);
            mlp.zero_grad();
            loss.backward();

            let learning_rate = 0.01;

            log::info!("loss: {:#?}", loss);
            log::info!("loss grad: {}", loss.grad());

            for l in label.iter() {
                if l.grad() > 0.0 {
                    log::info!("label grad: {}", l.grad());
                }
            }

            for o in out.iter() {
                if o.grad() > 0.0 {
                    log::info!("out grad: {}", o.grad());
                }
            }

            for p in mlp.parameters() {
                let grad = p.grad();
                if grad > 0.0 {
                    log::info!("grad: {}", grad);
                }

                p.sub_assign(grad * learning_rate);
            }

            log::info!("label: {:?} loss: {:?}", label_num, loss.data());
        }
    }
}
