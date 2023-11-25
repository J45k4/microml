use std::collections::HashMap;
use std::iter::zip;

use dataset::Point;
use dataset::generate_moons;
use microml::MLP;
use microml::Value;
use microml::calculate_accuracy;
use microml::cross_entropy_loss;
use microml::get_predicted_label;
use microml::one_hot_encode;
use microml::softmax;
use plotters::prelude::*;
use simple_logger::SimpleLogger;

fn plot_moons(data: &[Point], labels: &[i32]) -> anyhow::Result<()> {
    let root = BitMapBackend::new("moons_plot.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Moons Dataset", ("sans-serif", 40).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(-1.5f64..2.5f64, -1.0f64..1.5f64)?;

    chart.configure_mesh().draw()?;

    let red = RGBColor(255, 0, 0).mix(0.1);
    let blue = RGBColor(0, 0, 255).mix(0.1);

    for (point, &label) in data.iter().zip(labels.iter()) {
        chart.draw_series(PointSeries::of_element(
            vec![(point.x, point.y)],
            5,
            if label == 0 { &red } else { &blue },
            &|coord, size, style| {
                EmptyElement::at(coord) + Circle::new((0, 0), size, style.filled())
            },
        ))?;
    }

    Ok(())
}

// pub fn plot_predictions(points_by_label: HashMap<u32, Vec<(f64, f64)>>) {
//     let mut fg = Figure::new();

//     for (label, points) in points_by_label {
//         let color = match label {
//             0 => "blue",
//             1 => "red",
//             // Add more colors for more labels if necessary
//             _ => "green",
//         };

//         fg.axes2d().points(
//             points.iter().map(|(x, _)| *x),
//             points.iter().map(|(_, y)| *y),
//             &[Scatter::default().color(color).marker_size(1.0)],
//         );
//     }

//     // Display or save the plot
//     fg.show().unwrap(); // or fg.save("plot.png").unwrap();
// }

#[tokio::main]
async fn main() {
    SimpleLogger::new().with_level(log::LevelFilter::Info).init().unwrap();
    
    let train_dataset = generate_moons(2000, 0.1);
    log::info!("train_dataset: {:?}", train_dataset.labels.iter().take(10).collect::<Vec<_>>());
    let test_dataset = generate_moons(100, 0.01);
    plot_moons(&train_dataset.points, &train_dataset.labels).unwrap();

    let learning_rate = 0.01;
    let lambda = 0.001;
    let batch_size = 64; // Set your batch size

    let mlp = MLP::new(&[2, 50, 2]);

    let mut real_labels = Vec::new();
    let mut predicted_labels = Vec::new();
    let train_bathes = zip(train_dataset.points, train_dataset.labels).collect::<Vec<_>>();
    let mut i = 0;

    for epoch in 0..5 {
        // Shuffle your dataset here (if possible)

        let mut total_loss = 0.0;
        let mut batch_count = 0;

        for batch in train_bathes.chunks(batch_size) {
            let mut batch_loss = 0.0;
            let mut batch_gradients = Vec::new();

            for (point, label) in batch {
                real_labels.push(*label as u32);
                let input = vec![point.x, point.y];
                let out = mlp.forward(input.iter().map(|p| Value::new(*p)).collect::<Vec<Value>>());
                let out = softmax(&out);
                let label_hot = one_hot_encode(*label as usize, 2);
                let predicted_label = get_predicted_label(&out);
                predicted_labels.push(predicted_label as u32);

                let loss = cross_entropy_loss(&out, &label_hot);
                batch_loss += loss.data();
                loss.backward();

                if i % 5_111 == 0 { 
                    let out = out.iter().map(|v| v.data()).collect::<Vec<f64>>();
                    let label_hot = label_hot.iter().map(|v| v.data()).collect::<Vec<f64>>();
                    log::info!("loss: {} out: {:.4?} label: {:.4?} hot_label: {:.4?}", loss.data(), out, label, label_hot);
                }

                i += 1;

                // Store gradients for each parameter in batch_gradients
                for p in mlp.parameters() {
                    batch_gradients.push(p.grad() as f64);
                }

                mlp.zero_grad();
            }

            // Update parameters based on average gradients in the batch
            for (p, grad) in zip(mlp.parameters(), batch_gradients.iter()) {
                let l2_penalty = lambda * p.data(); // L2 penalty term
                p.sub_assign((*grad / batch_size as f64 + l2_penalty) * learning_rate);
            }

            total_loss += batch_loss;
            batch_count += 1;
        }

        let average_loss = total_loss / batch_count as f64;
        // Calculate accuracy and other metrics here

        let accuracy = calculate_accuracy(&real_labels, &predicted_labels);
        real_labels.clear();
        predicted_labels.clear();

        log::info!("epoch: {} average_loss: {:.4?} accuracy: {:.2?}", epoch, average_loss, accuracy);
    }

    let mut predictions = vec![];

    for (point, label) in zip(&test_dataset.points, test_dataset.labels) {
        let input = vec![point.x, point.y];
        let out = mlp.forward(input.iter().map(|p| Value::new(*p)).collect::<Vec<Value>>());
        let out = softmax(&out);
        let predicted_label = get_predicted_label(&out);
        predictions.push(predicted_label as i32);
        let label_hot = one_hot_encode(label as usize, 2);
        let loss = cross_entropy_loss(&out, &label_hot);
        let label_hot = label_hot.iter().map(|v| v.data()).collect::<Vec<f64>>();
        let out = out.iter().map(|v| v.data()).collect::<Vec<f64>>();
        log::info!("test loss: {} out: {:.4?} label: {:.4?} hot_label: {:.4?} predicted_label: {:?}", loss.data(), out, label, label_hot, predicted_label);
    }

    plot_moons(&test_dataset.points, &predictions).unwrap();
}