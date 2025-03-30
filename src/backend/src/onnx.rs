use crate::Classification;
use prost::Message;
use std::cell::RefCell;
use tract_onnx::prelude::*;

/// Type alias for a compiled and runnable ONNX model.
type Model = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

thread_local! {
    static MODEL: RefCell<Option<Model>> = RefCell::new(None);
}

/// Embedded ONNX model bytes for flower classification.
const FLOWERS: &[u8] = include_bytes!("../assets/flower_cnn_model.onnx");

/// Loads an ONNX model from raw bytes and compiles it into a runnable plan.
fn load_model(bytes: &[u8]) -> TractResult<Model> {
    let proto: tract_onnx::pb::ModelProto = tract_onnx::pb::ModelProto::decode(bytes)?;
    let model = tract_onnx::onnx()
        .model_for_proto_model(&proto)?
        .into_optimized()?
        .into_runnable()?;
    Ok(model)
}

/// Initializes the global model instance by loading and compiling the embedded ONNX file.
/// This function must be called before using `classify`.
pub fn setup() -> TractResult<()> {
    let model = load_model(FLOWERS)?;
    MODEL.with(|m| {
        *m.borrow_mut() = Some(model);
    });
    Ok(())
}

/// Classifies an image into one of the 5 flower categories using the preloaded model.
pub fn classify(image: Vec<u8>) -> Result<Vec<Classification>, anyhow::Error> {
    MODEL.with_borrow(|model_cell| {
        // Get a reference to the loaded model
        let model = model_cell
            .as_ref()
            .expect("Model is not initialized");
        
        // Decode and resize the image to match model input size (128x128)
        let image = image::load_from_memory(&image)?.to_rgb8();
        let image = image::imageops::resize(&image, 128, 128, image::imageops::Triangle);

        // Normalize image using channel-wise mean and std deviation
        const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
        const STD: [f32; 3] = [0.229, 0.224, 0.225];
        let tensor = tract_ndarray::Array4::from_shape_fn((1, 3, 128, 128), |(_, c, y, x)| {
            (image[(x as u32, y as u32)][c] as f32 / 255.0 - MEAN[c]) / STD[c]
        });

        // Run the model on the prepared tensor
        let result = model.run(tvec!(Tensor::from(tensor).into()))?;
        let logits = result[0].to_array_view::<f32>()?;

        // Apply softmax to get probabilities
        let exp: Vec<f32> = logits.iter().map(|x| x.exp()).collect();
        let sum_exp: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|x| x / sum_exp).collect();

        // Build a list of predictions with labels and confidence scores
        let mut scored: Vec<_> = probs
            .iter()
            .enumerate()
            .map(|(i, prob)| Classification {
                label: FLOWER_LABELS.get(i).unwrap_or(&"unknown").to_string(),
                score: (prob * 10000.0).round() / 100.0,
            })
            .collect();
        
        // Sort by descending score and return top 3
        scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        Ok(scored.into_iter().take(3).collect())
    })
}

/// The set of 5 possible classification labels.
const FLOWER_LABELS: [&'static str; 5] = [
    "daisy",
    "dandelion",
    "rose",
    "sunflower",
    "tulip",
];