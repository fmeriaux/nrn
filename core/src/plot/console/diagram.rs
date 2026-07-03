//! Rendering an [`ActivationDiagram`] to a colored, vertical list of layers:
//! each neuron a marker beside its value and an intensity bar. The output layer
//! reads its neurons as class probabilities. No connections are drawn, and the
//! ranked decision is left to the caller.

use super::{colored, colored_str};
use crate::plot::activations::{ActivationDiagram, DiagramLayer, Unit};
use crate::plot::scene::Color as SceneColor;

/// The hollow marker color of a silent (non-firing) neuron.
const SILENT_COLOR: SceneColor = SceneColor::rgb(90, 90, 90);

/// The full width, in block glyphs, of a neuron's activation-intensity bar.
const BAR_WIDTH: usize = 24;

impl ActivationDiagram {
    /// Renders the forward pass as a vertical list of layers from input to
    /// output: each neuron a colored marker beside its value — filled and tinted
    /// by activation intensity when firing, hollow when silent. The output layer's
    /// neurons read as their class probability. No connections are drawn; the
    /// ranked decision is presented separately by the caller.
    pub fn to_console(&self) -> String {
        let mut output = String::new();
        for layer in &self.layers {
            output.push_str(&render_layer(layer));
            output.push('\n');
        }
        output
    }
}

/// One layer as a bold heading naming its role and neuron count, then one line
/// per shown neuron.
fn render_layer(layer: &DiagramLayer) -> String {
    let mut block = bold(&layer.heading());
    block.push('\n');
    for unit in &layer.units {
        block.push_str(&render_unit(unit));
        block.push('\n');
    }
    block
}

/// One neuron. An output neuron reads as its class and probability; otherwise its
/// marker, index and value, followed by a bar whose length tracks activation
/// intensity when firing, or a right-aligned `silent` flag when not.
fn render_unit(unit: &Unit) -> String {
    if let Some(class) = unit.class {
        let filled = (unit.intensity * BAR_WIDTH as f32).round() as usize;
        let bar = colored_str(&"\u{2588}".repeat(filled), unit.marker_color());
        format!(
            "  {} class {class}  {:>5.1}%  {bar}",
            colored('\u{25cf}', unit.marker_color()),
            unit.value * 100.0
        )
    } else if unit.firing {
        let filled = (unit.intensity * BAR_WIDTH as f32).round() as usize;
        let bar = colored_str(&"\u{2588}".repeat(filled), unit.marker_color());
        format!(
            "  {} n{:<3} {:>9.4}  {bar}",
            colored('\u{25cf}', unit.marker_color()),
            unit.index,
            unit.value
        )
    } else {
        let flag = colored_str(&format!("{:>BAR_WIDTH$}", "silent"), SILENT_COLOR);
        format!(
            "  {} n{:<3} {:>9.4}  {flag}",
            colored('\u{25cb}', SILENT_COLOR),
            unit.index,
            unit.value
        )
    }
}

/// `text` wrapped in the ANSI bold escape, reset after.
fn bold(text: &str) -> String {
    format!("\u{1b}[1m{text}\u{1b}[0m")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{Activation, RELU};
    use crate::layers::Dense;
    use crate::model::NeuralNetwork;
    use crate::plot::activations::DiagramOptions;
    use ndarray::{Array1, Array2, array};

    /// A hidden ReLU layer whose middle neuron dies, then a two-class output
    /// layer resolving to probabilities `[0.3, 0.4]`.
    fn diagram() -> ActivationDiagram {
        let net = NeuralNetwork::single(Dense::new(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
            RELU.clone(),
        ))
        .with_layer(Dense::new(
            array![[0.3, 0.0, 0.0], [0.0, 0.0, 0.2]],
            array![0.0, 0.0],
            RELU.clone(),
        ));
        net.activation_diagram(array![1.0, 1.0].view(), &DiagramOptions::default())
            .unwrap()
    }

    #[test]
    fn to_console_heads_each_layer_with_its_role_and_count() {
        let text = diagram().to_console();
        assert!(text.contains("Input (2 features)"));
        assert!(text.contains(&format!("{} (3 units)", RELU.name())));
    }

    #[test]
    fn to_console_marks_firing_neurons_filled_and_silent_neurons_hollow() {
        let text = diagram().to_console();
        // The dead hidden neuron (value 0) is hollow and flagged; active ones are filled.
        assert!(text.contains('\u{25cb}'));
        assert!(text.contains("silent"));
        assert!(text.contains('\u{25cf}'));
        // Concrete activation values are printed beside their neuron.
        assert!(text.contains("n2"));
        assert!(text.contains("2.0000"));
    }

    #[test]
    fn to_console_reads_the_output_layer_as_class_probabilities() {
        let text = diagram().to_console();
        // The output neurons read as their class and probability, not raw values.
        assert!(text.contains("class 0"));
        assert!(text.contains("class 1"));
        assert!(text.contains("30.0%"));
        assert!(text.contains("40.0%"));
    }

    #[test]
    fn to_console_draws_an_intensity_bar_for_a_firing_neuron() {
        let text = diagram().to_console();
        // The peak-intensity hidden neuron draws a full-width bar of block glyphs.
        assert!(text.contains(&"\u{2588}".repeat(BAR_WIDTH)));
    }

    #[test]
    fn to_console_renders_headings_in_bold() {
        let text = diagram().to_console();
        assert!(text.contains(&bold("Input (2 features)")));
        assert!(text.contains(&bold(&format!("{} (2 units)", RELU.name()))));
    }

    #[test]
    fn to_console_flags_a_sampled_layer_in_its_heading() {
        let weights = Array2::from_shape_fn((50, 2), |(r, _)| r as f32);
        let net = NeuralNetwork::single(Dense::new(weights, Array1::zeros(50), RELU.clone()));
        let options = DiagramOptions {
            max_units: 8,
            ..DiagramOptions::default()
        };
        let text = net
            .activation_diagram(array![1.0, 1.0].view(), &options)
            .unwrap()
            .to_console();
        assert!(text.contains("showing the 8 most active of 50 units"));
    }
}
