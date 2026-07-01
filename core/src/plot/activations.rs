//! Activation diagram: a neural network's forward pass on a single instance,
//! as a backend-neutral model of colored neurons and weighted connections.
//!
//! [`NeuralNetwork::activation_diagram`] runs the instance through the network
//! and records, for every layer, each neuron's activation and the connections
//! feeding it. Renderers (console, image) consume the diagram behind their own
//! feature flags. Layers larger than [`DiagramOptions::max_units`] are sampled
//! evenly so the model stays drawable.

use crate::classification::Classification;
use crate::data::scalers::Scaler;
use crate::model::{FeatureCountMismatch, NeuralNetwork, PredictionError, Predictor};
use crate::plot::scene::Color;
use ndarray::{Array2, ArrayView1, Axis};

/// Activation magnitudes at or below this count as a silent neuron.
const FIRING_EPSILON: f32 = 1e-6;

/// The floor brightness of a firing neuron's marker, so a weakly active neuron
/// stays visible rather than fading to black.
const MARKER_FLOOR: f32 = 0.35;

/// How an [`ActivationDiagram`] caps and prunes a network too large to draw in full.
#[derive(Clone, Copy, Debug)]
pub struct DiagramOptions {
    /// Maximum neurons shown per layer; larger layers are sampled evenly.
    pub max_units: usize,
    /// Connections whose normalized magnitude falls below this are dropped.
    pub min_edge_magnitude: f32,
}

impl Default for DiagramOptions {
    fn default() -> Self {
        Self {
            max_units: 24,
            min_edge_magnitude: 0.0,
        }
    }
}

/// One neuron's state in the forward pass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Unit {
    /// The neuron's position in its layer, before any sampling.
    pub index: usize,
    /// The neuron's activation value.
    pub value: f32,
    /// The activation magnitude normalized to `[0, 1]` within its layer.
    pub intensity: f32,
    /// Whether the neuron's magnitude exceeds the silence threshold.
    pub firing: bool,
}

impl Unit {
    /// This neuron's marker color: positive activations blue, negative orange,
    /// dimmed toward black by intensity but never below a visibility floor.
    pub fn marker_color(&self) -> Color {
        let base = if self.value < 0.0 {
            Color::NEGATIVE
        } else {
            Color::POSITIVE
        };
        base.scaled(MARKER_FLOOR + (1.0 - MARKER_FLOOR) * self.intensity)
    }
}

/// One weighted connection from a source neuron in the previous layer to a target
/// neuron in this layer, indexing the two layers' shown neurons.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Edge {
    /// Position of the source neuron among the previous layer's shown units.
    pub source: usize,
    /// Position of the target neuron among this layer's shown units.
    pub target: usize,
    /// The connection's weight.
    pub weight: f32,
    /// The weight magnitude normalized to `[0, 1]` within this layer's weights.
    pub magnitude: f32,
}

impl Edge {
    /// This connection's color: blue for a positive weight, orange for a negative one.
    pub fn color(&self) -> Color {
        if self.weight < 0.0 {
            Color::NEGATIVE
        } else {
            Color::POSITIVE
        }
    }
}

/// What a layer represents in the forward pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerRole {
    /// The raw input features fed to the network.
    Input,
    /// A neuron layer applying the named activation.
    Activated(String),
}

/// One layer of the diagram: its shown neurons and the connections feeding them.
#[derive(Clone, Debug)]
pub struct DiagramLayer {
    /// What the layer represents.
    pub role: LayerRole,
    /// The neurons shown for this layer, in order.
    pub units: Vec<Unit>,
    /// The neuron count before any sampling.
    pub total_units: usize,
    /// Connections from the previous layer's shown neurons; empty for the input layer.
    pub edges: Vec<Edge>,
}

impl DiagramLayer {
    /// Whether this layer's neurons were sampled down from a larger layer.
    pub fn is_sampled(&self) -> bool {
        self.units.len() < self.total_units
    }

    /// A heading naming the layer's role and how many of its neurons are shown,
    /// flagging when the layer was sampled down to fit.
    pub fn heading(&self) -> String {
        let (name, noun) = match &self.role {
            LayerRole::Input => ("Input", "feature"),
            LayerRole::Activated(activation) => (activation.as_str(), "unit"),
        };
        if self.is_sampled() {
            format!(
                "{name} (showing {} of {} {noun}s)",
                self.units.len(),
                self.total_units
            )
        } else {
            let plural = if self.total_units == 1 { "" } else { "s" };
            format!("{name} ({} {noun}{plural})", self.total_units)
        }
    }
}

/// A neural network's forward pass on one instance: per-layer neuron activations,
/// the connections between them, and the resulting class ranking.
#[derive(Clone, Debug)]
pub struct ActivationDiagram {
    /// The layers from input to output, in order.
    pub layers: Vec<DiagramLayer>,
    /// The class probabilities, ranked most likely first.
    pub prediction: Classification,
}

impl NeuralNetwork {
    /// Builds an [`ActivationDiagram`] from a forward pass on a single `input`
    /// instance, capping and pruning per `options`.
    ///
    /// # Errors
    /// [`FeatureCountMismatch`] when `input`'s length does not match the network's input size.
    pub fn activation_diagram(
        &self,
        input: ArrayView1<f32>,
        options: &DiagramOptions,
    ) -> Result<ActivationDiagram, FeatureCountMismatch> {
        let activations = self.forward(input.insert_axis(Axis(1)))?;
        let shown: Vec<Vec<usize>> = activations
            .iter()
            .map(|values| shown_indices(values.nrows(), options.max_units))
            .collect();

        let layers = activations
            .iter()
            .enumerate()
            .map(|(stage, values)| {
                let column = values.column(0);
                let (role, edges) = if stage == 0 {
                    (LayerRole::Input, Vec::new())
                } else {
                    let layer = &self.layers[stage - 1];
                    let edges = edges_for(
                        &layer.weights,
                        &shown[stage - 1],
                        &shown[stage],
                        options.min_edge_magnitude,
                    );
                    (
                        LayerRole::Activated(layer.activation.name().to_string()),
                        edges,
                    )
                };
                DiagramLayer {
                    role,
                    units: units_for(column, &shown[stage]),
                    total_units: column.len(),
                    edges,
                }
            })
            .collect();

        let prediction = Classification::from_outputs(activations.last().unwrap().column(0));
        Ok(ActivationDiagram { layers, prediction })
    }
}

impl Predictor {
    /// Builds an [`ActivationDiagram`] for one raw `input`, applying the scaler
    /// first when present so the diagram reflects what the network sees.
    ///
    /// # Errors
    /// [`PredictionError::Scaling`] when a scaler is present and the input does not match
    /// its fitted feature count, or [`PredictionError::Network`] when it does not match the
    /// network's input size.
    pub fn activation_diagram(
        &self,
        input: ArrayView1<f32>,
        options: &DiagramOptions,
    ) -> Result<ActivationDiagram, PredictionError> {
        let mut input = input.to_owned();
        if let Some(scaler) = &self.scaler {
            scaler.apply_single_inplace(input.view_mut())?;
        }
        Ok(self.network.activation_diagram(input.view(), options)?)
    }
}

/// The neuron indices shown for a layer of `n` neurons under `cap`: every neuron
/// when it fits, otherwise `cap` indices spread evenly across the layer.
fn shown_indices(n: usize, cap: usize) -> Vec<usize> {
    let cap = cap.max(1);
    if n <= cap {
        return (0..n).collect();
    }
    if cap == 1 {
        return vec![0];
    }
    (0..cap).map(|i| i * (n - 1) / (cap - 1)).collect()
}

/// The shown neurons of `column`, with activation magnitude normalized to `[0, 1]`
/// by the layer's peak magnitude.
fn units_for(column: ArrayView1<f32>, shown: &[usize]) -> Vec<Unit> {
    let peak = column
        .iter()
        .fold(0.0_f32, |max, &value| max.max(value.abs()));
    shown
        .iter()
        .map(|&index| {
            let value = column[index];
            Unit {
                index,
                value,
                intensity: if peak > 0.0 { value.abs() / peak } else { 0.0 },
                firing: value.abs() > FIRING_EPSILON,
            }
        })
        .collect()
}

/// The connections from the previous layer's shown neurons to this layer's,
/// normalized by the layer's peak weight magnitude and pruned below `min_magnitude`.
/// `weights` is `(this layer's neurons, previous layer's neurons)`.
fn edges_for(
    weights: &Array2<f32>,
    shown_prev: &[usize],
    shown_cur: &[usize],
    min_magnitude: f32,
) -> Vec<Edge> {
    let peak = weights.iter().fold(0.0_f32, |max, &w| max.max(w.abs()));
    let mut edges = Vec::new();
    for (target, &cur) in shown_cur.iter().enumerate() {
        for (source, &prev) in shown_prev.iter().enumerate() {
            let weight = weights[(cur, prev)];
            let magnitude = if peak > 0.0 { weight.abs() / peak } else { 0.0 };
            if magnitude >= min_magnitude {
                edges.push(Edge {
                    source,
                    target,
                    weight,
                    magnitude,
                });
            }
        }
    }
    edges
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::{Activation, RELU};
    use crate::model::{NeuralNetwork, NeuronLayer};
    use ndarray::{Array1, Array2, array};

    fn relu_layer(weights: Array2<f32>, biases: Array1<f32>) -> NeuralNetwork {
        NeuralNetwork {
            layers: vec![NeuronLayer {
                weights,
                biases,
                activation: RELU.clone(),
            }],
        }
    }

    #[test]
    fn shown_indices_returns_every_neuron_when_it_fits() {
        assert_eq!(shown_indices(3, 8), vec![0, 1, 2]);
        assert_eq!(shown_indices(4, 4), vec![0, 1, 2, 3]);
    }

    #[test]
    fn shown_indices_spreads_evenly_when_capped() {
        assert_eq!(shown_indices(10, 4), vec![0, 3, 6, 9]);
        // A cap of one keeps the first neuron.
        assert_eq!(shown_indices(10, 1), vec![0]);
    }

    #[test]
    fn diagram_has_an_input_layer_then_one_layer_per_neuron_layer() {
        let net = relu_layer(array![[1.0, 0.0], [2.0, 0.0]], array![0.0, 0.0]);
        let diagram = net
            .activation_diagram(array![1.0, 1.0].view(), &DiagramOptions::default())
            .unwrap();

        assert_eq!(diagram.layers.len(), 2);
        assert_eq!(diagram.layers[0].role, LayerRole::Input);
        assert!(diagram.layers[0].edges.is_empty());
        assert_eq!(
            diagram.layers[1].role,
            LayerRole::Activated(RELU.name().to_string())
        );
    }

    #[test]
    fn input_layer_carries_the_raw_instance_values() {
        let net = relu_layer(array![[1.0, 0.0]], array![0.0]);
        let diagram = net
            .activation_diagram(array![-0.5, 2.0].view(), &DiagramOptions::default())
            .unwrap();

        let input = &diagram.layers[0];
        assert_eq!(input.units.len(), 2);
        assert_eq!(input.units[0].value, -0.5);
        assert_eq!(input.units[1].value, 2.0);
        // Magnitudes are normalized by the layer's peak (|2.0|).
        assert_eq!(input.units[0].intensity, 0.25);
        assert_eq!(input.units[1].intensity, 1.0);
    }

    #[test]
    fn a_dead_relu_neuron_is_silent_with_zero_intensity() {
        // Row 1's negative pre-activation is clamped to zero by ReLU.
        let net = relu_layer(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
        );
        let diagram = net
            .activation_diagram(array![1.0, 1.0].view(), &DiagramOptions::default())
            .unwrap();

        let layer = &diagram.layers[1];
        assert_eq!(layer.units[0].value, 1.0);
        assert!(layer.units[0].firing);
        assert_eq!(layer.units[0].intensity, 0.5);

        assert_eq!(layer.units[1].value, 0.0);
        assert!(!layer.units[1].firing);
        assert_eq!(layer.units[1].intensity, 0.0);

        assert_eq!(layer.units[2].intensity, 1.0);
    }

    #[test]
    fn edges_keep_weight_sign_and_normalize_magnitude_by_layer_peak() {
        let net = relu_layer(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
        );
        let diagram = net
            .activation_diagram(array![1.0, 1.0].view(), &DiagramOptions::default())
            .unwrap();

        let layer = &diagram.layers[1];
        // Three target neurons, two source inputs: a full bipartite set.
        assert_eq!(layer.edges.len(), 6);

        let strongest = layer
            .edges
            .iter()
            .find(|e| e.target == 2 && e.source == 0)
            .unwrap();
        assert_eq!(strongest.weight, 2.0);
        assert_eq!(strongest.magnitude, 1.0);

        let negative = layer
            .edges
            .iter()
            .find(|e| e.target == 1 && e.source == 0)
            .unwrap();
        assert_eq!(negative.weight, -1.0);
        assert_eq!(negative.magnitude, 0.5);
    }

    #[test]
    fn large_layers_are_sampled_down_to_the_cap() {
        let weights = Array2::from_shape_fn((50, 2), |(r, _)| r as f32);
        let net = relu_layer(weights, Array1::zeros(50));
        let options = DiagramOptions {
            max_units: 8,
            ..DiagramOptions::default()
        };
        let diagram = net
            .activation_diagram(array![1.0, 1.0].view(), &options)
            .unwrap();

        let layer = &diagram.layers[1];
        assert_eq!(layer.total_units, 50);
        assert_eq!(layer.units.len(), 8);
        assert!(layer.is_sampled());
        // Sampled units keep their original neuron index, not their shown position.
        assert_eq!(layer.units[0].index, 0);
        assert_eq!(layer.units[7].index, 49);
        // Edges only connect the shown neurons: 8 targets x 2 sources.
        assert_eq!(layer.edges.len(), 16);
        // The input layer is small enough to show in full.
        assert!(!diagram.layers[0].is_sampled());
    }

    #[test]
    fn a_weak_edge_threshold_prunes_low_magnitude_connections() {
        let net = relu_layer(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
        );
        let options = DiagramOptions {
            min_edge_magnitude: 0.75,
            ..DiagramOptions::default()
        };
        let diagram = net
            .activation_diagram(array![1.0, 1.0].view(), &options)
            .unwrap();

        // Only the peak-magnitude connection (the |2.0| weight) survives.
        let layer = &diagram.layers[1];
        assert!(layer.edges.iter().all(|e| e.magnitude >= 0.75));
        assert_eq!(layer.edges.len(), 1);
        assert_eq!(layer.edges[0].magnitude, 1.0);
    }

    #[test]
    fn a_units_marker_color_tracks_its_sign_and_intensity() {
        let positive = Unit {
            index: 0,
            value: 1.0,
            intensity: 1.0,
            firing: true,
        };
        let negative = Unit {
            index: 1,
            value: -1.0,
            intensity: 1.0,
            firing: true,
        };
        // At full intensity the marker is its base hue, blue for positive and orange
        // for negative.
        assert_eq!(positive.marker_color(), Color::POSITIVE);
        assert_eq!(negative.marker_color(), Color::NEGATIVE);
        // A weakly active neuron stays above the visibility floor (never black).
        let faint = Unit {
            intensity: 0.0,
            ..positive
        };
        assert_eq!(faint.marker_color(), Color::POSITIVE.scaled(MARKER_FLOOR));
    }

    #[test]
    fn an_edges_color_tracks_its_weight_sign() {
        let positive = Edge {
            source: 0,
            target: 0,
            weight: 0.5,
            magnitude: 0.5,
        };
        let negative = Edge {
            weight: -0.5,
            ..positive
        };
        assert_eq!(positive.color(), Color::POSITIVE);
        assert_eq!(negative.color(), Color::NEGATIVE);
    }

    #[test]
    fn the_prediction_ranks_the_output_activations() {
        let net = relu_layer(array![[1.0, 0.0], [3.0, 0.0]], array![0.0, 0.0]);
        let diagram = net
            .activation_diagram(array![1.0, 0.0].view(), &DiagramOptions::default())
            .unwrap();

        // Output activations [1.0, 3.0]: class 1 leads.
        assert_eq!(diagram.prediction.top().0, 1);
    }

    #[test]
    fn a_scaler_free_predictor_diagram_matches_the_bare_network() {
        let net = relu_layer(array![[1.0, 0.0], [3.0, 0.0]], array![0.0, 0.0]);
        let predictor = Predictor::new(net.clone(), None);
        let options = DiagramOptions::default();

        let from_predictor = predictor
            .activation_diagram(array![1.0, 0.0].view(), &options)
            .unwrap();
        let from_network = net
            .activation_diagram(array![1.0, 0.0].view(), &options)
            .unwrap();

        // Without a scaler the predictor feeds the raw input straight through.
        assert_eq!(from_predictor.layers[0].units, from_network.layers[0].units);
        assert_eq!(from_predictor.prediction, from_network.prediction);
    }

    #[test]
    fn a_predictor_diagram_rejects_an_instance_of_the_wrong_size() {
        let net = relu_layer(array![[1.0, 0.0], [3.0, 0.0]], array![0.0, 0.0]);
        let predictor = Predictor::new(net, None);

        // The network expects two features; a one-feature instance is rejected.
        // Without a scaler the mismatch surfaces from the network.
        let error = predictor
            .activation_diagram(array![1.0].view(), &DiagramOptions::default())
            .unwrap_err();
        assert_eq!(
            error,
            PredictionError::Network(FeatureCountMismatch {
                expected: 2,
                found: 1
            })
        );
    }
}
