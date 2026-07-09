//! Activation diagram: a neural network's forward pass on a single instance,
//! as a backend-neutral model of colored neurons and weighted connections.
//!
//! [`ActivationDiagram::from_activations`] records, for every layer, each neuron's
//! activation and the connections feeding it, showing the values it is given.
//! Renderers (console, image) consume the diagram behind their own feature flags.
//! Layers larger than [`DiagramOptions::max_units`] keep only their most active
//! neurons so the model stays drawable.

use crate::classification::ClassifierActivations;
use crate::data::scalers::Scaler;
use crate::model::{Activations, InputShapeMismatch, NeuralNetwork, PredictionError, Predictor};
use crate::plot::scene::Color;
use ndarray::{ArrayView1, ArrayView2, Axis, Ix2};

/// Activation magnitudes at or below this count as a silent neuron.
const FIRING_EPSILON: f32 = 1e-6;

/// The floor brightness of a firing neuron's marker, so a weakly active neuron
/// stays visible rather than fading to black.
const MARKER_FLOOR: f32 = 0.35;

/// How an [`ActivationDiagram`] caps and prunes a network too large to draw in full.
#[derive(Clone, Copy, Debug)]
pub struct DiagramOptions {
    /// Maximum neurons shown per layer; larger layers keep only their most active.
    pub max_units: usize,
    /// Connections whose normalized contribution `|weight × source activation|`
    /// falls below this are dropped, so a strong weight from a silent neuron
    /// (contributing nothing to this instance) is pruned away.
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
                "{name} (showing the {} most active of {} {noun}s)",
                self.units.len(),
                self.total_units
            )
        } else {
            let plural = if self.total_units == 1 { "" } else { "s" };
            format!("{name} ({} {noun}{plural})", self.total_units)
        }
    }

    /// A compact heading for space-constrained renderers: the role name and its
    /// neuron count, marking a sampled layer as `shown/total active`.
    pub fn short_heading(&self) -> String {
        let name = match &self.role {
            LayerRole::Input => "Input",
            LayerRole::Activated(activation) => activation.as_str(),
        };
        if self.is_sampled() {
            format!("{name} ({}/{} active)", self.units.len(), self.total_units)
        } else {
            format!("{name} ({})", self.total_units)
        }
    }
}

/// A neural network's forward pass on one instance: per-layer neuron activations and
/// the connections between them.
#[derive(Clone, Debug)]
pub struct ActivationDiagram {
    /// The layers from input to output, in order.
    pub layers: Vec<DiagramLayer>,
}

impl ActivationDiagram {
    /// Builds a diagram from `network` and the per-layer `activations` of a single forward
    /// pass, capping and pruning per `options`. Each neuron shows the value it is given.
    pub fn from_activations(
        network: &NeuralNetwork,
        activations: &Activations,
        options: &DiagramOptions,
    ) -> ActivationDiagram {
        let stages: Vec<ArrayView2<f32>> = activations
            .stages()
            .iter()
            .map(|values| {
                values
                    .view()
                    .into_dimensionality::<Ix2>()
                    .expect("the diagram visualizes rank-2 (features, samples) activations")
            })
            .collect();
        let shown: Vec<Vec<usize>> = stages
            .iter()
            .map(|values| shown_indices(values.column(0), options.max_units))
            .collect();

        let layers = stages
            .iter()
            .enumerate()
            .map(|(stage, values)| {
                let column = values.column(0);
                let (role, edges) = if stage == 0 {
                    (LayerRole::Input, Vec::new())
                } else {
                    let layer = &network.layers()[stage - 1];
                    let edges = match layer.weight_matrix() {
                        Some(weights) => edges_for(
                            weights,
                            stages[stage - 1].column(0),
                            &shown[stage - 1],
                            &shown[stage],
                            options.min_edge_magnitude,
                        ),
                        None => Vec::new(),
                    };
                    (
                        LayerRole::Activated(
                            layer.activation_name().unwrap_or_default().to_string(),
                        ),
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

        ActivationDiagram { layers }
    }
}

impl NeuralNetwork {
    /// Builds an [`ActivationDiagram`] from a forward pass on a single `input` instance,
    /// showing each layer's raw activations.
    ///
    /// # Errors
    /// [`InputShapeMismatch`] when `input`'s length does not match the network's input size.
    pub fn activation_diagram(
        &self,
        input: ArrayView1<f32>,
        options: &DiagramOptions,
    ) -> Result<ActivationDiagram, InputShapeMismatch> {
        let activations = self.forward(input.insert_axis(Axis(1)))?;
        Ok(ActivationDiagram::from_activations(
            self,
            &activations,
            options,
        ))
    }
}

impl Predictor {
    /// Builds an [`ActivationDiagram`] for one raw `input`, applying the scaler first when
    /// present and showing the output layer as class probabilities.
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
            // Scale the lone instance: features on the leading axis, a single sample.
            let mut expanded = input.view_mut().insert_axis(Axis(1)).into_dyn();
            scaler.apply_inplace(expanded.view_mut())?;
        }
        let activations = self
            .network
            .forward(input.view().insert_axis(Axis(1)))?
            .with_probabilities();
        Ok(ActivationDiagram::from_activations(
            &self.network,
            &activations,
            options,
        ))
    }
}

/// The neuron indices shown for a layer under `cap`: every neuron when the layer
/// fits, otherwise the `cap` most active — largest activation magnitude, ties
/// broken by the earlier index — returned in ascending index order for a stable
/// top-to-bottom layout.
fn shown_indices(column: ArrayView1<f32>, cap: usize) -> Vec<usize> {
    let cap = cap.max(1);
    let n = column.len();
    if n <= cap {
        return (0..n).collect();
    }
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| column[b].abs().total_cmp(&column[a].abs()).then(a.cmp(&b)));
    indices.truncate(cap);
    indices.sort_unstable();
    indices
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

/// The connections from the previous layer's shown neurons to this layer's, with
/// the drawn width tracking each weight's magnitude (normalized by the layer's
/// peak weight) and pruning by contribution: a connection whose
/// `|weight × source activation|`, normalized by the layer's peak contribution,
/// falls below `min_magnitude` is dropped. `weights` is
/// `(this layer's neurons, previous layer's neurons)` and `prev_activations`
/// carries every source neuron's activation.
fn edges_for(
    weights: ArrayView2<f32>,
    prev_activations: ArrayView1<f32>,
    shown_prev: &[usize],
    shown_cur: &[usize],
    min_magnitude: f32,
) -> Vec<Edge> {
    let weight_peak = weights.iter().fold(0.0_f32, |max, &w| max.max(w.abs()));
    let contribution_peak = weights
        .indexed_iter()
        .fold(0.0_f32, |max, ((_, prev), &w)| {
            max.max((w * prev_activations[prev]).abs())
        });
    let mut edges = Vec::new();
    for (target, &cur) in shown_cur.iter().enumerate() {
        for (source, &prev) in shown_prev.iter().enumerate() {
            let weight = weights[(cur, prev)];
            let contribution = if contribution_peak > 0.0 {
                (weight * prev_activations[prev]).abs() / contribution_peak
            } else {
                0.0
            };
            if contribution >= min_magnitude {
                let magnitude = if weight_peak > 0.0 {
                    weight.abs() / weight_peak
                } else {
                    0.0
                };
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
    use crate::activations::{Activation, RELU, SOFTMAX};
    use crate::layers::Dense;
    use crate::model::NeuralNetwork;
    use ndarray::{Array1, Array2, array};

    fn relu_network(weights: Array2<f32>, biases: Array1<f32>) -> NeuralNetwork {
        NeuralNetwork::single(Dense::new(weights, biases, RELU.clone()))
    }

    #[test]
    fn shown_indices_returns_every_neuron_when_it_fits() {
        assert_eq!(
            shown_indices(array![0.0, 1.0, 2.0].view(), 8),
            vec![0, 1, 2]
        );
        assert_eq!(
            shown_indices(array![0.0, 1.0, 2.0, 3.0].view(), 4),
            vec![0, 1, 2, 3]
        );
    }

    #[test]
    fn shown_indices_keeps_the_most_active_neurons_in_index_order() {
        // The two strongest by magnitude are index 1 (|-0.9|) and index 3 (0.8),
        // returned in ascending index order rather than activation order.
        let column = array![0.1, -0.9, 0.2, 0.8, -0.3];
        assert_eq!(shown_indices(column.view(), 2), vec![1, 3]);
        // A cap of one keeps the single most active neuron.
        assert_eq!(shown_indices(column.view(), 1), vec![1]);
    }

    #[test]
    fn diagram_has_an_input_layer_then_one_layer_per_neuron_layer() {
        let net = relu_network(array![[1.0, 0.0], [2.0, 0.0]], array![0.0, 0.0]);
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
        let net = relu_network(array![[1.0, 0.0]], array![0.0]);
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
        // Row 1's negative pre-activation is clamped to zero by ReLU. Tested on a
        // hidden layer, whose intensity is peak-normalized (an output layer's
        // intensity tracks its probability instead).
        let net = NeuralNetwork::single(Dense::new(
            array![[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0]],
            array![0.0, 0.0, 0.0],
            RELU.clone(),
        ))
        .with_layer(Dense::new(
            array![[1.0, 0.0, 0.0]],
            array![0.0],
            RELU.clone(),
        ));
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
        let net = relu_network(
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
        let net = relu_network(weights, Array1::zeros(50));
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
        // Activations equal the neuron index here, so the eight most active are
        // the highest indices, kept in ascending order.
        assert_eq!(layer.units[0].index, 42);
        assert_eq!(layer.units[7].index, 49);
        // Edges only connect the shown neurons: 8 targets x 2 sources.
        assert_eq!(layer.edges.len(), 16);
        // The input layer is small enough to show in full.
        assert!(!diagram.layers[0].is_sampled());
    }

    #[test]
    fn a_weak_edge_threshold_prunes_low_magnitude_connections() {
        let net = relu_network(
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
    fn the_edge_threshold_prunes_by_contribution_not_weight_alone() {
        // Layer 1 leaves neuron 0 firing (1.0) and neuron 1 silent (0.0). Layer 2
        // weights a tiny link from the firing neuron and a strong one from the
        // silent one: contribution flows only through neuron 0, so that is the
        // edge a weight-only threshold would have wrongly dropped.
        let net = NeuralNetwork::single(Dense::new(
            array![[1.0], [-1.0]],
            array![0.0, 0.0],
            RELU.clone(),
        ))
        .with_layer(Dense::new(array![[0.1, 5.0]], array![0.0], RELU.clone()));
        let options = DiagramOptions {
            min_edge_magnitude: 0.5,
            ..DiagramOptions::default()
        };
        let diagram = net
            .activation_diagram(array![1.0].view(), &options)
            .unwrap();

        // Only the connection from the firing neuron survives; the strong weight
        // out of the silent neuron carries no contribution and is pruned.
        let output = &diagram.layers[2];
        assert_eq!(output.edges.len(), 1);
        assert_eq!(output.edges[0].source, 0);
        assert_eq!(output.edges[0].weight, 0.1);
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
    fn the_predictor_diagram_shows_the_output_layer_as_probabilities() {
        // Output logits [0.3, 0.4]: the predictor converts the output stage, so its neurons
        // carry the softmax probabilities rather than the raw logits.
        let net = relu_network(array![[0.3, 0.0], [0.4, 0.0]], array![0.0, 0.0]);
        let predictor = Predictor::new(net, None);
        let diagram = predictor
            .activation_diagram(array![1.0, 0.0].view(), &DiagramOptions::default())
            .unwrap();

        let probabilities = SOFTMAX.apply(array![[0.3], [0.4]].view().into_dyn());
        let output = diagram.layers.last().unwrap();
        assert_eq!(output.units[0].value, probabilities[[0, 0]]);
        assert_eq!(output.units[1].value, probabilities[[1, 0]]);
    }

    #[test]
    fn a_scaler_free_predictor_diagram_feeds_the_raw_input_through() {
        let net = relu_network(array![[1.0, 0.0], [3.0, 0.0]], array![0.0, 0.0]);
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
    }

    #[test]
    fn a_predictor_diagram_rejects_an_instance_of_the_wrong_size() {
        let net = relu_network(array![[1.0, 0.0], [3.0, 0.0]], array![0.0, 0.0]);
        let predictor = Predictor::new(net, None);

        // The network expects two features; a one-feature instance is rejected.
        // Without a scaler the mismatch surfaces from the network.
        let error = predictor
            .activation_diagram(array![1.0].view(), &DiagramOptions::default())
            .unwrap_err();
        assert_eq!(
            error,
            PredictionError::Network(InputShapeMismatch {
                expected: vec![2],
                found: vec![1]
            })
        );
    }
}
