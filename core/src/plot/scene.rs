//! Backend-neutral visual model.
//!
//! A [`Figure`] describes *what* to draw — panels, series, ranges, colors — with no
//! commitment to *how* or *where* it is rendered. Renderers (image, console) consume
//! this model behind their own feature flags.

/// An RGB color.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

impl Color {
    /// The color from its red, green and blue components.
    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    /// A distinct color for the categorical index, cycling through a fixed palette.
    pub fn category(index: usize) -> Self {
        CATEGORY_PALETTE[index % CATEGORY_PALETTE.len()]
    }

    /// The color of the training series.
    pub const TRAIN: Color = Color::rgb(214, 39, 40);
    /// The color of the validation series.
    pub const VALIDATION: Color = Color::rgb(255, 127, 14);
    /// The color of the test series.
    pub const TEST: Color = Color::rgb(44, 160, 44);
    /// The color of the decision boundary overlay.
    pub const BOUNDARY: Color = Color::rgb(20, 20, 20);

    /// The soft canvas a rasterized diagram is drawn on.
    pub const CANVAS: Color = Color::rgb(246, 247, 250);
    /// The band shading every other layer column of a diagram.
    pub const LAYER_BAND: Color = Color::rgb(236, 239, 244);

    /// The color of a positive weight or activation.
    pub const POSITIVE: Color = Color::rgb(8, 119, 189);
    /// The color of a negative weight or activation.
    pub const NEGATIVE: Color = Color::rgb(245, 147, 34);

    /// This color scaled toward black by `factor`, clamped to `[0, 1]` — `1`
    /// leaves it unchanged, `0` yields black.
    pub fn scaled(self, factor: f32) -> Color {
        let factor = factor.clamp(0.0, 1.0);
        let scale = |channel: u8| (channel as f32 * factor).round() as u8;
        Color::rgb(scale(self.red), scale(self.green), scale(self.blue))
    }
}

/// A categorical palette (the matplotlib `tab10` colors) for per-class scatter points.
const CATEGORY_PALETTE: [Color; 10] = [
    Color::rgb(31, 119, 180),
    Color::rgb(255, 127, 14),
    Color::rgb(44, 160, 44),
    Color::rgb(214, 39, 40),
    Color::rgb(148, 103, 189),
    Color::rgb(140, 86, 75),
    Color::rgb(227, 119, 194),
    Color::rgb(127, 127, 127),
    Color::rgb(188, 189, 34),
    Color::rgb(23, 190, 207),
];

/// One plotted series within a panel. A series carries an optional label, shown in the
/// panel legend only when set.
#[derive(Debug)]
pub enum Series {
    /// A polyline connecting the points in order.
    Line {
        points: Vec<(f32, f32)>,
        color: Color,
        label: Option<String>,
    },
    /// A scatter of discrete circular markers of the given radius.
    Points {
        points: Vec<(f32, f32)>,
        color: Color,
        label: Option<String>,
        radius: u32,
    },
}

/// A single set of axes with its series and value ranges.
#[derive(Debug)]
pub struct Panel {
    pub title: String,
    pub x_range: (f32, f32),
    pub y_range: (f32, f32),
    pub x_label: Option<String>,
    pub y_label: Option<String>,
    pub show_legend: bool,
    pub series: Vec<Series>,
}

impl Panel {
    /// The width-to-height ratio of the panel's data domain, defaulting to 1
    /// for a degenerate range.
    pub fn data_aspect(&self) -> f32 {
        let width = self.x_range.1 - self.x_range.0;
        let height = self.y_range.1 - self.y_range.0;
        if width > 0.0 && height > 0.0 {
            width / height
        } else {
            1.0
        }
    }
}

/// A complete figure: one or more panels stacked vertically.
#[derive(Debug)]
pub struct Figure {
    pub panels: Vec<Panel>,
    /// Whether renderers should preserve the data aspect ratio rather than fill
    /// their canvas.
    pub preserve_aspect: bool,
}

impl Figure {
    /// A figure whose axes share units, so renderers preserve its data aspect.
    pub fn spatial(panels: Vec<Panel>) -> Self {
        Self {
            panels,
            preserve_aspect: true,
        }
    }

    /// A figure whose axes are independent, so renderers fill their canvas.
    pub fn chart(panels: Vec<Panel>) -> Self {
        Self {
            panels,
            preserve_aspect: false,
        }
    }

    /// The data aspect to preserve — the first panel's width-to-height ratio —
    /// or `None` for a chart or an empty figure.
    pub fn data_aspect(&self) -> Option<f32> {
        if !self.preserve_aspect {
            return None;
        }
        self.panels.first().map(Panel::data_aspect)
    }
}

/// Widens each `[min, max]` range by `padding_factor` of its extent on both sides.
pub(crate) fn add_padding(mins: &[f32], maxs: &[f32], padding_factor: f32) -> (Vec<f32>, Vec<f32>) {
    let mut padded_mins = Vec::with_capacity(mins.len());
    let mut padded_maxs = Vec::with_capacity(maxs.len());

    for (&min, &max) in mins.iter().zip(maxs.iter()) {
        let range = max - min;
        padded_mins.push(min - range * padding_factor);
        padded_maxs.push(max + range * padding_factor);
    }

    (padded_mins, padded_maxs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgb_sets_components() {
        let color = Color::rgb(1, 2, 3);
        assert_eq!(color.red, 1);
        assert_eq!(color.green, 2);
        assert_eq!(color.blue, 3);
    }

    #[test]
    fn category_cycles_through_the_palette() {
        assert_eq!(Color::category(0), CATEGORY_PALETTE[0]);
        assert_eq!(Color::category(9), CATEGORY_PALETTE[9]);
        // Wraps around past the end of the palette.
        assert_eq!(Color::category(10), CATEGORY_PALETTE[0]);
        assert_eq!(Color::category(13), CATEGORY_PALETTE[3]);
    }

    #[test]
    fn scaled_dims_each_channel_toward_black() {
        let color = Color::rgb(200, 100, 40);
        // Half brightness halves every channel; the bounds are the extremes.
        assert_eq!(color.scaled(0.5), Color::rgb(100, 50, 20));
        assert_eq!(color.scaled(1.0), color);
        assert_eq!(color.scaled(0.0), Color::rgb(0, 0, 0));
        // Out-of-range factors clamp rather than overshoot.
        assert_eq!(color.scaled(2.0), color);
    }

    #[test]
    fn add_padding_widens_each_range_symmetrically() {
        let (mins, maxs) = add_padding(&[0.0, 10.0], &[10.0, 30.0], 0.1);
        // Extent 10 → ±1; extent 20 → ±2.
        assert_eq!(mins, vec![-1.0, 8.0]);
        assert_eq!(maxs, vec![11.0, 32.0]);
    }

    #[test]
    fn add_padding_on_empty_bounds_is_empty() {
        let (mins, maxs) = add_padding(&[], &[], 0.05);
        assert!(mins.is_empty());
        assert!(maxs.is_empty());
    }

    fn panel(x_range: (f32, f32), y_range: (f32, f32)) -> Panel {
        Panel {
            title: String::new(),
            x_range,
            y_range,
            x_label: None,
            y_label: None,
            show_legend: false,
            series: Vec::new(),
        }
    }

    #[test]
    fn data_aspect_is_the_width_over_height_ratio() {
        assert_eq!(panel((0.0, 8.0), (0.0, 2.0)).data_aspect(), 4.0);
        assert_eq!(panel((0.0, 10.0), (0.0, 10.0)).data_aspect(), 1.0);
    }

    #[test]
    fn data_aspect_defaults_to_one_for_a_degenerate_range() {
        assert_eq!(panel((1.0, 1.0), (0.0, 5.0)).data_aspect(), 1.0);
        assert_eq!(panel((0.0, 5.0), (2.0, 2.0)).data_aspect(), 1.0);
    }

    #[test]
    fn spatial_figure_data_aspect_reads_the_first_panel() {
        let figure = Figure::spatial(vec![
            panel((0.0, 6.0), (0.0, 2.0)),
            panel((0.0, 1.0), (0.0, 9.0)),
        ]);
        assert_eq!(figure.data_aspect(), Some(3.0));
    }

    #[test]
    fn chart_figure_has_no_data_aspect() {
        // A chart's axes are independent, so there is no aspect to preserve even
        // when its panels carry ranges.
        let figure = Figure::chart(vec![panel((0.0, 6.0), (0.0, 2.0))]);
        assert_eq!(figure.data_aspect(), None);
    }

    #[test]
    fn figure_data_aspect_is_none_when_empty() {
        assert_eq!(Figure::spatial(Vec::new()).data_aspect(), None);
    }
}
