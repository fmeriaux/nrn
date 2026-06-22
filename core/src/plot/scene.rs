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
}

impl Figure {
    /// The data aspect of the first panel, or `None` when the figure is empty.
    pub fn data_aspect(&self) -> Option<f32> {
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
    fn figure_data_aspect_reads_the_first_panel() {
        let figure = Figure {
            panels: vec![panel((0.0, 6.0), (0.0, 2.0)), panel((0.0, 1.0), (0.0, 9.0))],
        };
        assert_eq!(figure.data_aspect(), Some(3.0));
    }

    #[test]
    fn figure_data_aspect_is_none_when_empty() {
        let figure = Figure { panels: Vec::new() };
        assert_eq!(figure.data_aspect(), None);
    }
}
