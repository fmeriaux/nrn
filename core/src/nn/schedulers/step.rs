use crate::schedulers::Scheduler;
use crate::training::LearningRate;

pub struct StepDecay {
    initial: LearningRate,
    steps: usize,
    current_step: usize,
    decay_factor: f32,
}

impl StepDecay {
    /// Creates a new [`StepDecay`] scheduler.
    ///
    /// # Panics
    /// Will panic if `steps` is zero or if `decay_factor` is not in (0, 1).
    ///
    pub fn new(initial: LearningRate, steps: usize, decay_factor: f32) -> Self {
        assert!(steps > 0, "Steps must be greater than zero.");
        assert!(
            decay_factor > 0.0 && decay_factor < 1.0,
            "Decay factor must be in (0, 1)."
        );

        Self {
            initial,
            steps,
            current_step: 0,
            decay_factor,
        }
    }
}

impl Scheduler for StepDecay {
    fn step(&mut self) -> LearningRate {
        let learning_rate = LearningRate::new(
            self.initial.value() * self.decay_factor.powi((self.current_step / self.steps) as i32),
        );
        self.current_step += 1;
        learning_rate
    }
}
