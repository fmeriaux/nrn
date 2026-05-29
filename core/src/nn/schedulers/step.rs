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
    /// Returns the learning rate for the current step and advances the counter.
    ///
    /// The decay is applied in discrete plateaus: `lr = initial * decay_factor^floor(t / steps)`.
    /// The learning rate stays constant for `steps` steps, then drops by `decay_factor`, and so on.
    /// For a continuous exponential decay, use `CosineAnnealing` instead.
    fn step(&mut self) -> LearningRate {
        let learning_rate = LearningRate::new(
            self.initial.value()
                * self
                    .decay_factor
                    .powi((self.current_step / self.steps) as i32),
        );
        self.current_step += 1;
        learning_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_stays_constant_within_each_period() {
        // initial=0.1, steps=3, decay=0.5
        // Steps 0-2: floor(t/3)=0 → LR = 0.1 * 0.5^0 = 0.1
        // Steps 3-5: floor(t/3)=1 → LR = 0.1 * 0.5^1 = 0.05
        let mut sched = StepDecay::new(LearningRate::new(0.1), 3, 0.5);
        for _ in 0..3 {
            assert!((sched.step().value() - 0.1).abs() < 1e-6);
        }
        for _ in 0..3 {
            assert!((sched.step().value() - 0.05).abs() < 1e-6);
        }
    }

    #[test]
    fn decay_never_reaches_zero() {
        let mut sched = StepDecay::new(LearningRate::new(0.1), 1, 0.5);
        for _ in 0..100 {
            assert!(sched.step().value() > 0.0);
        }
    }
}
