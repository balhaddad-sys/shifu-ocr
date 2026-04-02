use pyo3::prelude::*;

mod arena;
mod neuron;
mod field;

/// Shifu Neural — Rust core for the neural field.
#[pymodule]
fn shifu_neural(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<field::NeuralField>()?;
    m.add_class::<field::NeuronProxy>()?;
    m.add_class::<field::NeuronsProxyOwned>()?;
    Ok(())
}
