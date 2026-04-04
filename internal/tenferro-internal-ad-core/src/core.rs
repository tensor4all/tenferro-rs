pub use chainrules_core::NodeId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AdMode {
    Primal,
    Forward,
    Reverse,
}

#[doc(hidden)]
pub struct AdValue<T> {
    primal: T,
    tangent: Option<T>,
    mode: AdMode,
}

impl<T> AdValue<T> {
    pub fn forward(primal: T, tangent: T) -> Self {
        Self {
            primal,
            tangent: Some(tangent),
            mode: AdMode::Forward,
        }
    }

    pub fn mode(&self) -> AdMode {
        self.mode
    }

    pub fn primal_ref(&self) -> &T {
        &self.primal
    }

    pub fn tangent_ref(&self) -> Option<&T> {
        self.tangent.as_ref()
    }

    pub fn map_preserving_metadata<U>(self, f: impl Fn(T) -> U) -> AdValue<U> {
        AdValue {
            primal: f(self.primal),
            tangent: self.tangent.map(f),
            mode: self.mode,
        }
    }
}
