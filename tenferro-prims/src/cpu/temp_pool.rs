use std::any::{Any, TypeId};
use std::collections::{BTreeMap, HashMap};

/// Reusable CPU-side temporary storage for typed vectors.
///
/// The pool is crate-private for now so later CPU execution helpers can use it
/// without exposing it as part of the public API.
#[derive(Default)]
#[allow(dead_code)]
pub(crate) struct TempPool {
    typed_vecs: HashMap<TypeId, BTreeMap<usize, Vec<Box<dyn Any + Send>>>>,
}

#[allow(dead_code)]
impl TempPool {
    /// Take a reusable temporary vector with at least `len` capacity.
    pub(crate) fn take_vec<T: Send + 'static>(&mut self, len: usize) -> Vec<T> {
        let type_id = TypeId::of::<T>();
        let mut taken = None;
        let mut remove_type_bucket = false;

        if let Some(bucket) = self.typed_vecs.get_mut(&type_id) {
            taken = take_typed_vec_from_bucket(bucket, len);
            remove_type_bucket = bucket.is_empty();
        }

        if remove_type_bucket {
            self.typed_vecs.remove(&type_id);
        }

        taken.unwrap_or_else(|| Vec::with_capacity(len))
    }

    /// Return a temporary vector to the pool for later reuse.
    pub(crate) fn put_vec<T: Send + 'static>(&mut self, mut vec: Vec<T>) {
        let cap = vec.capacity();
        if cap == 0 {
            return;
        }
        vec.clear();
        self.typed_vecs
            .entry(TypeId::of::<T>())
            .or_default()
            .entry(cap)
            .or_default()
            .push(Box::new(vec));
    }
}

#[allow(dead_code)]
fn take_typed_vec_from_bucket<T: Send + 'static>(
    bucket: &mut BTreeMap<usize, Vec<Box<dyn Any + Send>>>,
    min_capacity: usize,
) -> Option<Vec<T>> {
    let cap = bucket.range(min_capacity..).next().map(|(&cap, _)| cap)?;
    let boxed = {
        let entries = bucket.get_mut(&cap)?;
        let boxed = entries.pop()?;
        boxed
    };
    if bucket.get(&cap).is_some_and(|entries| entries.is_empty()) {
        bucket.remove(&cap);
    }
    Some(
        *boxed
            .downcast::<Vec<T>>()
            .expect("typed temp pool bucket had wrong type"),
    )
}
