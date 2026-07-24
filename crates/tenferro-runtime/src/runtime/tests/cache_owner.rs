use std::error::Error as StdError;
use std::sync::Arc;

use crate::runtime::{
    CacheOwnerError, CacheOwnerFailure, CacheOwnerId, CacheStats, IdentityKind, RuntimeCacheError,
    RuntimeCacheOwner, RuntimeStateError,
};

#[derive(Debug, thiserror::Error)]
#[error("source a")]
struct SourceA;

#[derive(Debug, thiserror::Error)]
#[error("source b")]
struct SourceB;

#[derive(Debug)]
struct Owner;

impl RuntimeCacheOwner for Owner {
    fn cache_stats(&self) -> Result<CacheStats, CacheOwnerError> {
        Ok(CacheStats {
            entries: 2,
            retained_bytes: 128,
            hits: 3,
            misses: 4,
            evictions: 5,
            clears: 6,
        })
    }

    fn clear_caches(&self) -> Result<(), CacheOwnerError> {
        Ok(())
    }
}

fn owner_id(value: &str) -> CacheOwnerId {
    CacheOwnerId::new(value).unwrap_or_else(|error| panic!("{error}"))
}

fn owner_error_a() -> CacheOwnerError {
    CacheOwnerError::new(Arc::new(SourceA))
}

#[test]
fn cache_owner_ids_validate_namespaced_ascii_with_cache_owner_kind() {
    let owner = CacheOwnerId::new("tenferro.cache.owner").unwrap_or_else(|error| panic!("{error}"));
    assert_eq!(owner.as_str(), "tenferro.cache.owner");

    let error = CacheOwnerId::new("owner").unwrap_err();
    assert_eq!(error.kind(), IdentityKind::CacheOwner);
}

#[test]
fn cache_owner_ids_sort_stably_by_validated_text() {
    let mut owners = [
        owner_id("tenferro.cache.z"),
        owner_id("tenferro.cache.a"),
        owner_id("tenferro.cache.m"),
    ];

    owners.sort();

    assert_eq!(owners[0].as_str(), "tenferro.cache.a");
    assert_eq!(owners[1].as_str(), "tenferro.cache.m");
    assert_eq!(owners[2].as_str(), "tenferro.cache.z");
}

#[test]
fn cache_owner_error_preserves_arc_identity_and_error_source() {
    let source: Arc<dyn StdError + Send + Sync> = Arc::new(SourceA);
    let error = CacheOwnerError::new(source.clone());

    assert!(Arc::ptr_eq(error.source_arc(), &source));
    assert_eq!(
        StdError::source(&error)
            .expect("source should be preserved")
            .to_string(),
        "source a"
    );
    assert!(format!("{error:?}").contains("source a"));
}

#[test]
fn runtime_cache_owner_trait_is_object_safe_and_returns_stats() {
    let owner: Arc<dyn RuntimeCacheOwner> = Arc::new(Owner);
    let stats = owner
        .cache_stats()
        .unwrap_or_else(|error| panic!("{error}"));

    assert_eq!(
        stats,
        CacheStats {
            entries: 2,
            retained_bytes: 128,
            hits: 3,
            misses: 4,
            evictions: 5,
            clears: 6,
        }
    );
    owner
        .clear_caches()
        .unwrap_or_else(|error| panic!("{error}"));
}

#[test]
fn runtime_cache_error_aggregate_preserves_runtime_and_owner_failures_in_order() {
    let error = RuntimeCacheError::Aggregate {
        runtime: Some(RuntimeStateError::Poisoned {
            lock: "runtime.cache",
        }),
        owners: Box::new([
            CacheOwnerFailure {
                owner: owner_id("tenferro.cache.a"),
                source: owner_error_a(),
            },
            CacheOwnerFailure {
                owner: owner_id("tenferro.cache.b"),
                source: CacheOwnerError::new(Arc::new(SourceB)),
            },
        ]),
    };

    let RuntimeCacheError::Aggregate { runtime, owners } = error;
    assert!(matches!(
        runtime,
        Some(RuntimeStateError::Poisoned {
            lock: "runtime.cache"
        })
    ));
    assert_eq!(owners[0].owner.as_str(), "tenferro.cache.a");
    assert_eq!(owners[0].source.to_string(), "source a");
    assert_eq!(owners[1].owner.as_str(), "tenferro.cache.b");
    assert_eq!(owners[1].source.to_string(), "source b");
}
