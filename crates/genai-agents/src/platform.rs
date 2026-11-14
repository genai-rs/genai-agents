//! What differs between native targets and WebAssembly: `Send` bounds, clocks and timers.
//!
//! Natively the futures of tools and models must be `Send`, so that a run can move between the
//! threads of a runtime; on WebAssembly (`target_family = "wasm"`) there is one thread and
//! nothing needs to be. [`MaybeSend`] and [`MaybeSync`] are `Send` and `Sync` natively and
//! nothing on WebAssembly.

use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

/// `Send` natively; no bound on WebAssembly.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSend: Send {}
#[cfg(not(target_family = "wasm"))]
impl<T: Send + ?Sized> MaybeSend for T {}

/// `Send` natively; no bound on WebAssembly.
#[cfg(target_family = "wasm")]
pub trait MaybeSend {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSend for T {}

/// `Sync` natively; no bound on WebAssembly.
#[cfg(not(target_family = "wasm"))]
pub trait MaybeSync: Sync {}
#[cfg(not(target_family = "wasm"))]
impl<T: Sync + ?Sized> MaybeSync for T {}

/// `Sync` natively; no bound on WebAssembly.
#[cfg(target_family = "wasm")]
pub trait MaybeSync {}
#[cfg(target_family = "wasm")]
impl<T: ?Sized> MaybeSync for T {}

/// A boxed future: `Send` natively, not on WebAssembly.
#[cfg(not(target_family = "wasm"))]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;
/// A boxed future: `Send` natively, not on WebAssembly.
#[cfg(target_family = "wasm")]
pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

/// A boxed stream: `Send` natively, not on WebAssembly.
#[cfg(not(target_family = "wasm"))]
pub type BoxStream<'a, T> = Pin<Box<dyn futures::Stream<Item = T> + Send + 'a>>;
/// A boxed stream: `Send` natively, not on WebAssembly.
#[cfg(target_family = "wasm")]
pub type BoxStream<'a, T> = Pin<Box<dyn futures::Stream<Item = T> + 'a>>;

/// A monotonic clock: `std::time::Instant` natively, `web_time::Instant` on WebAssembly
/// (`std::time::Instant::now` panics on `wasm32-unknown-unknown`).
#[cfg(not(target_family = "wasm"))]
pub(crate) use std::time::Instant;
#[cfg(target_family = "wasm")]
pub(crate) use web_time::Instant;

#[cfg(not(target_family = "wasm"))]
pub(crate) use std::time::{SystemTime, UNIX_EPOCH};
#[cfg(target_family = "wasm")]
pub(crate) use web_time::{SystemTime, UNIX_EPOCH};

/// Waits for `duration`: tokio's timer natively (a tokio runtime with the time driver must be
/// running), the host's `setTimeout` on WebAssembly.
#[cfg(not(target_family = "wasm"))]
pub(crate) async fn sleep(duration: Duration) {
    tokio::time::sleep(duration).await;
}

/// Waits for `duration`: tokio's timer natively (a tokio runtime with the time driver must be
/// running), the host's `setTimeout` on WebAssembly.
#[cfg(target_family = "wasm")]
pub(crate) async fn sleep(duration: Duration) {
    let deadline = Instant::now() + duration;
    loop {
        let left = deadline.saturating_duration_since(Instant::now());
        if left.is_zero() {
            return;
        }
        // Whole milliseconds, rounded up, at most 2^31 - 1: a longer delay is waited in steps.
        #[allow(clippy::cast_possible_truncation)]
        let millis = left.as_nanos().div_ceil(1_000_000).min(i32::MAX as u128) as u32;
        gloo_timers::future::TimeoutFuture::new(millis).await;
    }
}

/// A number that differs between calls and runs: for cache keys and jitter, not for security.
pub(crate) fn random_u64() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_nanos());
    #[allow(clippy::cast_possible_truncation)]
    let mut x = (nanos as u64) ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    // splitmix64
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// A jitter in `[-1, 1]`.
pub(crate) fn random_jitter() -> f64 {
    #[allow(clippy::cast_precision_loss)]
    let unit = (random_u64() >> 11) as f64 / (1_u64 << 53) as f64;
    unit.mul_add(2.0, -1.0)
}
