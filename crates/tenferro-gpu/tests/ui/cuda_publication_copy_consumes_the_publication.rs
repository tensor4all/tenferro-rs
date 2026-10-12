use tenferro_gpu::cuda::{CudaHeldSession, PinnedHostBuffer, SourcePublication};

/// The device copy consumes the publication, so no later consumption can race the copy: the
/// publication is gone and the caller cannot hand the same value to a second route.
fn use_after_copy(publication: SourcePublication, consumer: &mut CudaHeldSession<'_>) {
    let pending = publication.copy_into(consumer).expect("submitted");
    let _still_mine = publication.bytes();
    let _ = pending;
}

/// The same holds for the source half's host route: a consumed publication cannot be downloaded.
fn download_after_copy(
    publication: SourcePublication,
    consumer: &mut CudaHeldSession<'_>,
    buffer: PinnedHostBuffer,
) {
    let _pending = publication.copy_into(consumer).expect("submitted");
    let _second = publication.download_into(buffer);
}

fn main() {
    let _ = (use_after_copy, download_after_copy);
}
