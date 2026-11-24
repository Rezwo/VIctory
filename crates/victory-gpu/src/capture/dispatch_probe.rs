//! One-shot texture statistics immediately before/after a recorded dispatch.

use super::read_back;
use crate::descriptor::{View, host::HostHeap};
use crate::host::HostList;
use crate::object::This;
use crate::resource::HostTexture;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use victory_shader::ResourceType;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

struct Selector {
    frame: u64,
    hash: String,
}

impl Selector {
    fn parse(value: &str) -> Option<Self> {
        let (frame, hash) = value.trim().split_once(':')?;
        let frame: u64 = frame.parse().ok()?;
        (frame > 0 && hash.len() == 16 && hash.bytes().all(|b| b.is_ascii_hexdigit())).then(|| {
            Self {
                frame,
                hash: hash.to_ascii_lowercase(),
            }
        })
    }
}

static CLAIMED: AtomicBool = AtomicBool::new(false);
static PENDING: AtomicBool = AtomicBool::new(false);

pub(crate) fn mark_pending() {
    PENDING.store(true, Ordering::Relaxed);
}

pub(crate) fn pending() -> bool {
    PENDING.load(Ordering::Relaxed)
}

pub(crate) fn finish_reporting() {
    PENDING.store(false, Ordering::Relaxed);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Before,
    After,
}

struct Target {
    phase: Phase,
    id: usize,
    texture: Arc<HostTexture>,
}

pub(crate) struct Probe {
    frame: u64,
    hash: String,
    targets: Vec<Target>,
}

impl Probe {
    pub(crate) fn new(frame: u64, hash: &str) -> Self {
        Self {
            frame,
            hash: hash.to_owned(),
            targets: Vec::new(),
        }
    }

    pub(crate) fn selected(pipeline: &crate::pipeline::host::HostPipeline) -> Option<Self> {
        static SELECTOR: OnceLock<Option<Selector>> = OnceLock::new();
        let selected = SELECTOR
            .get_or_init(|| {
                let value = std::env::var("VICTORY_DISPATCH_PROBE").ok()?;
                let parsed = Selector::parse(&value);
                if parsed.is_none() {
                    tracing::warn!("VICTORY_DISPATCH_PROBE expects FRAME:16_DIGIT_SHADER_HASH");
                }
                parsed
            })
            .as_ref()?;
        if CLAIMED.load(Ordering::Relaxed)
            || crate::queue::presented().saturating_add(1) != selected.frame
            || !pipeline
                .stages
                .iter()
                .any(|s| s.shader.hash == selected.hash)
        {
            return None;
        }
        Some(Self::new(selected.frame, &selected.hash))
    }

    pub(crate) fn claim(&self) -> bool {
        if CLAIMED.swap(true, Ordering::Relaxed) {
            return false;
        }
        if self.targets.iter().all(|t| t.phase == Phase::After)
            || self.targets.iter().all(|t| t.phase == Phase::Before)
        {
            tracing::warn!(
                record_frame = self.frame,
                hash = %self.hash,
                eligible_views = self.targets.len(),
                "dispatch probe could not identify both a backed R8 input and output"
            );
        }
        true
    }

    pub(crate) fn observe(&mut self, kind: ResourceType, view: &View, heap: &HostHeap, slot: u32) {
        use crate::descriptor::host::placed_texture;
        let placed = placed_texture(view);
        let view = placed.as_ref().unwrap_or(view);
        let (phase, id) = match (kind, view) {
            (ResourceType::Srv, View::Shader(id, _)) => (Phase::Before, *id),
            (ResourceType::Uav, View::Unordered(id, _, _)) => (Phase::After, *id),
            _ => return,
        };
        if !crate::descriptor::host::probe_r8_view(view)
            || self.targets.iter().any(|t| t.id == id && t.phase == phase)
            || self.targets.iter().filter(|t| t.phase == phase).count() >= 2
        {
            return;
        }
        let Some(texture) = crate::resource::host_copy(id as This) else {
            return;
        };
        if !heap.probe_slot_backs(slot, &texture.texture) {
            return; // rejected/null or backed by a different texture
        }
        // SAFETY: the host copy keeps the texture alive.
        let d = unsafe { texture.texture.GetDesc() };
        if d.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || d.DepthOrArraySize != 1
            || d.MipLevels != 1
            || d.SampleDesc.Count != 1
            || !matches!(d.Format, DXGI_FORMAT_R8_UNORM | DXGI_FORMAT_R8_UINT)
            || d.Width.saturating_mul(u64::from(d.Height)) > 4 * 1024 * 1024
        {
            return;
        }
        self.targets.push(Target { phase, id, texture });
    }

    pub(crate) fn record(&self, list: &HostList, after: bool) {
        let phase = if after { Phase::After } else { Phase::Before };
        let Some(host) = crate::host::host() else {
            return;
        };
        for target in self.targets.iter().filter(|t| t.phase == phase) {
            let pair = list.with_probe_copy_source(&target.texture, || {
                read_back(&host.device, &list.list, &target.texture.texture)
            });
            let Some((buffer, footprint)) = pair.flatten() else {
                tracing::warn!(
                    record_frame = self.frame,
                    hash = %self.hash,
                    resource = format_args!("{:#x}", target.id),
                    ?phase,
                    "dispatch probe could not record a texture copy"
                );
                continue;
            };
            list.retain_dispatch_probe(Sample {
                frame: self.frame,
                hash: self.hash.clone(),
                id: target.id,
                phase,
                buffer,
                footprint,
                reported: AtomicBool::new(false),
            });
        }
    }
}

pub(crate) struct Sample {
    frame: u64,
    hash: String,
    id: usize,
    phase: Phase,
    buffer: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    reported: AtomicBool,
}

#[derive(Debug, PartialEq, Eq)]
struct Stats {
    nonzero: usize,
    max: u8,
}

fn r8_stats(bytes: &[u8], width: usize, height: usize, pitch: usize) -> Option<Stats> {
    if width == 0 || height == 0 || width > pitch {
        return None;
    }
    let mut stats = Stats { nonzero: 0, max: 0 };
    for y in 0..height {
        let start = y.checked_mul(pitch)?;
        for &pixel in bytes.get(start..start.checked_add(width)?)? {
            stats.nonzero += usize::from(pixel != 0);
            stats.max = stats.max.max(pixel);
        }
    }
    Some(stats)
}

impl Sample {
    /// Only after the submission fence has completed.
    pub(crate) fn report(&self, completed_fence: u64) {
        if self.reported.swap(true, Ordering::Relaxed) {
            return;
        }
        let f = self.footprint.Footprint;
        let Some(stats) = self.stats() else {
            tracing::warn!(
                record_frame = self.frame, hash = %self.hash, resource = format_args!("{:#x}", self.id),
                phase = ?self.phase, "dispatch probe could not map readback"
            );
            return;
        };
        tracing::info!(
            target: "victory_gpu::dispatch_probe",
            record_frame = self.frame,
            completed_fence,
            hash = %self.hash,
            phase = ?self.phase,
            resource = format_args!("{:#x}", self.id),
            width = f.Width,
            height = f.Height,
            format = f.Format.0,
            pixels = u64::from(f.Width) * u64::from(f.Height),
            nonzero = stats.nonzero,
            max_raw = stats.max,
            "dispatch texture sample"
        );
    }

    fn stats(&self) -> Option<Stats> {
        let f = self.footprint.Footprint;
        let start = usize::try_from(self.footprint.Offset).ok()?;
        let (width, height, pitch) = (f.Width as usize, f.Height as usize, f.RowPitch as usize);
        let bytes = height
            .checked_sub(1)?
            .checked_mul(pitch)?
            .checked_add(width)?;
        let end = start.checked_add(bytes)?;
        let mut mapped = std::ptr::null_mut();
        // SAFETY: the completed copy populated these footprint rows; map only
        // through the last texel (the trailing row padding need not exist).
        unsafe {
            self.buffer
                .Map(
                    0,
                    Some(&D3D12_RANGE {
                        Begin: start,
                        End: end,
                    }),
                    Some(&mut mapped),
                )
                .ok()?;
            let stats = (!mapped.is_null())
                .then(|| {
                    r8_stats(
                        std::slice::from_raw_parts(mapped.cast::<u8>().add(start), bytes),
                        width,
                        height,
                        pitch,
                    )
                })
                .flatten();
            self.buffer
                .Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            stats
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selector_requires_one_frame_and_a_shader_hash() {
        let parsed = Selector::parse("2879:0123456789ABCDEF").unwrap();
        assert_eq!(parsed.frame, 2879);
        assert_eq!(parsed.hash, "0123456789abcdef");
        for invalid in [
            "0:0123456789abcdef",
            "0123456789abcdef",
            "1:xx",
            "1:0123456789abcdeg",
        ] {
            assert!(Selector::parse(invalid).is_none());
        }
    }

    #[test]
    fn r8_statistics_ignore_padding_and_count_real_pixels() {
        let raw = [0, 7, 255, 255, 9, 0];
        assert_eq!(r8_stats(&raw, 2, 2, 4), Some(Stats { nonzero: 2, max: 9 }));
        assert_eq!(r8_stats(&raw[..5], 2, 2, 4), None);
    }
}
