//! An SFZ sampler.
//!
//! [SFZ](https://sfzformat.com) describes an instrument as plain text: which
//! sample plays for which keys and velocities, and how. This crate reads
//! that text ([`Sfz`]), loads the samples it names from wherever the caller
//! keeps them ([`SampleSource`]: a directory, or bytes in memory when there
//! is no file system, as in a web page), and plays the result
//! ([`Instrument`]) with a [`Sampler`] processor or, one note per voice,
//! with [`SamplerVoice`] inside a [`Synth`](manguebyte_dsp::Synth).
//!
//! ```
//! use manguebyte_core::AudioBuffer;
//! use manguebyte_dsp::event::NoteRef;
//! use manguebyte_dsp::{ProcessSpec, Processor};
//! use manguebyte_formats::wav::{self, WavFormat};
//! use manguebyte_sampler::{Instrument, MemorySource, Sampler};
//!
//! // A sample: a second of A4 at 44.1 kHz, as a WAV file in memory.
//! let mut tone = AudioBuffer::<f32>::new(1, 44_100);
//! for (n, sample) in tone.channel_mut(0).iter_mut().enumerate() {
//!     *sample = (std::f64::consts::TAU * 440.0 * n as f64 / 44_100.0).sin() as f32;
//! }
//! let mut bytes = Vec::new();
//! wav::write(&mut bytes, &tone, 44_100, WavFormat::Float32).unwrap();
//! let mut source = MemorySource::new().with("piano/a4.wav", bytes);
//!
//! let instrument = Instrument::from_sfz(
//!     "<control> default_path=piano/
//!      <region> sample=a4.wav pitch_keycenter=69 lokey=57 hikey=81 ampeg_release=0.2",
//!     &mut source,
//! )?;
//!
//! // A sampler that starts with this instrument; its loader would send
//! // others.
//! let (mut sampler, _loader) = Sampler::<f32>::with_instrument(instrument, 32);
//! sampler.prepare(&ProcessSpec { sample_rate: 48_000.0, max_block_size: 512, num_channels: 2 });
//! sampler.note_on(NoteRef::new(0, 72), 0.8); // C5: the A4 sample, a minor third up
//! let mut block = AudioBuffer::<f32>::new(2, 512);
//! sampler.process(&mut block);
//! assert!(block.peak() > 0.1);
//! # Ok::<(), manguebyte_sampler::InstrumentError>(())
//! ```
//!
//! # Threads
//!
//! Parsing and loading read files, decode audio and allocate, so they run
//! off the audio thread. The finished [`Instrument`] is sent whole through
//! a [`SamplerLoader`]; the [`Sampler`] takes it at the start of a block,
//! and sends the one it replaces back to the loader, which frees it. The
//! audio thread never allocates, frees, locks or waits, and rendering does
//! not panic; `tests/realtime.rs` checks the allocator with every feature
//! in use.
//!
//! # Opcodes
//!
//! Headers: `<control>`, `<global>`, `<master>`, `<group>` and `<region>`,
//! each region inheriting from the global, master and group above it.
//! `<curve>`, `<effect>`, `<midi>` and `<sample>` are reported and their
//! contents ignored. Directives: `#define $NAME value` and
//! `#include "file"` (with a source). In `<control>`: `default_path`,
//! `note_offset` and `octave_offset` (both shift incoming notes).
//!
//! | Opcodes | What they do here |
//! |---|---|
//! | `sample` | The sample's path, or `*sine` (sounds the key's pitch) or `*silence`. |
//! | `lokey`, `hikey`, `key`, `lovel`, `hivel`, `lochan`, `hichan` | The notes that play the region. Keys may be numbers or names, with `c4` = 60; `hikey=-1` or `key=-1` (SFZ 2) keeps the region off the keyboard. |
//! | `pitch_keycenter`, `pitch_keytrack`, `transpose`, `tune` | Pitch: `(key - pitch_keycenter) * pitch_keytrack + 100 * transpose + tune` cents. `pitch_keycenter=sample` takes the WAV file's root key. |
//! | `bend_up`, `bend_down` | Cents at the ends of the pitch wheel. |
//! | `seq_length`, `seq_position` | Round robin. Each region counts the notes that match its keys, velocities, channels and trigger, and plays on its turn. |
//! | `lorand`, `hirand` | Each note draws one number in `0..1` from the instrument's seeded generator; a region plays when it falls in `lorand..hirand`. |
//! | `trigger` | `attack`, `release` (waits for the pedal), `release_key` (does not), `first`, `legato`. A release-triggered region plays with its note-on's velocity and never loops. |
//! | `volume`, `amplitude`, `pan`, `amp_veltrack` (also read as `ampeg_veltrack`), `rt_decay` | Gain. Velocity gives `1 - t + t * (v / 127)²` for `amp_veltrack` = `100 * t`; `rt_decay` takes decibels per second held off a release-triggered region. Mono samples pan at constant power, unity in the centre; stereo samples are balanced. |
//! | `offset`, `end` | The frames played; `end` is included, and `end=-1` silences the region (it still chokes). The interpolator hears silence outside them, so a region cut from a longer file never hears its neighbours. |
//! | `loop_mode`, `loop_start`, `loop_end`, `loop_crossfade` | `no_loop`, `one_shot`, `loop_continuous`, `loop_sustain`. Without them a WAV file's `smpl` loop is used, continuously. Loop points are read through the loop by the interpolator, so the join adds nothing; `loop_crossfade` fades the loop's last frames into those before its start, linearly in amplitude. A released `loop_sustain` leaves its loop at the first frame that reads the same with the loop as without it, so the release never clicks, even inside the crossfade. |
//! | `ampeg_delay`, `ampeg_start`, `ampeg_attack`, `ampeg_hold`, `ampeg_decay`, `ampeg_sustain`, `ampeg_release`, `ampeg_vel2*` | The amplitude envelope: a straight-line attack, a decay straight in decibels, and a release that falls 90 dB in its time. |
//! | `group`, `off_by`, `off_mode`, `off_time`, `polyphony` | Choke groups and voice limits. `polyphony` counts the voices of the region's `group` when a `group` opcode applies to it, and otherwise those of the regions under the header where it is written. |
//! | `note_polyphony` | Most voices of one note number in the region's `group`; older strikes fade. |
//! | `amp_velcurve_N` | Gain at velocity N (0 to 1), straight between the points written, with 0 at velocity 0 and 1 at 127 unless written; replaces the `amp_veltrack` curve. |
//! | `offset_random` | Up to this many frames, drawn per note, added to `offset`. |
//! | `fil_type`, `cutoff`, `resonance`, `fil_keytrack`, `fil_keycenter`, `fil_veltrack`, `fil_random` | A filter, only when `cutoff` is written: `lpf_1p`, `hpf_1p`, `lpf_2p` (default), `hpf_2p`, `bpf_2p`, `brf_2p`, `lpf_4p`, `hpf_4p`, `lpf_6p`, `hpf_6p` (the `_sv` names are read as the plain ones; other types fall back to `lpf_2p` with a warning). `resonance` is the boost at the cutoff in decibels. Topology-preserving state-variable stages, so the cutoff moves without clicks. |
//! | `fileg_*`, `fileg_depth`, `pitcheg_*`, `pitcheg_depth` | Filter and pitch envelopes, with the stages of `ampeg_*`; depth in cents. |
//! | `amplfo_*`, `pitchlfo_*`, `fillfo_*` (`freq`, `depth`, `delay`, `fade`) | SFZ 1 sine LFOs on gain (dB), pitch and cutoff (cents). |
//! | `lfoN_freq`, `lfoN_wave`, `lfoN_delay`, `lfoN_fade`, `lfoN_phase`, `lfoN_pitch`, `lfoN_volume`, `lfoN_amplitude`, `lfoN_pan`, `lfoN_cutoff` | SFZ 2 LFOs 1 to 4, each moving any of these. Waves: 0 triangle, 1 sine, 2 pulse 75 %, 3 square, 4 pulse 25 %, 5 pulse 12.5 %, 6 saw up, 7 saw down. LFOs and envelopes are worked out every 16 samples of the note. |
//! | `region_label`, `group_label`, `sw_label` and other `*_label` | Names for people: no warning, kept in [`Region::unknown`] for tools that show them. |
//!
//! Every other opcode is kept in [`Region::unknown`] and reported as a
//! [`Warning`]; values outside an opcode's range are clamped with a
//! warning; text that cannot be read is an [`SfzError`] naming its line.
//! Inherited opcodes are shared by the regions that inherit them, and a
//! parse refuses text whose regions would take more than
//! [`sfz::MAX_REGION_OPCODES`] opcodes in all, so the memory a parse holds
//! grows with the text (after `#define` replacement) and not with the
//! product of inherited opcodes and the regions that inherit them.
//!
//! Sample paths keep a leading `..`, so `../Samples/a.wav` (or
//! `default_path=../Samples/`) reads from beside the `.sfz` file's
//! directory with a [`FileSource`]; a [`MemorySource`] finds such a path
//! under the same name, `../` included.
//!
//! # Pitch
//!
//! A voice reads its sample at a position kept to 32 bits of fraction, so
//! pitch is exact to far better than a thousandth of a cent, and between
//! frames with the [`Interpolation`] chosen: a windowed sinc by default,
//! whose measured quality and cost are stated there. A sample played at its
//! own pitch and rate is read as it is.

#![forbid(unsafe_code)]

mod envelope;
mod instrument;
mod interpolate;
mod player;
mod region;
mod sampler;
pub mod sfz;
mod source;
mod voice;

pub use instrument::{
    DEFAULT_SEED, Instrument, InstrumentBuilder, InstrumentError, LoadError, MAX_SAMPLE_FRAMES,
};
pub use interpolate::Interpolation;
pub use region::{EnvelopeSpec, KeyCenter, LoopMode, OffMode, Region, Trigger};
pub use sampler::{MAX_HELD_NOTES, MAX_LOADER_CAPACITY, MAX_POLYPHONY, Sampler, SamplerLoader};
pub use sfz::{Control, Location, Opcode, Sfz, SfzError, SfzErrorKind, Warning, WarningKind};
pub use source::{FileSource, MemorySource, SampleSource, normalize_path};
pub use voice::{MAX_LAYERS, SamplerVoice};

/// Makes the tables every instrument shares, if they are not made yet: the
/// windowed-sinc kernel of [`Interpolation::Sinc`], 512 phases of 64 taps
/// and the curve they come from, about 256 KiB. They are made once per
/// program, the first time an instrument is loaded, and kept until it ends;
/// making them takes a few milliseconds. Call this where that does not
/// matter, so that no load has to: `manguebyte-web` does when an engine is
/// made, since it loads instruments on a browser's audio thread, between
/// render quanta.
pub fn prepare_tables() {
    drop(interpolate::SincTable::shared());
}
