//! An SFZ instrument with its samples decoded, ready to play.

use std::collections::HashMap;
use std::fmt;
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use manguebyte_formats::FormatError;

use crate::interpolate::SincTable;
use crate::region::{KeyCenter, LoopMode, OffMode, Region, Trigger};
use crate::sfz::{Location, Sfz, SfzError, Warning};
use crate::source::{FileSource, SampleSource};

/// Longest sample, in frames. Positions are kept with 32 bits of fraction
/// in 64 bits, which leaves room for this many frames and a pitch far up.
pub const MAX_SAMPLE_FRAMES: usize = 1 << 30;

/// The seed an instrument's random numbers start from unless
/// [`Instrument::set_seed`] says otherwise.
pub const DEFAULT_SEED: u64 = 0x5EED_5A3B_1E00_0001;

/// What went wrong while loading an instrument's samples.
#[derive(Debug)]
#[non_exhaustive]
pub enum LoadError {
    /// The source could not supply a sample.
    Read {
        /// The sample's path.
        path: String,
        /// The region that asked for it.
        location: Location,
        /// What the source said.
        error: io::Error,
    },
    /// A sample's bytes are not audio this framework can decode.
    Decode {
        /// The sample's path.
        path: String,
        /// The region that asked for it.
        location: Location,
        /// What the decoder said.
        error: FormatError,
    },
    /// A sample name starting with `*` that is not `*sine` or `*silence`.
    UnknownGenerator {
        /// The name.
        path: String,
        /// The region that asked for it.
        location: Location,
    },
    /// A sample longer than [`MAX_SAMPLE_FRAMES`].
    TooLong {
        /// The sample's path.
        path: String,
        /// The region that asked for it.
        location: Location,
    },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read {
                path,
                location,
                error,
            } => write!(f, "{location}: could not read sample \"{path}\": {error}"),
            Self::Decode {
                path,
                location,
                error,
            } => write!(f, "{location}: could not decode sample \"{path}\": {error}"),
            Self::UnknownGenerator { path, location } => {
                write!(f, "{location}: unknown generator \"{path}\"")
            }
            Self::TooLong { path, location } => write!(
                f,
                "{location}: sample \"{path}\" is longer than {MAX_SAMPLE_FRAMES} frames"
            ),
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Read { error, .. } => Some(error),
            Self::Decode { error, .. } => Some(error),
            _ => None,
        }
    }
}

/// Anything that can stop SFZ text from becoming an [`Instrument`].
#[derive(Debug)]
pub enum InstrumentError {
    /// The text is not valid SFZ.
    Sfz(SfzError),
    /// A sample could not be loaded.
    Load(LoadError),
}

impl fmt::Display for InstrumentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sfz(error) => error.fmt(f),
            Self::Load(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for InstrumentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Sfz(error) => Some(error),
            Self::Load(error) => Some(error),
        }
    }
}

impl From<SfzError> for InstrumentError {
    fn from(error: SfzError) -> Self {
        Self::Sfz(error)
    }
}

impl From<LoadError> for InstrumentError {
    fn from(error: LoadError) -> Self {
        Self::Load(error)
    }
}

/// A decoded sample: one or two channels.
pub(crate) struct SampleData {
    pub(crate) left: Vec<f32>,
    pub(crate) right: Option<Vec<f32>>,
    pub(crate) rate: f64,
    /// The first loop a WAV file's `smpl` chunk gives, both ends included.
    file_loop: Option<(u64, u64)>,
    /// The key a WAV file's `smpl` chunk says the sample sounds at.
    root_key: Option<u8>,
}

impl SampleData {
    pub(crate) fn frames(&self) -> usize {
        self.left.len()
    }
}

/// A loop of a zone, with the frames its crossfade replaces.
pub(crate) struct Loop {
    pub(crate) start: u64,
    /// Last frame of the loop, included.
    pub(crate) end: u64,
    pub(crate) sustain_only: bool,
    /// The last `tail_left.len()` frames of the loop, crossfaded into the
    /// frames before its start.
    pub(crate) tail_left: Vec<f32>,
    pub(crate) tail_right: Vec<f32>,
}

impl Loop {
    pub(crate) fn len(&self) -> u64 {
        self.end - self.start + 1
    }

    pub(crate) fn crossfade(&self) -> u64 {
        self.tail_left.len() as u64
    }
}

/// A region turned into numbers, with its sample found.
pub(crate) struct Zone {
    /// `None` when the region never sounds: `end=-1`, an empty sample, or
    /// an offset past its end.
    pub(crate) sample: Option<usize>,
    pub(crate) lokey: u8,
    pub(crate) hikey: u8,
    pub(crate) lovel: u8,
    pub(crate) hivel: u8,
    /// Channels counted from 0.
    pub(crate) lochan: u8,
    pub(crate) hichan: u8,
    pub(crate) lorand: f64,
    pub(crate) hirand: f64,
    pub(crate) seq_length: u32,
    pub(crate) seq_position: u32,
    pub(crate) trigger: Trigger,
    /// Linear gain from `volume` and `amplitude`.
    pub(crate) gain: f64,
    /// -1 to 1.
    pub(crate) pan: f64,
    /// -1 to 1.
    pub(crate) veltrack: f64,
    pub(crate) rt_decay: f64,
    pub(crate) keycenter: f64,
    pub(crate) keytrack: f64,
    /// `transpose` and `tune`, in cents.
    pub(crate) detune: f64,
    pub(crate) offset: u64,
    /// Last frame played, included.
    pub(crate) end: u64,
    pub(crate) one_shot: bool,
    pub(crate) lp: Option<Loop>,
    pub(crate) envelope: crate::region::EnvelopeSpec,
    pub(crate) group: i64,
    pub(crate) off_by: Option<i64>,
    pub(crate) off_mode: OffMode,
    pub(crate) off_time: f64,
    pub(crate) polyphony: Option<u32>,
    /// The voices the polyphony limit counts.
    pub(crate) polyphony_scope: PolyphonyScope,
    pub(crate) bend_up: f64,
    pub(crate) bend_down: f64,
    /// Gain by velocity, from `amp_velcurve_N`; `None` uses `veltrack`.
    pub(crate) velcurve: Option<Arc<[f32; 128]>>,
    pub(crate) offset_random: u64,
    pub(crate) note_polyphony: Option<u32>,
    pub(crate) filter: crate::region::FilterSpec,
    pub(crate) fileg: crate::region::EnvelopeSpec,
    pub(crate) fileg_depth: f64,
    pub(crate) pitcheg: crate::region::EnvelopeSpec,
    pub(crate) pitcheg_depth: f64,
    pub(crate) lfos: [crate::region::LfoSpec; crate::region::LFO_SLOTS],
    /// Whether anything moves while the note plays (an LFO, a pitch or
    /// filter envelope, or a filter at all), so that the player works it out.
    pub(crate) modulated: bool,
}

/// `amp_velcurve_N` points → gain for every velocity: straight lines between
/// the points, with velocity 0 at 0 and 127 at 1 unless written.
fn velcurve(points: &[(u8, f64)]) -> Option<Arc<[f32; 128]>> {
    if points.is_empty() {
        return None;
    }
    let mut p: Vec<(u8, f64)> = points.to_vec();
    if !p.iter().any(|x| x.0 == 0) {
        p.push((0, 0.0));
    }
    if !p.iter().any(|x| x.0 == 127) {
        p.push((127, 1.0));
    }
    p.sort_by_key(|x| x.0);
    let mut table = [0.0f32; 128];
    for w in p.windows(2) {
        let ((a, ga), (b, gb)) = (w[0], w[1]);
        for v in a..=b {
            let t = if b > a {
                f64::from(v - a) / f64::from(b - a)
            } else {
                0.0
            };
            table[usize::from(v)] = (ga + (gb - ga) * t) as f32;
        }
    }
    Some(Arc::new(table))
}

/// The voices a region's `polyphony` limit counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PolyphonyScope {
    /// Those of the regions with this `group` value.
    Group(i64),
    /// Those of the regions under the header with this number.
    Header(u32),
}

impl Zone {
    pub(crate) fn accepts(&self, channel: u8, key: u8, velocity: u8) -> bool {
        (self.lokey..=self.hikey).contains(&key)
            && (self.lovel..=self.hivel).contains(&velocity)
            && (self.lochan..=self.hichan).contains(&channel)
    }

    pub(crate) fn accepts_random(&self, random: f64) -> bool {
        random >= self.lorand && (random < self.hirand || self.hirand >= 1.0)
    }

    /// The sums of the squares of the first `frames` frames a note reads
    /// from `sample`, its sample, on the left and on the right (the same
    /// for a mono sample): from the offset, round the loop if it has one,
    /// to the end. Reads each frame twice at most, however often the loop
    /// goes round.
    fn squares(&self, sample: &SampleData, frames: u64) -> (f64, f64) {
        let sums = |from: u64, count: u64| -> (f64, f64) {
            let range = |channel: &[f32]| -> f64 {
                let (Ok(start), Ok(count)) = (usize::try_from(from), usize::try_from(count)) else {
                    return 0.0;
                };
                channel
                    .get(start..)
                    .unwrap_or_default()
                    .iter()
                    .take(count)
                    .map(|&sample| f64::from(sample) * f64::from(sample))
                    .sum()
            };
            let left = range(&sample.left);
            let right = sample.right.as_deref().map_or(left, range);
            (left, right)
        };
        let Some(lp) = &self.lp else {
            let count = (self.end + 1).saturating_sub(self.offset).min(frames);
            return sums(self.offset, count);
        };
        // Where `Player::start` puts the first frame read: an offset past
        // the loop goes round it.
        let first = if self.offset > lp.end {
            lp.start + (self.offset - lp.start) % lp.len()
        } else {
            self.offset
        };
        let before = (lp.end + 1 - first).min(frames);
        let (mut left, mut right) = sums(first, before);
        let rest = frames - before;
        let rounds = rest / lp.len();
        let part = rest % lp.len();
        if rounds > 0 {
            let (whole_left, whole_right) = sums(lp.start, lp.len());
            left += whole_left * rounds as f64;
            right += whole_right * rounds as f64;
        }
        let (part_left, part_right) = sums(lp.start, part);
        (left + part_left, right + part_right)
    }
}

/// An SFZ instrument with every sample decoded and every region resolved,
/// ready to be handed to a [`Sampler`](crate::Sampler) or
/// [`SamplerVoice`](crate::SamplerVoice).
///
/// Building one reads and decodes files and allocates, so it is done off
/// the audio thread. The instrument does not depend on the output sample
/// rate: pitch is worked out per note from each sample's own rate.
///
/// The instrument also holds the state the format makes it keep between
/// notes: the round-robin counter of each region and the random number
/// generator. Both are atomics, so an instrument shared by several voices
/// keeps one sequence; the audio thread only ever loads and stores them,
/// which never waits.
///
/// Cloning is cheap: the clone shares the decoded samples and the resolved
/// regions, which never change once loaded, and copies only the state kept
/// between notes, from which it then goes its own way. So one instrument
/// can play in a [`Sampler`](crate::Sampler) while a clone of it is played
/// elsewhere, measured say, without either one's notes moving the other's
/// round robins; the samples are freed with the last of them. Like every
/// drop of an instrument, dropping a clone may free, so it belongs off the
/// audio thread.
///
/// ```
/// use manguebyte_sampler::{Instrument, MemorySource};
///
/// let instrument = Instrument::from_sfz(
///     "<region> sample=*sine lokey=0 hikey=127",
///     &mut MemorySource::new(),
/// )?;
/// assert_eq!(instrument.num_regions(), 1);
/// assert!(instrument.warnings().is_empty());
/// let clone = instrument.clone();
/// assert_eq!(clone.sample_bytes(), instrument.sample_bytes());
/// # Ok::<(), manguebyte_sampler::InstrumentError>(())
/// ```
pub struct Instrument {
    pub(crate) zones: Arc<[Zone]>,
    pub(crate) samples: Arc<[SampleData]>,
    /// Per key, the zones that start on a key going down.
    pub(crate) attack_by_key: Arc<[Vec<u32>]>,
    /// Per key, the zones that start on a key coming up.
    pub(crate) release_by_key: Arc<[Vec<u32>]>,
    pub(crate) sequence: Vec<AtomicU32>,
    random: AtomicU64,
    seed: u64,
    pub(crate) note_shift: i32,
    /// Shared by every instrument: see [`prepare_tables`](crate::prepare_tables).
    pub(crate) sinc: Arc<SincTable>,
    warnings: Arc<[Warning]>,
}

impl Clone for Instrument {
    /// The same instrument, its samples and regions shared, its round
    /// robins and random numbers where this one's are, and from then on its
    /// own. Allocates (the round-robin counters).
    fn clone(&self) -> Self {
        Self {
            zones: Arc::clone(&self.zones),
            samples: Arc::clone(&self.samples),
            attack_by_key: Arc::clone(&self.attack_by_key),
            release_by_key: Arc::clone(&self.release_by_key),
            sequence: self
                .sequence
                .iter()
                .map(|counter| AtomicU32::new(counter.load(Ordering::Relaxed)))
                .collect(),
            random: AtomicU64::new(self.random.load(Ordering::Relaxed)),
            seed: self.seed,
            note_shift: self.note_shift,
            sinc: Arc::clone(&self.sinc),
            warnings: Arc::clone(&self.warnings),
        }
    }
}

impl fmt::Debug for Instrument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Instrument")
            .field("regions", &self.zones.len())
            .field("samples", &self.samples.len())
            .field("seed", &self.seed)
            .field("warnings", &self.warnings.len())
            .finish_non_exhaustive()
    }
}

impl Instrument {
    /// Parses `text`, reading `#include`d files and samples from `source`,
    /// and loads the result.
    pub fn from_sfz(text: &str, source: &mut dyn SampleSource) -> Result<Self, InstrumentError> {
        let sfz = Sfz::parse_with(text, source)?;
        Ok(Self::load(&sfz, source)?)
    }

    /// Reads the `.sfz` file at `path` and loads it, with samples and
    /// includes relative to the file's directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, InstrumentError> {
        let path = path.as_ref();
        let text = std::fs::read(path).map_err(|error| {
            InstrumentError::Load(LoadError::Read {
                path: path.display().to_string(),
                location: Location {
                    file: None,
                    line: 0,
                },
                error,
            })
        })?;
        let directory = path.parent().unwrap_or(Path::new(""));
        let mut source = FileSource::new(directory);
        Self::from_sfz(&String::from_utf8_lossy(&text), &mut source)
    }

    /// Loads the samples of a parsed instrument from `source`. Each
    /// different path is read and decoded once.
    ///
    /// Samples may be WAV, AIFF or FLAC, of any rate, with one or two
    /// channels (a file with more plays its first two). A WAV file's
    /// `smpl` chunk supplies its first loop and its root key, used when the
    /// SFZ text does not give them. Two generators need no file: `*sine`,
    /// one cycle of a sine that sounds at the pitch of the key played, and
    /// `*silence`.
    pub fn load(sfz: &Sfz, source: &mut dyn SampleSource) -> Result<Self, LoadError> {
        let mut samples: Vec<SampleData> = Vec::new();
        let mut by_path: HashMap<Arc<str>, Option<usize>> = HashMap::new();
        for region in &sfz.regions {
            if !by_path.contains_key(&*region.sample) {
                let loaded = load_sample(&region.sample, &region.location, source)?;
                let index = loaded.map(|data| {
                    samples.push(data);
                    samples.len() - 1
                });
                by_path.insert(Arc::clone(&region.sample), index);
            }
        }
        Ok(Self::assemble(sfz, &by_path, samples))
    }

    /// Resolves every region of `sfz` against its sample, whose place in
    /// `samples` `by_path` gives (`None` for one with no frames).
    fn assemble(
        sfz: &Sfz,
        by_path: &HashMap<Arc<str>, Option<usize>>,
        samples: Vec<SampleData>,
    ) -> Self {
        let zones: Vec<Zone> = sfz
            .regions
            .iter()
            .map(|region| {
                let index = by_path.get(&*region.sample).copied().flatten();
                let generated = region.sample.starts_with('*');
                zone(region, index.map(|i| (i, &samples[i])), generated)
            })
            .collect();
        let mut attack_by_key = vec![Vec::new(); 128];
        let mut release_by_key = vec![Vec::new(); 128];
        for (index, zone) in zones.iter().enumerate() {
            let lists = match zone.trigger {
                Trigger::Attack | Trigger::First | Trigger::Legato => &mut attack_by_key,
                Trigger::Release | Trigger::ReleaseKey => &mut release_by_key,
            };
            for key in zone.lokey..=zone.hikey.min(127) {
                lists[usize::from(key)].push(index as u32);
            }
        }
        let sequence = zones.iter().map(|_| AtomicU32::new(0)).collect();
        Self {
            zones: zones.into(),
            samples: samples.into(),
            attack_by_key: attack_by_key.into(),
            release_by_key: release_by_key.into(),
            sequence,
            random: AtomicU64::new(DEFAULT_SEED),
            seed: DEFAULT_SEED,
            note_shift: sfz.control.note_shift(),
            sinc: SincTable::shared(),
            warnings: sfz.warnings.as_slice().into(),
        }
    }

    /// Whether a note on `key` starts a region when its key goes down: a
    /// region with an `attack`, `first` or `legato` trigger whose keys,
    /// after `note_offset` and `octave_offset`, take it in, whatever its
    /// velocities, channels, random range and round robin. A region that
    /// never sounds (`*silence`, `end=-1`) counts, since it is there on
    /// purpose. Real-time safe.
    ///
    /// ```
    /// use manguebyte_sampler::{Instrument, MemorySource};
    ///
    /// let instrument = Instrument::from_sfz(
    ///     "<region> sample=*sine key=36
    ///      <region> sample=*silence key=38 trigger=release",
    ///     &mut MemorySource::new(),
    /// )?;
    /// assert!(instrument.covers_key(36));
    /// assert!(!instrument.covers_key(38), "only when it comes up");
    /// assert!(!instrument.covers_key(88));
    /// # Ok::<(), manguebyte_sampler::InstrumentError>(())
    /// ```
    pub fn covers_key(&self, key: u8) -> bool {
        let key = i32::from(key.min(127)) + self.note_shift;
        usize::try_from(key)
            .ok()
            .and_then(|key| self.attack_by_key.get(key))
            .is_some_and(|zones| !zones.is_empty())
    }

    /// About how much sound a note plays: the energy of what a note on
    /// `key` at `velocity` (0 to 1, as [`Sampler::note_on`](crate::Sampler::note_on)
    /// takes it), its key held, reads from its samples in its first
    /// `seconds`, at the gains it plays them with, both channels added, in
    /// full-scale seconds: a sine at full scale in both channels for a second
    /// is 1, so is 0 dBFS of it on one of them for two.
    ///
    /// Every region the note starts on channel 0 counts: layers add up, and
    /// regions that take turns (`seq_length`) or share the random range
    /// (`lorand`, `hirand`) count for the share of the notes they play.
    /// Each is read as the note reads it: from its `offset`, at its pitch
    /// for the key, round its loop, to its `end`, at its `volume`,
    /// `amplitude`, velocity tracking and pan. Its envelope and a loop's
    /// crossfade are left out, as is everything after the first `seconds`,
    /// so this is an estimate, for a level to start from before the
    /// instrument is played and measured. It reads each frame of a region
    /// twice at most, however often a loop goes round, so it costs about a
    /// pass over what the note plays in `seconds`, and does not allocate.
    /// A velocity or a length of time that is not a positive number, or an
    /// infinite length, gives 0.
    ///
    /// ```
    /// use manguebyte_sampler::{Instrument, MemorySource};
    ///
    /// let instrument = Instrument::from_sfz("<region> sample=*sine", &mut MemorySource::new())?;
    /// // A full-scale sine in both channels, middle C, for a second.
    /// assert!((instrument.note_energy(60, 1.0, 1.0) - 1.0).abs() < 1e-3);
    /// // At half the velocity, the square law: a quarter of the gain.
    /// let half = instrument.note_energy(60, 0.5, 1.0);
    /// assert!((half - (64.0f64 / 127.0).powi(4)).abs() < 1e-3);
    /// # Ok::<(), manguebyte_sampler::InstrumentError>(())
    /// ```
    pub fn note_energy(&self, key: u8, velocity: f32, seconds: f64) -> f64 {
        if velocity.is_nan() || velocity <= 0.0 || !seconds.is_finite() || seconds <= 0.0 {
            return 0.0;
        }
        let velocity = (f64::from(velocity) * 127.0).round().clamp(1.0, 127.0) as u8;
        let curve = (f64::from(velocity) / 127.0).powi(2);
        let Some(key) = u8::try_from(i32::from(key.min(127)) + self.note_shift)
            .ok()
            .filter(|&key| key < 128)
        else {
            return 0.0;
        };
        let Some(zones) = self.attack_by_key.get(usize::from(key)) else {
            return 0.0;
        };
        let mut energy = 0.0;
        for &index in zones {
            let Some(zone) = self.zones.get(index as usize) else {
                continue;
            };
            // A legato region waits for another key to be down.
            if zone.trigger == Trigger::Legato || !zone.accepts(0, key, velocity) {
                continue;
            }
            let Some(sample) = zone.sample.and_then(|index| self.samples.get(index)) else {
                continue;
            };
            let low = zone.lorand.clamp(0.0, 1.0);
            let high = if zone.hirand >= 1.0 {
                1.0
            } else {
                zone.hirand.clamp(0.0, 1.0)
            };
            let share = (high - low).max(0.0) / f64::from(zone.seq_length.max(1));
            // The gain, velocity tracking and pan law of `Player`.
            let velocity_gain = if zone.veltrack >= 0.0 {
                1.0 - zone.veltrack + zone.veltrack * curve
            } else {
                1.0 + zone.veltrack * curve
            };
            let gain = zone.gain * velocity_gain;
            let pan = zone.pan.clamp(-1.0, 1.0);
            let (left_gain, right_gain) = if sample.right.is_some() {
                ((1.0 - pan).min(1.0), (1.0 + pan).min(1.0))
            } else {
                let angle = (pan + 1.0) * core::f64::consts::FRAC_PI_4;
                (
                    core::f64::consts::SQRT_2 * angle.cos(),
                    core::f64::consts::SQRT_2 * angle.sin(),
                )
            };
            let cents = (f64::from(key) - zone.keycenter) * zone.keytrack + zone.detune;
            let speed = 2f64.powf(cents / 1_200.0).clamp(1.0 / 65_536.0, 256.0);
            // Frames of the sample a second of the note reads.
            let rate = sample.rate * speed;
            let frames = (seconds * rate).round();
            if !rate.is_finite() || rate <= 0.0 || frames.is_nan() || frames < 1.0 {
                continue;
            }
            let (left, right) = zone.squares(sample, frames as u64);
            energy += share
                * gain
                * gain
                * (left_gain * left_gain * left + right_gain * right_gain * right)
                / rate;
        }
        energy
    }

    /// How many regions it has.
    pub fn num_regions(&self) -> usize {
        self.zones.len()
    }

    /// How many different samples it loaded.
    pub fn num_samples(&self) -> usize {
        self.samples.len()
    }

    /// Bytes of decoded audio it holds.
    pub fn sample_bytes(&self) -> usize {
        self.samples
            .iter()
            .map(|s| (s.left.len() + s.right.as_ref().map_or(0, Vec::len)) * 4)
            .sum()
    }

    /// What the parser had to say about the text.
    pub fn warnings(&self) -> &[Warning] {
        &self.warnings
    }

    /// Sets the seed of the random numbers that choose between regions with
    /// `lorand` and `hirand`, and starts the sequence again from it. The
    /// same seed and the same notes give the same choices.
    pub fn set_seed(&mut self, seed: u64) {
        self.seed = seed;
        self.random.store(seed, Ordering::Relaxed);
    }

    /// The seed set last.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Starts every round robin and the random sequence again, as when the
    /// instrument was loaded. Real-time safe.
    pub fn restart(&self) {
        for counter in &self.sequence {
            counter.store(0, Ordering::Relaxed);
        }
        self.random.store(self.seed, Ordering::Relaxed);
    }

    /// The next random number, in `0..1`. Real-time safe.
    pub(crate) fn next_random(&self) -> f64 {
        // SplitMix64 (G. Steele, D. Lea, C. Flood, "Fast splittable
        // pseudorandom number generators", OOPSLA 2014).
        let state = self
            .random
            .load(Ordering::Relaxed)
            .wrapping_add(0x9E37_79B9_7F4A_7C15);
        self.random.store(state, Ordering::Relaxed);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Advances the round robin of zone `index` and says whether this is
    /// its turn. Real-time safe.
    pub(crate) fn take_turn(&self, index: usize) -> bool {
        let (Some(zone), Some(counter)) = (self.zones.get(index), self.sequence.get(index)) else {
            return false;
        };
        let count = counter.load(Ordering::Relaxed);
        counter.store(
            count.wrapping_add(1) % zone.seq_length.max(1),
            Ordering::Relaxed,
        );
        count % zone.seq_length.max(1) == zone.seq_position.saturating_sub(1)
    }
}

/// An [`Instrument`] loaded one sample at a time, for a thread that may
/// only work in short steps, such as a browser's audio thread between two
/// render quanta. [`Instrument::load`] does the same work in one go and
/// gives the same instrument.
///
/// [`paths`](Self::paths) lists the files the instrument needs, each once;
/// [`load`](Self::load) reads and decodes one of them; and
/// [`finish`](Self::finish) loads whatever is still missing and resolves the
/// regions against their samples.
///
/// ```
/// use manguebyte_sampler::{InstrumentBuilder, MemorySource, Sfz};
/// # use manguebyte_core::AudioBuffer;
/// # use manguebyte_formats::wav::{self, WavFormat};
/// # let mut wav_bytes = Vec::new();
/// # wav::write(&mut wav_bytes, &AudioBuffer::<f32>::new(1, 64), 24_000, WavFormat::Int16).unwrap();
///
/// let sfz = Sfz::parse(
///     "<control> default_path=kit/
///      <region> sample=kick.wav key=36
///      <region> sample=kick.wav key=35
///      <region> sample=*sine key=60",
/// )?;
/// let mut builder = InstrumentBuilder::new(sfz);
/// assert_eq!(builder.paths().collect::<Vec<_>>(), ["kit/kick.wav"]);
///
/// // One step: this file, now.
/// let mut source = MemorySource::new().with("kit/kick.wav", wav_bytes);
/// assert!(builder.load("kit/kick.wav", &mut source)?);
/// assert_eq!(builder.missing().count(), 0);
///
/// let instrument = builder.finish(&mut source)?;
/// assert_eq!(instrument.num_samples(), 2, "the file and the sine");
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct InstrumentBuilder {
    sfz: Sfz,
    /// Every different sample the regions name, in the order they first
    /// name it, with the region that does.
    order: Vec<(Arc<str>, Location)>,
    /// The samples loaded so far, by name: their place in `samples`, or
    /// `None` for one with no frames.
    by_path: HashMap<Arc<str>, Option<usize>>,
    samples: Vec<SampleData>,
}

impl fmt::Debug for InstrumentBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InstrumentBuilder")
            .field("regions", &self.sfz.regions.len())
            .field("samples", &self.order.len())
            .field("loaded", &self.by_path.len())
            .finish_non_exhaustive()
    }
}

impl InstrumentBuilder {
    /// A builder for the parsed instrument `sfz`, with no sample loaded.
    pub fn new(sfz: Sfz) -> Self {
        let mut order: Vec<(Arc<str>, Location)> = Vec::new();
        let mut seen: HashMap<Arc<str>, ()> = HashMap::new();
        for region in &sfz.regions {
            if seen.insert(Arc::clone(&region.sample), ()).is_none() {
                order.push((Arc::clone(&region.sample), region.location.clone()));
            }
        }
        Self {
            sfz,
            order,
            by_path: HashMap::new(),
            samples: Vec::new(),
        }
    }

    /// The parsed instrument.
    pub fn sfz(&self) -> &Sfz {
        &self.sfz
    }

    /// The files the instrument reads, each once, in the order its regions
    /// first name them, as they are handed to a [`SampleSource`]: after
    /// `default_path`, normalised by [`normalize_path`](crate::normalize_path).
    /// Generators such as `*sine` are not files and are left out.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.order
            .iter()
            .map(|(path, _)| &**path)
            .filter(|path| !path.starts_with('*'))
    }

    /// The files of [`paths`](Self::paths) not loaded yet.
    pub fn missing(&self) -> impl Iterator<Item = &str> {
        self.paths()
            .filter(|path| !self.by_path.contains_key(*path))
    }

    /// Reads the file at `path` from `source` and decodes it, if the
    /// instrument needs it and it is not loaded yet. Returns whether it
    /// did.
    ///
    /// # Errors
    ///
    /// [`LoadError`] if the source cannot supply the file or it is not
    /// audio this framework decodes.
    pub fn load(&mut self, path: &str, source: &mut dyn SampleSource) -> Result<bool, LoadError> {
        if path.starts_with('*') || self.by_path.contains_key(path) {
            return Ok(false);
        }
        let Some(index) = self.order.iter().position(|(name, _)| &**name == path) else {
            return Ok(false);
        };
        self.load_index(index, source)?;
        Ok(true)
    }

    fn load_index(&mut self, index: usize, source: &mut dyn SampleSource) -> Result<(), LoadError> {
        let (path, location) = &self.order[index];
        let loaded = load_sample(path, location, source)?;
        let place = loaded.map(|data| {
            self.samples.push(data);
            self.samples.len() - 1
        });
        self.by_path.insert(Arc::clone(path), place);
        Ok(())
    }

    /// Loads every sample still missing (generators included) from
    /// `source`, and makes the instrument.
    ///
    /// # Errors
    ///
    /// [`LoadError`] for the first sample that cannot be loaded.
    pub fn finish(mut self, source: &mut dyn SampleSource) -> Result<Instrument, LoadError> {
        for index in 0..self.order.len() {
            if !self.by_path.contains_key(&self.order[index].0) {
                self.load_index(index, source)?;
            }
        }
        Ok(Instrument::assemble(&self.sfz, &self.by_path, self.samples))
    }
}

/// Reads and decodes one sample. `None` when it has no frames.
fn load_sample(
    path: &str,
    location: &Location,
    source: &mut dyn SampleSource,
) -> Result<Option<SampleData>, LoadError> {
    if let Some(name) = path.strip_prefix('*') {
        return match name {
            "sine" => Ok(Some(sine_cycle())),
            "silence" => Ok(None),
            _ => Err(LoadError::UnknownGenerator {
                path: path.to_owned(),
                location: location.clone(),
            }),
        };
    }
    let bytes = source.read(path).map_err(|error| LoadError::Read {
        path: path.to_owned(),
        location: location.clone(),
        error,
    })?;
    let file = manguebyte_formats::decode(&bytes).map_err(|error| LoadError::Decode {
        path: path.to_owned(),
        location: location.clone(),
        error,
    })?;
    let frames = file.buffer.num_frames();
    if frames == 0 || file.buffer.num_channels() == 0 {
        return Ok(None);
    }
    if frames > MAX_SAMPLE_FRAMES {
        return Err(LoadError::TooLong {
            path: path.to_owned(),
            location: location.clone(),
        });
    }
    let (file_loop, root_key) = wav_sample_info(&bytes);
    let left = file.buffer.channel(0).to_vec();
    let right = (file.buffer.num_channels() > 1).then(|| file.buffer.channel(1).to_vec());
    Ok(Some(SampleData {
        left,
        right,
        rate: f64::from(file.sample_rate.max(1)),
        file_loop: file_loop.filter(|&(start, end)| start <= end && end < frames as u64),
        root_key,
    }))
}

/// Frames in the cycle `*sine` plays.
const SINE_FRAMES: usize = 2_048;

/// One cycle of a sine, looped, at a rate that makes it sound middle C
/// (261.63 Hz) at its key center, which is fixed at 60.
fn sine_cycle() -> SampleData {
    let left = (0..SINE_FRAMES)
        .map(|n| (core::f64::consts::TAU * n as f64 / SINE_FRAMES as f64).sin() as f32)
        .collect();
    let middle_c = 440.0 * 2f64.powf(-9.0 / 12.0);
    SampleData {
        left,
        right: None,
        rate: middle_c * SINE_FRAMES as f64,
        file_loop: Some((0, SINE_FRAMES as u64 - 1)),
        root_key: Some(60),
    }
}

/// The first loop and the root key in a WAV file's `smpl` chunk, if it has
/// one. Anything malformed is ignored.
fn wav_sample_info(bytes: &[u8]) -> (Option<(u64, u64)>, Option<u8>) {
    // Offsets are worked out in 64 bits, where a chunk walk of a file that
    // fits in memory cannot overflow, and only then turned into indices, so
    // that a hostile chunk size cannot overflow a 32-bit `usize`.
    let word = |at: u64| -> Option<u32> {
        let start = usize::try_from(at).ok()?;
        let slice = bytes.get(start..start.checked_add(4)?)?;
        Some(u32::from_le_bytes([slice[0], slice[1], slice[2], slice[3]]))
    };
    let is_wave =
        matches!(bytes.get(..4), Some(b"RIFF" | b"RF64")) && bytes.get(8..12) == Some(b"WAVE");
    if !is_wave {
        return (None, None);
    }
    let mut position = 12u64;
    while let Some(size) = position.checked_add(4).and_then(word) {
        let id = usize::try_from(position)
            .ok()
            .and_then(|start| bytes.get(start..start.checked_add(4)?));
        let Some(body) = position.checked_add(8) else {
            break;
        };
        let at = |offset: u64| body.checked_add(offset).and_then(word);
        if id == Some(b"smpl") {
            let size = u64::from(size);
            let unity = at(12).filter(|_| size >= 36);
            let loops = at(28).filter(|_| size >= 36).unwrap_or(0);
            let root = unity
                .and_then(|key| u8::try_from(key).ok())
                .filter(|&key| key < 128);
            let first_loop = if loops > 0 && size >= 36 + 24 {
                match (at(36 + 8), at(36 + 12)) {
                    (Some(start), Some(end)) => Some((u64::from(start), u64::from(end))),
                    _ => None,
                }
            } else {
                None
            };
            return (first_loop, root);
        }
        let padded = u64::from(size) + u64::from(size & 1);
        let Some(next) = body.checked_add(padded) else {
            break;
        };
        position = next;
    }
    (None, None)
}

/// Resolves a region against its sample.
fn zone(region: &Region, sample: Option<(usize, &SampleData)>, generated: bool) -> Zone {
    let keycenter = if generated {
        60
    } else {
        match region.pitch_keycenter {
            KeyCenter::Key(key) => key,
            KeyCenter::FromSample => sample.and_then(|(_, data)| data.root_key).unwrap_or(60),
        }
    };
    let gain = 10f64.powf(region.volume / 20.0) * region.amplitude / 100.0;
    // A key of -1 is below every key: `lokey=-1` lets key 0 in, and
    // `hikey=-1` keeps every key out, which an empty range does.
    let (lokey, hikey) = match u8::try_from(region.hikey) {
        Ok(hikey) => (region.lokey.max(0) as u8, hikey),
        Err(_) => (1, 0),
    };
    let polyphony_scope = if region.group_set {
        PolyphonyScope::Group(region.group)
    } else {
        PolyphonyScope::Header(region.polyphony_header)
    };
    let mut zone = Zone {
        sample: None,
        lokey,
        hikey,
        lovel: region.lovel,
        hivel: region.hivel,
        lochan: region.lochan.saturating_sub(1),
        hichan: region.hichan.saturating_sub(1),
        lorand: region.lorand,
        hirand: region.hirand,
        seq_length: region.seq_length.max(1),
        seq_position: region.seq_position.max(1),
        trigger: region.trigger,
        gain,
        pan: region.pan / 100.0,
        veltrack: region.amp_veltrack / 100.0,
        rt_decay: region.rt_decay,
        keycenter: f64::from(keycenter),
        keytrack: region.pitch_keytrack,
        detune: f64::from(region.transpose) * 100.0 + region.tune,
        offset: region.offset,
        end: 0,
        one_shot: false,
        lp: None,
        envelope: region.ampeg,
        group: region.group,
        off_by: region.off_by,
        off_mode: region.off_mode,
        off_time: region.off_time,
        polyphony: region.polyphony,
        polyphony_scope,
        bend_up: region.bend_up,
        bend_down: region.bend_down,
        velcurve: velcurve(&region.amp_velcurve),
        offset_random: region.offset_random,
        note_polyphony: region.note_polyphony,
        filter: region.filter,
        fileg: region.fileg,
        fileg_depth: region.fileg_depth,
        pitcheg: region.pitcheg,
        pitcheg_depth: region.pitcheg_depth,
        lfos: region.lfos,
        modulated: region.filter.cutoff.is_some()
            || region.pitcheg_depth != 0.0
            || region.lfos.iter().any(|l| {
                l.freq > 0.0
                    && (l.pitch != 0.0 || l.volume != 0.0 || l.amplitude != 0.0 || l.pan != 0.0)
            }),
    };
    let Some((index, data)) = sample else {
        return zone;
    };
    let last = data.frames() as u64 - 1;
    let end = match region.end {
        Some(end) if end < 0 => return zone,
        Some(0) | None => last,
        Some(end) => (end as u64).min(last),
    };
    if region.offset > end {
        return zone;
    }
    zone.sample = Some(index);
    zone.end = end;
    let mode = region.loop_mode.unwrap_or(if data.file_loop.is_some() {
        LoopMode::Continuous
    } else {
        LoopMode::NoLoop
    });
    zone.one_shot = mode == LoopMode::OneShot;
    if matches!(mode, LoopMode::Continuous | LoopMode::Sustain) {
        let start = region
            .loop_start
            .or(data.file_loop.map(|(start, _)| start))
            .unwrap_or(0);
        let loop_end = region
            .loop_end
            .filter(|&loop_end| loop_end > 0)
            .or(data.file_loop.map(|(_, loop_end)| loop_end))
            .unwrap_or(end)
            .min(end);
        if start <= loop_end {
            let len = loop_end - start + 1;
            let wanted = (region.loop_crossfade * data.rate).round();
            let crossfade = if wanted.is_finite() && wanted >= 1.0 {
                (wanted as u64).min(len).min(start)
            } else {
                0
            };
            let tail = |channel: &[f32]| -> Vec<f32> {
                (0..crossfade)
                    .map(|n| {
                        let at = (loop_end + 1 - crossfade + n) as usize;
                        let before = at - len as usize;
                        let fade = (n + 1) as f64 / (crossfade + 1) as f64;
                        let mixed = f64::from(channel[at]) * (1.0 - fade)
                            + f64::from(channel[before]) * fade;
                        mixed as f32
                    })
                    .collect()
            };
            zone.lp = Some(Loop {
                start,
                end: loop_end,
                sustain_only: mode == LoopMode::Sustain,
                tail_left: tail(&data.left),
                tail_right: data.right.as_deref().map(tail).unwrap_or_default(),
            });
        }
    }
    zone
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemorySource;

    /// A WAV file of `frames` frames with a `smpl` chunk.
    fn wav_with_smpl(frames: usize, unity: u32, start: u32, end: u32) -> Vec<u8> {
        let buffer = manguebyte_core::AudioBuffer::<f32>::new(1, frames);
        let mut bytes = Vec::new();
        manguebyte_formats::wav::write(
            &mut bytes,
            &buffer,
            44_100,
            manguebyte_formats::wav::WavFormat::Float32,
        )
        .unwrap();
        let mut chunk = Vec::new();
        for value in [0u32, 0, 22_675, unity, 0, 0, 0, 1, 0] {
            chunk.extend_from_slice(&value.to_le_bytes());
        }
        for value in [0u32, 0, start, end, 0, 0] {
            chunk.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(b"smpl");
        bytes.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&chunk);
        let riff = (bytes.len() - 8) as u32;
        bytes[4..8].copy_from_slice(&riff.to_le_bytes());
        bytes
    }

    /// A WAV file of 44 bytes of header and four frames of 16-bit audio,
    /// in a data chunk that claims 0xFFFFFFD0 bytes, so that the chunk after
    /// it would start at 0xFFFFFFFC and the offsets past that overflow 32
    /// bits. `examples/wasm_smoke.rs` builds the same file and loads it on
    /// a 32-bit target.
    fn hostile_wav() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        // PCM, one channel, 48 kHz, 96 000 bytes a second, 2-byte frames,
        // 16 bits.
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&48_000u32.to_le_bytes());
        bytes.extend_from_slice(&96_000u32.to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&0xFFFF_FFD0u32.to_le_bytes());
        for sample in [0i16, 1_000, 2_000, 1_000] {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn smpl_chunk_gives_loop_and_root() {
        let bytes = wav_with_smpl(100, 57, 10, 89);
        assert_eq!(wav_sample_info(&bytes), (Some((10, 89)), Some(57)));
        // Damaged chunks are ignored, never trusted.
        for cut in [12, 20, bytes.len() - 20] {
            assert_eq!(wav_sample_info(&bytes[..cut]).0, None, "{cut}");
        }
        assert_eq!(
            wav_sample_info(b"RIFF\xff\xff\xff\xffWAVEsmpl\xff\xff\xff\xff"),
            (None, None)
        );
        let mut source = MemorySource::new().with("a.wav", bytes);
        let sfz = Sfz::parse("<region> sample=a.wav pitch_keycenter=sample").unwrap();
        let instrument = Instrument::load(&sfz, &mut source).unwrap();
        let zone = &instrument.zones[0];
        assert_eq!(zone.keycenter, 57.0);
        let lp = zone.lp.as_ref().unwrap();
        assert_eq!((lp.start, lp.end, lp.sustain_only), (10, 89, false));
    }

    #[test]
    fn a_chunk_size_that_runs_the_walk_to_the_end_of_32_bits_is_ignored() {
        // The chunk after the data would start four bytes short of 2^32;
        // with 32-bit offsets the next ones would overflow. On a 64-bit
        // host this shows the walk ends and the file loads; the
        // `wasm_smoke` example checks it on wasm32.
        let bytes = hostile_wav();
        assert_eq!(wav_sample_info(&bytes), (None, None));
        let mut source = MemorySource::new().with("h.wav", bytes);
        let instrument = Instrument::from_sfz("<region> sample=h.wav", &mut source).unwrap();
        assert_eq!(instrument.sample_bytes(), 16);
    }

    #[test]
    fn regions_resolve_against_their_samples() {
        let mut source = MemorySource::new().with("a.wav", wav_with_smpl(100, 60, 0, 0));
        let text = "<region> sample=a.wav end=-1
                    <region> sample=a.wav end=500 loop_mode=loop_sustain loop_start=20 loop_end=900 loop_crossfade=1
                    <region> sample=a.wav offset=200
                    <region> sample=a.wav loop_mode=one_shot
                    <region> sample=*silence";
        let sfz = Sfz::parse(text).unwrap();
        let instrument = Instrument::load(&sfz, &mut source).unwrap();
        assert_eq!(instrument.num_samples(), 1);
        assert_eq!(instrument.sample_bytes(), 400);
        let zones = &instrument.zones;
        assert!(zones[0].sample.is_none());
        assert_eq!(zones[1].end, 99);
        let lp = zones[1].lp.as_ref().unwrap();
        // The loop end is held to the end, and the crossfade to the frames
        // before the loop start.
        assert_eq!(
            (lp.start, lp.end, lp.len(), lp.crossfade()),
            (20, 99, 80, 20)
        );
        assert!(zones[2].sample.is_none());
        assert!(zones[3].one_shot);
        assert!(zones[4].sample.is_none());
    }

    #[test]
    fn load_errors_name_the_region() {
        let sfz = Sfz::parse("\n<region> sample=missing.wav").unwrap();
        let error = Instrument::load(&sfz, &mut MemorySource::new()).unwrap_err();
        assert!(matches!(&error, LoadError::Read { location, .. } if location.line == 2));
        assert!(error.to_string().contains("missing.wav"));

        let mut source = MemorySource::new().with("bad.wav", b"RIFF1234WAVEjunk".to_vec());
        let sfz = Sfz::parse("<region> sample=bad.wav").unwrap();
        assert!(matches!(
            Instrument::load(&sfz, &mut source),
            Err(LoadError::Decode { .. })
        ));
        let sfz = Sfz::parse("<region> sample=*saw").unwrap();
        assert!(matches!(
            Instrument::load(&sfz, &mut source),
            Err(LoadError::UnknownGenerator { .. })
        ));
        let error = Instrument::open("no such dir/x.sfz").unwrap_err();
        assert!(matches!(
            error,
            InstrumentError::Load(LoadError::Read { .. })
        ));
    }

    #[test]
    fn random_numbers_and_round_robin_restart() {
        let mut source = MemorySource::new();
        let mut instrument = Instrument::from_sfz(
            "<region> sample=*sine seq_length=3 seq_position=2",
            &mut source,
        )
        .unwrap();
        let first: Vec<f64> = (0..4).map(|_| instrument.next_random()).collect();
        assert!(first.iter().all(|r| (0.0..1.0).contains(r)));
        let turns: Vec<bool> = (0..6).map(|_| instrument.take_turn(0)).collect();
        assert_eq!(turns, [false, true, false, false, true, false]);
        instrument.restart();
        let again: Vec<f64> = (0..4).map(|_| instrument.next_random()).collect();
        assert_eq!(first, again);
        assert!(!instrument.take_turn(0) && instrument.take_turn(0));
        instrument.set_seed(7);
        assert_eq!(instrument.seed(), 7);
        assert_ne!(instrument.next_random(), first[0]);
        assert!(!instrument.take_turn(99));
    }

    /// A WAV file of `channels` channels, each holding `samples`, at 1 kHz.
    fn plain_wav(samples: &[f32], channels: usize) -> Vec<u8> {
        let mut buffer = manguebyte_core::AudioBuffer::<f32>::new(channels, samples.len());
        for channel in 0..channels {
            buffer.channel_mut(channel).copy_from_slice(samples);
        }
        let mut bytes = Vec::new();
        manguebyte_formats::wav::write(
            &mut bytes,
            &buffer,
            1_000,
            manguebyte_formats::wav::WavFormat::Float32,
        )
        .unwrap();
        bytes
    }

    #[test]
    fn a_notes_energy_is_what_it_reads_at_its_gains() {
        // A second of 0.5 at 1 kHz: 0.25 a frame, in both channels of a mono
        // sample at the centre, 0.5 full-scale seconds in all.
        let half = vec![0.5f32; 1_000];
        let mut source = MemorySource::new()
            .with("mono.wav", plain_wav(&half, 1))
            .with("stereo.wav", plain_wav(&half, 2));
        let mut energy = |text: &str, key: u8, velocity: f32, seconds: f64| {
            Instrument::from_sfz(text, &mut source)
                .unwrap()
                .note_energy(key, velocity, seconds)
        };
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        let mono = "<region> sample=mono.wav pitch_keycenter=60";
        assert!(close(energy(mono, 60, 1.0, 10.0), 0.5), "to its end");
        assert!(close(energy(mono, 60, 1.0, 0.25), 0.125), "for a quarter");
        // An octave up reads the second in half the time.
        assert!(close(energy(mono, 72, 1.0, 10.0), 0.25));
        assert!(close(
            energy(
                "<control> note_offset=12 <region> sample=mono.wav pitch_keycenter=72",
                60,
                1.0,
                10.0
            ),
            0.5
        ));
        // Gains: volume, amplitude, velocity tracking, the pan of a stereo
        // sample, which turns the other side down.
        let quarter = energy("<region> sample=mono.wav volume=-6.0206", 60, 1.0, 10.0);
        assert!((quarter - 0.125).abs() < 1e-6, "{quarter}");
        assert!(close(
            energy("<region> sample=mono.wav amplitude=50", 60, 1.0, 10.0),
            0.125
        ));
        let soft = (64.0f64 / 127.0).powi(4) * 0.5;
        assert!(close(energy(mono, 60, 0.5, 10.0), soft));
        assert!(close(
            energy("<region> sample=mono.wav amp_veltrack=0", 60, 0.5, 10.0),
            0.5
        ));
        assert!(close(
            energy("<region> sample=stereo.wav", 60, 1.0, 10.0),
            0.5
        ));
        assert!(close(
            energy("<region> sample=stereo.wav pan=100", 60, 1.0, 10.0),
            0.25
        ));
        // A mono sample panned keeps its power.
        assert!(close(
            energy("<region> sample=mono.wav pan=-60", 60, 1.0, 10.0),
            0.5
        ));
        // Layers add up; turns and random ranges share.
        let layers = "<region> sample=mono.wav <region> sample=mono.wav";
        assert!(close(energy(layers, 60, 1.0, 10.0), 1.0));
        let turns = "<region> sample=mono.wav seq_length=2 seq_position=1
                     <region> sample=mono.wav seq_length=2 seq_position=2";
        assert!(close(energy(turns, 60, 1.0, 10.0), 0.5));
        let chance = "<region> sample=mono.wav hirand=0.25 <region> sample=mono.wav lorand=0.25";
        assert!(close(energy(chance, 60, 1.0, 10.0), 0.5));
        // What a note on its own does not start counts for nothing.
        for silent in [
            "<region> sample=mono.wav trigger=legato",
            "<region> sample=mono.wav trigger=release",
            "<region> sample=mono.wav lovel=127",
            "<region> sample=mono.wav key=61",
            "<region> sample=mono.wav end=-1",
        ] {
            assert_eq!(energy(silent, 60, 0.5, 10.0), 0.0, "{silent}");
        }
        assert_eq!(energy(mono, 60, 0.0, 10.0), 0.0);
        assert_eq!(energy(mono, 60, f32::NAN, 10.0), 0.0);
        assert_eq!(energy(mono, 60, 1.0, f64::INFINITY), 0.0);
        assert_eq!(energy(mono, 60, 1.0, -1.0), 0.0);
    }

    #[test]
    fn a_notes_energy_goes_round_its_loop() {
        // 0.5 for 500 frames, then a loop of 100 frames of 1.
        let mut frames = vec![0.5f32; 500];
        frames.extend([1.0f32; 100]);
        let mut source = MemorySource::new().with("loop.wav", plain_wav(&frames, 1));
        let looped = Instrument::from_sfz(
            "<region> sample=loop.wav loop_mode=loop_continuous loop_start=500 loop_end=599",
            &mut source,
        )
        .unwrap();
        // Half a second before the loop, then the loop for the rest: 2 a
        // second of it, both channels.
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
        assert!(close(looped.note_energy(60, 1.0, 0.5), 0.25));
        assert!(close(looped.note_energy(60, 1.0, 10.0), 0.25 + 9.5 * 2.0));
        assert!(close(looped.note_energy(60, 1.0, 0.55), 0.25 + 0.1));
        // An offset past the loop starts inside it.
        let inside = Instrument::from_sfz(
            "<region> sample=loop.wav offset=650 end=599 loop_mode=loop_sustain loop_start=500 loop_end=599",
            &mut source,
        )
        .unwrap();
        assert_eq!(
            inside.note_energy(60, 1.0, 1.0),
            0.0,
            "an offset past the end plays nothing"
        );
        let wrapped = Instrument::from_sfz(
            "<region> sample=loop.wav offset=550 loop_mode=loop_sustain loop_start=100 loop_end=199",
            &mut source,
        )
        .unwrap();
        // An offset past the loop enters it where `Player` does, at 150;
        // the loop holds 0.5, so a second of it is 0.25 a frame.
        assert!(close(wrapped.note_energy(60, 1.0, 1.0), 2.0 * 0.25 * 1.0));
    }

    #[test]
    fn a_clone_shares_the_samples_and_keeps_its_own_round_robins() {
        let mut source = MemorySource::new().with("a.wav", wav_with_smpl(100, 60, 0, 0));
        let mut instrument = Instrument::from_sfz(
            "<region> sample=a.wav seq_length=2 seq_position=1
             <region> sample=a.wav seq_length=2 seq_position=2",
            &mut source,
        )
        .unwrap();
        instrument.set_seed(11);
        assert!(instrument.take_turn(0));
        let _ = instrument.next_random();
        let clone = instrument.clone();
        // The same memory, not a copy of it.
        assert!(Arc::ptr_eq(&clone.samples, &instrument.samples));
        assert!(Arc::ptr_eq(&clone.zones, &instrument.zones));
        assert!(Arc::ptr_eq(&clone.attack_by_key, &instrument.attack_by_key));
        assert_eq!(clone.sample_bytes(), instrument.sample_bytes());
        assert_eq!((clone.num_regions(), clone.seed()), (2, 11));
        // It starts where the original was, then goes its own way: had the
        // two shared one counter, the original's call here would have been
        // its turn.
        assert_eq!(clone.next_random(), instrument.next_random());
        assert!(!clone.take_turn(0) && !instrument.take_turn(0));
        let turns: Vec<bool> = (0..4).map(|_| clone.take_turn(0)).collect();
        assert_eq!(turns, [true, false, true, false]);
        assert!(instrument.take_turn(0), "the original's turn is its own");
        // Dropped, the original lets go of the samples the clone still has.
        drop(instrument);
        assert_eq!(Arc::strong_count(&clone.samples), 1);
        assert_eq!(clone.samples[0].frames(), 100);
    }
}
