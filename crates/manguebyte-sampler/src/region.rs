//! What one region of an SFZ instrument says, as typed values.

use std::sync::Arc;

use crate::sfz::{Location, Opcode, SfzError, SfzErrorKind, Warning, WarningKind, parse_number};

/// When a region plays (`trigger`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Trigger {
    /// When its key goes down (`attack`, the default).
    #[default]
    Attack,
    /// When its key comes up, or, if the sustain pedal holds the note, when
    /// the pedal comes up (`release`).
    Release,
    /// When its key goes down while no other key on the channel is down
    /// (`first`).
    First,
    /// When its key goes down while another key on the channel is down
    /// (`legato`).
    Legato,
    /// When its key comes up, whatever the pedal does (`release_key`).
    ReleaseKey,
}

/// How a region's sample loops (`loop_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopMode {
    /// Play from `offset` to `end` once; releasing the key starts the
    /// envelope's release (`no_loop`).
    NoLoop,
    /// Play from `offset` to `end` once, ignoring the release of the key
    /// (`one_shot`). Choke groups and voice stealing still stop it.
    OneShot,
    /// Repeat between `loop_start` and `loop_end` for as long as the voice
    /// sounds, release included (`loop_continuous`).
    Continuous,
    /// Repeat while the note is held, then play on to `end` (`loop_sustain`).
    Sustain,
}

/// How a voice stops when a choke group (`off_by`) silences it
/// (`off_mode`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OffMode {
    /// Fade out over 6 ms, as the format describes (`fast`, the default).
    #[default]
    Fast,
    /// Start the envelope's release, as if the key came up (`normal`).
    Normal,
    /// Fade out over `off_time` seconds (`time`).
    Time,
}

/// The note a sample sounds at when played unchanged (`pitch_keycenter`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyCenter {
    /// A key number.
    Key(u8),
    /// The key the sample file states (`pitch_keycenter=sample`), from a
    /// WAV file's `smpl` chunk; 60 if it states none.
    FromSample,
}

/// The amplitude envelope of a region (`ampeg_*`): delay, attack, hold,
/// decay, sustain and release. Times in seconds, levels in percent.
///
/// Each `vel2*` value is added to its stage in proportion to the note's
/// velocity: all of it at velocity 127, none at 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvelopeSpec {
    /// Seconds of silence before the attack (`ampeg_delay`).
    pub delay: f64,
    /// Level the attack starts from, in percent (`ampeg_start`).
    pub start: f64,
    /// Seconds to rise to full level, in a straight line (`ampeg_attack`).
    pub attack: f64,
    /// Seconds to stay at full level (`ampeg_hold`).
    pub hold: f64,
    /// Seconds to fall to the sustain level, in a straight line in
    /// decibels (`ampeg_decay`).
    pub decay: f64,
    /// Level held while the key is down, in percent (`ampeg_sustain`).
    pub sustain: f64,
    /// Seconds for the release to fall by 90 dB, in a straight line in
    /// decibels (`ampeg_release`).
    pub release: f64,
    /// Added to the delay at full velocity (`ampeg_vel2delay`).
    pub vel2delay: f64,
    /// Added to the attack at full velocity (`ampeg_vel2attack`).
    pub vel2attack: f64,
    /// Added to the hold at full velocity (`ampeg_vel2hold`).
    pub vel2hold: f64,
    /// Added to the decay at full velocity (`ampeg_vel2decay`).
    pub vel2decay: f64,
    /// Added to the sustain at full velocity, in percent
    /// (`ampeg_vel2sustain`).
    pub vel2sustain: f64,
    /// Added to the release at full velocity (`ampeg_vel2release`).
    pub vel2release: f64,
}

impl Default for EnvelopeSpec {
    fn default() -> Self {
        Self {
            delay: 0.0,
            start: 0.0,
            attack: 0.0,
            hold: 0.0,
            decay: 0.0,
            sustain: 100.0,
            release: 0.001,
            vel2delay: 0.0,
            vel2attack: 0.0,
            vel2hold: 0.0,
            vel2decay: 0.0,
            vel2sustain: 0.0,
            vel2release: 0.0,
        }
    }
}

impl EnvelopeSpec {
    /// The defaults of a modulation envelope (`fileg_*`, `pitcheg_*`):
    /// everything zero, sustain and release included.
    pub const MODULATION: Self = Self {
        delay: 0.0,
        start: 0.0,
        attack: 0.0,
        hold: 0.0,
        decay: 0.0,
        sustain: 0.0,
        release: 0.0,
        vel2delay: 0.0,
        vel2attack: 0.0,
        vel2hold: 0.0,
        vel2decay: 0.0,
        vel2sustain: 0.0,
        vel2release: 0.0,
    };
}

/// The filter type (`fil_type`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FilterKind {
    /// One-pole lowpass, 6 dB per octave.
    Lowpass1,
    /// One-pole highpass.
    Highpass1,
    /// Two-pole lowpass, 12 dB per octave: the format's default.
    #[default]
    Lowpass2,
    /// Two-pole highpass.
    Highpass2,
    /// Two-pole bandpass.
    Bandpass2,
    /// Two-pole band reject.
    Bandreject2,
    /// Four-pole lowpass, 24 dB per octave.
    Lowpass4,
    /// Four-pole highpass.
    Highpass4,
    /// Six-pole lowpass, 36 dB per octave.
    Lowpass6,
    /// Six-pole highpass.
    Highpass6,
}

/// The filter of a region (`fil_type`, `cutoff`, `resonance`,
/// `fil_keytrack`, `fil_keycenter`, `fil_veltrack`, `fil_random`). There
/// is no filter unless `cutoff` is written.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterSpec {
    /// `fil_type`.
    pub kind: FilterKind,
    /// Hertz; `None` means no filter.
    pub cutoff: Option<f64>,
    /// Decibels of resonance peak, 0 to 40.
    pub resonance: f64,
    /// Cents of cutoff per key away from `keycenter`.
    pub keytrack: f64,
    /// The key at which key tracking leaves the cutoff alone.
    pub keycenter: u8,
    /// Cents of cutoff at full velocity.
    pub veltrack: f64,
    /// Up to this many cents, drawn per note, added to the cutoff.
    pub random: f64,
}

impl Default for FilterSpec {
    fn default() -> Self {
        Self {
            kind: FilterKind::Lowpass2,
            cutoff: None,
            resonance: 0.0,
            keytrack: 0.0,
            keycenter: 60,
            veltrack: 0.0,
            random: 0.0,
        }
    }
}

/// A low-frequency oscillator and what it moves. The SFZ 1 LFOs
/// (`amplfo_*`, `pitchlfo_*`, `fillfo_*`) are slots 0, 1 and 2, each moving
/// one thing; the SFZ 2 ones (`lfo1_*` to `lfo4_*`) are slots 3 to 6 and
/// may move several.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LfoSpec {
    /// Hertz; at 0 the LFO does nothing.
    pub freq: f64,
    /// `lfoN_wave`: 0 triangle, 1 sine, 2 pulse 75 %, 3 square, 4 pulse
    /// 25 %, 5 pulse 12.5 %, 6 saw up, 7 saw down. SFZ 1 LFOs are sines.
    pub wave: u8,
    /// Seconds before it starts.
    pub delay: f64,
    /// Seconds it takes to reach full depth after the delay.
    pub fade: f64,
    /// Starting phase, 0 to 1.
    pub phase: f64,
    /// Depth in cents of pitch.
    pub pitch: f64,
    /// Depth in decibels of gain.
    pub volume: f64,
    /// Depth in percent of gain.
    pub amplitude: f64,
    /// Depth of pan, -100 to 100.
    pub pan: f64,
    /// Depth in cents of filter cutoff.
    pub cutoff: f64,
}

/// Slots of [`Region::lfos`].
pub const LFO_SLOTS: usize = 7;

/// One region of an instrument: a sample, the notes and velocities it
/// answers, and how it is played. The fields are the opcodes of the same
/// name, with the format's defaults.
///
/// See the [crate documentation](crate#opcodes) for what each opcode does
/// here.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Region {
    /// Where the region's header was written.
    pub location: Location,
    /// The sample's path, after `default_path`, or a generator such as
    /// `*sine`. Regions that inherit it share one copy.
    pub sample: Arc<str>,
    /// Lowest key that plays it, -1 to 127.
    pub lokey: i8,
    /// Highest key that plays it, -1 to 127. At -1 (SFZ 2's way of keeping
    /// a region off the keyboard) no key plays it.
    pub hikey: i8,
    /// The key at which the sample sounds unchanged.
    pub pitch_keycenter: KeyCenter,
    /// Lowest velocity, 0 to 127.
    pub lovel: u8,
    /// Highest velocity, 0 to 127.
    pub hivel: u8,
    /// Lowest MIDI channel, 1 to 16.
    pub lochan: u8,
    /// Highest MIDI channel, 1 to 16.
    pub hichan: u8,
    /// The region plays when the note's random number, in `0..1`, is at
    /// least this...
    pub lorand: f64,
    /// ...and less than this. At 1 every number up to 1 is taken.
    pub hirand: f64,
    /// Length of the round-robin cycle.
    pub seq_length: u32,
    /// The region's turn in the cycle, from 1.
    pub seq_position: u32,
    /// When it plays.
    pub trigger: Trigger,
    /// Gain in decibels.
    pub volume: f64,
    /// Gain in percent.
    pub amplitude: f64,
    /// Stereo position, -100 (left) to 100 (right).
    pub pan: f64,
    /// Fine tuning in cents.
    pub tune: f64,
    /// Transposition in semitones.
    pub transpose: i32,
    /// Cents of pitch per key away from the key center.
    pub pitch_keytrack: f64,
    /// How much velocity changes the gain, -100 to 100 percent
    /// (`amp_veltrack`; `ampeg_veltrack` is read as the same opcode).
    pub amp_veltrack: f64,
    /// Decibels per second a release-triggered region loses for the time
    /// its note was held.
    pub rt_decay: f64,
    /// First frame played.
    pub offset: u64,
    /// Last frame played; `-1` means the region never sounds (it still
    /// silences its choke group). `None` plays to the end of the sample.
    pub end: Option<i64>,
    /// How it loops. `None` loops continuously if the sample file has a
    /// loop, and not at all otherwise.
    pub loop_mode: Option<LoopMode>,
    /// First frame of the loop. `None` takes the sample file's loop.
    pub loop_start: Option<u64>,
    /// Last frame of the loop, included. `None` takes the sample file's
    /// loop, or the end.
    pub loop_end: Option<u64>,
    /// Seconds over which the end of the loop fades into the frames before
    /// its start.
    pub loop_crossfade: f64,
    /// The amplitude envelope.
    pub ampeg: EnvelopeSpec,
    /// The choke group the region belongs to (`group`, also
    /// `polyphony_group`).
    pub group: i64,
    /// The group whose notes silence this region (`off_by`).
    pub off_by: Option<i64>,
    /// How it is silenced.
    pub off_mode: OffMode,
    /// Seconds of fade for [`OffMode::Time`].
    pub off_time: f64,
    /// Most voices that may sound at once among the regions the limit
    /// covers: those of the region's `group` when a `group` opcode applies
    /// to the region, and otherwise those under the header where
    /// `polyphony` was written (this region alone, if it was written in the
    /// region).
    pub polyphony: Option<u32>,
    /// Cents of pitch at the top of the pitch wheel.
    pub bend_up: f64,
    /// Cents of pitch at the bottom of the pitch wheel (usually negative).
    pub bend_down: f64,
    /// Gain at given velocities (`amp_velcurve_N`), 0 to 1, between which
    /// the curve is straight; when any is written it replaces the
    /// `amp_veltrack` curve.
    pub amp_velcurve: Vec<(u8, f64)>,
    /// Up to this many frames, drawn per note, added to `offset`.
    pub offset_random: u64,
    /// Most voices of one note number the region plays at once
    /// (`note_polyphony`); older ones fade.
    pub note_polyphony: Option<u32>,
    /// The filter (`fil_type`, `cutoff`, …).
    pub filter: FilterSpec,
    /// The filter envelope (`fileg_*`).
    pub fileg: EnvelopeSpec,
    /// Cents of cutoff at the envelope's peak (`fileg_depth`).
    pub fileg_depth: f64,
    /// The pitch envelope (`pitcheg_*`).
    pub pitcheg: EnvelopeSpec,
    /// Cents of pitch at the envelope's peak (`pitcheg_depth`).
    pub pitcheg_depth: f64,
    /// The LFOs: `amplfo`, `pitchlfo`, `fillfo`, then `lfo1` to `lfo4`.
    pub lfos: [LfoSpec; LFO_SLOTS],
    /// Opcodes the sampler does not play, as written. Inherited ones share
    /// their text with every region that inherits them.
    pub unknown: Vec<Opcode>,
    /// Whether a `group` (or `polyphony_group`) opcode applies.
    pub(crate) group_set: bool,
    /// The header `polyphony` was written under, numbered in the order the
    /// parser met headers.
    pub(crate) polyphony_header: u32,
}

impl Region {
    fn new(location: Location) -> Self {
        Self {
            location,
            sample: Arc::from(""),
            lokey: 0,
            hikey: 127,
            pitch_keycenter: KeyCenter::Key(60),
            lovel: 0,
            hivel: 127,
            lochan: 1,
            hichan: 16,
            lorand: 0.0,
            hirand: 1.0,
            seq_length: 1,
            seq_position: 1,
            trigger: Trigger::Attack,
            volume: 0.0,
            amplitude: 100.0,
            pan: 0.0,
            tune: 0.0,
            transpose: 0,
            pitch_keytrack: 100.0,
            amp_veltrack: 100.0,
            rt_decay: 0.0,
            offset: 0,
            end: None,
            loop_mode: None,
            loop_start: None,
            loop_end: None,
            loop_crossfade: 0.0,
            ampeg: EnvelopeSpec::default(),
            group: 0,
            off_by: None,
            off_mode: OffMode::Fast,
            off_time: 0.006,
            polyphony: None,
            bend_up: 200.0,
            bend_down: -200.0,
            amp_velcurve: Vec::new(),
            offset_random: 0,
            note_polyphony: None,
            filter: FilterSpec::default(),
            fileg: EnvelopeSpec::MODULATION,
            fileg_depth: 0.0,
            pitcheg: EnvelopeSpec::MODULATION,
            pitcheg_depth: 0.0,
            lfos: [LfoSpec::default(); LFO_SLOTS],
            unknown: Vec::new(),
            group_set: false,
            polyphony_header: 0,
        }
    }

    /// Builds a region from the opcodes that apply to it, the inherited ones
    /// first. `None` if it names no sample.
    pub(crate) fn from_opcodes<'o>(
        opcodes: impl Iterator<Item = &'o Opcode>,
        location: Location,
        warnings: &mut Vec<Warning>,
    ) -> Result<Option<Self>, SfzError> {
        let mut region = Self::new(location);
        for opcode in opcodes {
            region.apply(opcode, warnings)?;
        }
        if region.sample.is_empty() {
            return Ok(None);
        }
        Ok(Some(region))
    }

    fn apply(&mut self, opcode: &Opcode, warnings: &mut Vec<Warning>) -> Result<(), SfzError> {
        let mut value = Value { opcode, warnings };
        match &*opcode.name {
            "sample" => {
                if opcode.value.is_empty() {
                    return Err(value.invalid());
                }
                self.sample = Arc::clone(&opcode.value);
            }
            "lokey" => self.lokey = value.key_number(-1.0)? as i8,
            "hikey" => self.hikey = value.key_number(-1.0)? as i8,
            "key" => {
                let key = value.key_number(-1.0)? as i8;
                self.lokey = key;
                self.hikey = key;
                // `key=-1` takes the region off the keyboard and leaves its
                // pitch alone.
                if let Ok(center) = u8::try_from(key) {
                    self.pitch_keycenter = KeyCenter::Key(center);
                }
            }
            "pitch_keycenter" => {
                self.pitch_keycenter = if opcode.value.eq_ignore_ascii_case("sample") {
                    KeyCenter::FromSample
                } else {
                    KeyCenter::Key(value.key()?)
                };
            }
            "lovel" => self.lovel = value.number(0.0, 127.0)? as u8,
            "hivel" => self.hivel = value.number(0.0, 127.0)? as u8,
            "lochan" => self.lochan = value.number(1.0, 16.0)? as u8,
            "hichan" => self.hichan = value.number(1.0, 16.0)? as u8,
            "lorand" => self.lorand = value.number(0.0, 1.0)?,
            "hirand" => self.hirand = value.number(0.0, 1.0)?,
            "seq_length" => self.seq_length = value.number(1.0, 100.0)? as u32,
            "seq_position" => self.seq_position = value.number(1.0, 100.0)? as u32,
            "trigger" => {
                self.trigger = match &*opcode.value {
                    "attack" => Trigger::Attack,
                    "release" => Trigger::Release,
                    "first" => Trigger::First,
                    "legato" => Trigger::Legato,
                    "release_key" => Trigger::ReleaseKey,
                    _ => return Err(value.invalid()),
                };
            }
            "volume" => self.volume = value.number(-144.0, 48.0)?,
            "amplitude" => self.amplitude = value.number(0.0, 100.0)?,
            "pan" => self.pan = value.number(-100.0, 100.0)?,
            "tune" => self.tune = value.number(-9_600.0, 9_600.0)?,
            "transpose" => self.transpose = value.number(-127.0, 127.0)? as i32,
            "pitch_keytrack" => self.pitch_keytrack = value.number(-1_200.0, 1_200.0)?,
            "amp_veltrack" | "ampeg_veltrack" => {
                self.amp_veltrack = value.number(-100.0, 100.0)?;
            }
            "rt_decay" => self.rt_decay = value.number(0.0, 200.0)?,
            "offset" => self.offset = value.number(0.0, 4_294_967_296.0)? as u64,
            "end" => self.end = Some(value.number(-1.0, 4_294_967_296.0)? as i64),
            "loop_mode" | "loopmode" => {
                self.loop_mode = Some(match &*opcode.value {
                    "no_loop" => LoopMode::NoLoop,
                    "one_shot" => LoopMode::OneShot,
                    "loop_continuous" => LoopMode::Continuous,
                    "loop_sustain" => LoopMode::Sustain,
                    _ => return Err(value.invalid()),
                });
            }
            "loop_start" | "loopstart" => {
                self.loop_start = Some(value.number(0.0, 4_294_967_296.0)? as u64);
            }
            "loop_end" | "loopend" => {
                self.loop_end = Some(value.number(0.0, 4_294_967_296.0)? as u64);
            }
            "loop_crossfade" => self.loop_crossfade = value.number(0.0, 10.0)?,
            "ampeg_delay" | "amp_delay" => self.ampeg.delay = value.number(0.0, 100.0)?,
            "ampeg_start" | "amp_start" => self.ampeg.start = value.number(0.0, 100.0)?,
            "ampeg_attack" | "amp_attack" => self.ampeg.attack = value.number(0.0, 100.0)?,
            "ampeg_hold" | "amp_hold" => self.ampeg.hold = value.number(0.0, 100.0)?,
            "ampeg_decay" | "amp_decay" => self.ampeg.decay = value.number(0.0, 100.0)?,
            "ampeg_sustain" | "amp_sustain" => self.ampeg.sustain = value.number(0.0, 100.0)?,
            "ampeg_release" | "amp_release" => self.ampeg.release = value.number(0.0, 100.0)?,
            "ampeg_vel2delay" | "amp_vel2delay" => {
                self.ampeg.vel2delay = value.number(-100.0, 100.0)?;
            }
            "ampeg_vel2attack" | "amp_vel2attack" => {
                self.ampeg.vel2attack = value.number(-100.0, 100.0)?;
            }
            "ampeg_vel2hold" | "amp_vel2hold" => {
                self.ampeg.vel2hold = value.number(-100.0, 100.0)?;
            }
            "ampeg_vel2decay" | "amp_vel2decay" => {
                self.ampeg.vel2decay = value.number(-100.0, 100.0)?;
            }
            "ampeg_vel2sustain" | "amp_vel2sustain" => {
                self.ampeg.vel2sustain = value.number(-100.0, 100.0)?;
            }
            "ampeg_vel2release" | "amp_vel2release" => {
                self.ampeg.vel2release = value.number(-100.0, 100.0)?;
            }
            "group" | "polyphony_group" => {
                self.group = value.number(-2_147_483_648.0, 4_294_967_295.0)? as i64;
                self.group_set = true;
            }
            "off_by" => {
                self.off_by = Some(value.number(-2_147_483_648.0, 4_294_967_295.0)? as i64);
            }
            "off_mode" => {
                self.off_mode = match &*opcode.value {
                    "fast" => OffMode::Fast,
                    "normal" => OffMode::Normal,
                    "time" => OffMode::Time,
                    _ => return Err(value.invalid()),
                };
            }
            "off_time" => self.off_time = value.number(0.0, 100.0)?,
            "polyphony" => self.polyphony = Some(value.number(1.0, 256.0)? as u32),
            "bend_up" | "bendup" => self.bend_up = value.number(-9_600.0, 9_600.0)?,
            "bend_down" | "benddown" => self.bend_down = value.number(-9_600.0, 9_600.0)?,
            "offset_random" => self.offset_random = value.number(0.0, 4_294_967_296.0)? as u64,
            "note_polyphony" => self.note_polyphony = Some(value.number(1.0, 256.0)? as u32),
            "fil_type" | "filtype" => {
                self.filter.kind = match &*opcode.value {
                    "lpf_1p" => FilterKind::Lowpass1,
                    "hpf_1p" => FilterKind::Highpass1,
                    "lpf_2p" | "lpf_2p_sv" => FilterKind::Lowpass2,
                    "hpf_2p" | "hpf_2p_sv" => FilterKind::Highpass2,
                    "bpf_2p" | "bpf_2p_sv" | "bpf_1p" => FilterKind::Bandpass2,
                    "brf_2p" | "brf_2p_sv" | "brf_1p" => FilterKind::Bandreject2,
                    "lpf_4p" => FilterKind::Lowpass4,
                    "hpf_4p" => FilterKind::Highpass4,
                    "lpf_6p" => FilterKind::Lowpass6,
                    "hpf_6p" => FilterKind::Highpass6,
                    _ => {
                        // Another type (peaking, comb, ladder variants…): the
                        // nearest kept is the default lowpass.
                        warnings.push(Warning {
                            location: opcode.location.clone(),
                            kind: WarningKind::UnknownOpcode {
                                name: Arc::clone(&opcode.name),
                                value: Arc::clone(&opcode.value),
                            },
                        });
                        FilterKind::Lowpass2
                    }
                };
            }
            "cutoff" => self.filter.cutoff = Some(value.number(1.0, 100_000.0)?),
            "resonance" => self.filter.resonance = value.number(0.0, 40.0)?,
            "fil_keytrack" => self.filter.keytrack = value.number(0.0, 1_200.0)?,
            "fil_keycenter" => self.filter.keycenter = value.key()?,
            "fil_veltrack" => self.filter.veltrack = value.number(-9_600.0, 9_600.0)?,
            "fil_random" => self.filter.random = value.number(0.0, 9_600.0)?,
            "fileg_depth" => self.fileg_depth = value.number(-12_000.0, 12_000.0)?,
            "pitcheg_depth" => self.pitcheg_depth = value.number(-12_000.0, 12_000.0)?,
            name if name.starts_with("fileg_") || name.starts_with("pitcheg_") => {
                let (eg, field) = if let Some(f) = name.strip_prefix("fileg_") {
                    (&mut self.fileg, f)
                } else {
                    (&mut self.pitcheg, &name["pitcheg_".len()..])
                };
                if !set_envelope(eg, field, &mut value)? {
                    unknown(opcode, warnings, &mut self.unknown);
                }
            }
            name if name.starts_with("amp_velcurve_") => {
                match name["amp_velcurve_".len()..].parse::<u8>() {
                    Ok(v) if v <= 127 => {
                        let gain = value.number(0.0, 1.0)?;
                        self.amp_velcurve.retain(|p| p.0 != v);
                        self.amp_velcurve.push((v, gain));
                    }
                    _ => unknown(opcode, warnings, &mut self.unknown),
                }
            }
            name if lfo_opcode(name).is_some() => {
                let (slot, field) = lfo_opcode(name).unwrap_or((0, ""));
                if !set_lfo(&mut self.lfos[slot], slot, field, &mut value)? {
                    unknown(opcode, warnings, &mut self.unknown);
                }
            }
            // Names and labels for people and editors: nothing to play, so no
            // warning, but kept in `unknown` for tools that show them.
            name if name.ends_with("_label") || name == "label" => {
                self.unknown.push(opcode.clone())
            }
            _ => {
                warnings.push(Warning {
                    location: opcode.location.clone(),
                    kind: WarningKind::UnknownOpcode {
                        name: Arc::clone(&opcode.name),
                        value: Arc::clone(&opcode.value),
                    },
                });
                self.unknown.push(opcode.clone());
            }
        }
        Ok(())
    }
}

fn unknown(opcode: &Opcode, warnings: &mut Vec<Warning>, kept: &mut Vec<Opcode>) {
    warnings.push(Warning {
        location: opcode.location.clone(),
        kind: WarningKind::UnknownOpcode {
            name: Arc::clone(&opcode.name),
            value: Arc::clone(&opcode.value),
        },
    });
    kept.push(opcode.clone());
}

/// One stage of an envelope written as `<prefix>_<field>`. `false` when the
/// field is not one.
fn set_envelope(
    eg: &mut EnvelopeSpec,
    field: &str,
    value: &mut Value<'_>,
) -> Result<bool, SfzError> {
    match field {
        "delay" => eg.delay = value.number(0.0, 100.0)?,
        "start" => eg.start = value.number(0.0, 100.0)?,
        "attack" => eg.attack = value.number(0.0, 100.0)?,
        "hold" => eg.hold = value.number(0.0, 100.0)?,
        "decay" => eg.decay = value.number(0.0, 100.0)?,
        "sustain" => eg.sustain = value.number(0.0, 100.0)?,
        "release" => eg.release = value.number(0.0, 100.0)?,
        "vel2delay" => eg.vel2delay = value.number(-100.0, 100.0)?,
        "vel2attack" => eg.vel2attack = value.number(-100.0, 100.0)?,
        "vel2hold" => eg.vel2hold = value.number(-100.0, 100.0)?,
        "vel2decay" => eg.vel2decay = value.number(-100.0, 100.0)?,
        "vel2sustain" => eg.vel2sustain = value.number(-100.0, 100.0)?,
        "vel2release" => eg.vel2release = value.number(-100.0, 100.0)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// `amplfo_x` → (0, x), `pitchlfo_x` → (1, x), `fillfo_x` → (2, x),
/// `lfoN_x` → (2 + N, x) for N in 1..=4.
fn lfo_opcode(name: &str) -> Option<(usize, &str)> {
    for (prefix, slot) in [("amplfo_", 0), ("pitchlfo_", 1), ("fillfo_", 2)] {
        if let Some(f) = name.strip_prefix(prefix) {
            return Some((slot, f));
        }
    }
    let rest = name.strip_prefix("lfo")?;
    let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
    let n: usize = rest[..digits].parse().ok()?;
    let field = rest[digits..].strip_prefix('_')?;
    (1..=4).contains(&n).then_some((2 + n, field))
}

/// One field of an LFO. `false` when the field is not one this sampler plays
/// (such as one driven by a controller).
fn set_lfo(
    lfo: &mut LfoSpec,
    slot: usize,
    field: &str,
    value: &mut Value<'_>,
) -> Result<bool, SfzError> {
    match (slot, field) {
        (_, "freq") => lfo.freq = value.number(0.0, 100.0)?,
        (_, "delay") => lfo.delay = value.number(0.0, 100.0)?,
        (_, "fade") => lfo.fade = value.number(0.0, 100.0)?,
        (0, "depth") => lfo.volume = value.number(-10.0, 10.0)?,
        (1, "depth") => lfo.pitch = value.number(-1_200.0, 1_200.0)?,
        (2, "depth") => lfo.cutoff = value.number(-1_200.0, 1_200.0)?,
        (3.., "wave") => lfo.wave = value.number(0.0, 7.0)? as u8,
        (3.., "phase") => lfo.phase = value.number(0.0, 1.0)?,
        (3.., "pitch") => lfo.pitch = value.number(-9_600.0, 9_600.0)?,
        (3.., "volume") => lfo.volume = value.number(-144.0, 48.0)?,
        (3.., "amplitude") => lfo.amplitude = value.number(-100.0, 100.0)?,
        (3.., "pan") => lfo.pan = value.number(-100.0, 100.0)?,
        (3.., "cutoff" | "cutoff1") => lfo.cutoff = value.number(-9_600.0, 9_600.0)?,
        _ => return Ok(false),
    }
    Ok(true)
}

/// An opcode's value being read, with somewhere to put a warning.
struct Value<'a> {
    opcode: &'a Opcode,
    warnings: &'a mut Vec<Warning>,
}

impl Value<'_> {
    fn invalid(&self) -> SfzError {
        SfzError {
            location: self.opcode.location.clone(),
            kind: SfzErrorKind::InvalidValue {
                opcode: self.opcode.name.to_string(),
                value: self.opcode.value.to_string(),
            },
        }
    }

    /// A number in `min..=max`; outside it, the nearest end with a warning.
    /// Opcodes whose values are whole numbers are cut to them by the caller.
    fn number(&mut self, min: f64, max: f64) -> Result<f64, SfzError> {
        let parsed = parse_number(&self.opcode.value).ok_or_else(|| self.invalid())?;
        Ok(self.clamped(parsed, min, max))
    }

    fn clamped(&mut self, parsed: f64, min: f64, max: f64) -> f64 {
        let used = parsed.clamp(min, max);
        if used != parsed {
            self.warnings.push(Warning {
                location: self.opcode.location.clone(),
                kind: WarningKind::OutOfRange {
                    opcode: Arc::clone(&self.opcode.name),
                    value: Arc::clone(&self.opcode.value),
                    used,
                },
            });
        }
        used
    }

    /// A key, as a number or a note name such as `c#4` (with `c4` = 60).
    fn key(&mut self) -> Result<u8, SfzError> {
        Ok(self.key_number(0.0)? as u8)
    }

    /// A key as [`key`](Self::key) reads it, held to `min..=127`.
    fn key_number(&mut self, min: f64) -> Result<f64, SfzError> {
        let parsed = match parse_number(&self.opcode.value) {
            Some(number) => number.trunc(),
            None => parse_note_name(&self.opcode.value).ok_or_else(|| self.invalid())?,
        };
        Ok(self.clamped(parsed, min, 127.0))
    }
}

/// Reads a note name: a letter, an optional `#` or `b`, and an octave, with
/// `c4` as key 60 and `c-1` as key 0.
fn parse_note_name(text: &str) -> Option<f64> {
    let text = text.trim();
    let mut chars = text.chars();
    let class = match chars.next()?.to_ascii_lowercase() {
        'c' => 0,
        'd' => 2,
        'e' => 4,
        'f' => 5,
        'g' => 7,
        'a' => 9,
        'b' => 11,
        _ => return None,
    };
    let rest = chars.as_str();
    let (accidental, octave) = if let Some(after) = rest.strip_prefix('#') {
        (1, after)
    } else if let Some(after) = rest.strip_prefix('b') {
        (-1, after)
    } else {
        (0, rest)
    };
    let digits = octave.strip_prefix('-').unwrap_or(octave);
    if digits.is_empty() || digits.len() > 3 || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let octave: i32 = octave.parse().ok()?;
    Some(f64::from((octave + 1) * 12 + class + accidental))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Sfz;

    #[test]
    fn note_names() {
        assert_eq!(parse_note_name("c4"), Some(60.0));
        assert_eq!(parse_note_name("C#4"), Some(61.0));
        assert_eq!(parse_note_name("db4"), Some(61.0));
        assert_eq!(parse_note_name("a4"), Some(69.0));
        assert_eq!(parse_note_name("c-1"), Some(0.0));
        assert_eq!(parse_note_name("g9"), Some(127.0));
        for bad in ["", "h4", "c", "c#", "cx4", "c4.5", "c99999"] {
            assert_eq!(parse_note_name(bad), None, "{bad}");
        }
    }

    #[test]
    fn every_listed_opcode_is_read() {
        let text = "<region> sample=s.wav lokey=10 hikey=20 pitch_keycenter=15 lovel=5 hivel=6
            lochan=2 hichan=3 lorand=0.25 hirand=0.5 seq_length=4 seq_position=2
            trigger=release_key volume=-3 amplitude=50 pan=10 tune=-7 transpose=-12
            pitch_keytrack=50 ampeg_veltrack=-20 rt_decay=3 offset=100 end=900
            loop_mode=loop_sustain loop_start=200 loop_end=800 loop_crossfade=0.01
            ampeg_delay=0.1 ampeg_start=10 ampeg_attack=0.2 ampeg_hold=0.3 ampeg_decay=0.4
            ampeg_sustain=50 ampeg_release=0.6 ampeg_vel2attack=-0.1 group=7 off_by=8
            off_mode=time off_time=0.05 polyphony=3 bend_up=1200 bend_down=-100";
        let sfz = Sfz::parse(text).unwrap();
        assert!(sfz.warnings.is_empty(), "{:?}", sfz.warnings);
        let r = &sfz.regions[0];
        assert_eq!(
            (r.lokey, r.hikey, r.pitch_keycenter),
            (10, 20, KeyCenter::Key(15))
        );
        assert_eq!((r.lovel, r.hivel, r.lochan, r.hichan), (5, 6, 2, 3));
        assert_eq!(
            (r.lorand, r.hirand, r.seq_length, r.seq_position),
            (0.25, 0.5, 4, 2)
        );
        assert_eq!(r.trigger, Trigger::ReleaseKey);
        assert_eq!(
            (r.volume, r.amplitude, r.pan, r.tune),
            (-3.0, 50.0, 10.0, -7.0)
        );
        assert_eq!(
            (r.transpose, r.pitch_keytrack, r.amp_veltrack),
            (-12, 50.0, -20.0)
        );
        assert_eq!((r.rt_decay, r.offset, r.end), (3.0, 100, Some(900)));
        assert_eq!(r.loop_mode, Some(LoopMode::Sustain));
        assert_eq!(
            (r.loop_start, r.loop_end, r.loop_crossfade),
            (Some(200), Some(800), 0.01)
        );
        let e = r.ampeg;
        assert_eq!(
            (
                e.delay, e.start, e.attack, e.hold, e.decay, e.sustain, e.release
            ),
            (0.1, 10.0, 0.2, 0.3, 0.4, 50.0, 0.6)
        );
        assert_eq!(e.vel2attack, -0.1);
        assert_eq!(
            (r.group, r.off_by, r.off_mode, r.off_time),
            (7, Some(8), OffMode::Time, 0.05)
        );
        assert_eq!(
            (r.polyphony, r.bend_up, r.bend_down),
            (Some(3), 1200.0, -100.0)
        );
    }

    #[test]
    fn out_of_range_values_are_clamped_with_a_warning() {
        let sfz = Sfz::parse("<region> sample=a.wav hikey=200 pan=-300").unwrap();
        assert_eq!(sfz.regions[0].hikey, 127);
        assert_eq!(sfz.regions[0].pan, -100.0);
        assert_eq!(sfz.warnings.len(), 2);
        assert!(matches!(
            sfz.warnings[1].kind,
            WarningKind::OutOfRange { used, .. } if used == -100.0
        ));
    }

    #[test]
    fn key_sets_the_range_and_the_center_and_later_opcodes_override() {
        let sfz = Sfz::parse("<group> key=50 <region> sample=a.wav hikey=52").unwrap();
        let r = &sfz.regions[0];
        assert_eq!(
            (r.lokey, r.hikey, r.pitch_keycenter),
            (50, 52, KeyCenter::Key(50))
        );
        let sfz = Sfz::parse("<region> sample=a.wav pitch_keycenter=sample").unwrap();
        assert_eq!(sfz.regions[0].pitch_keycenter, KeyCenter::FromSample);
    }
}
