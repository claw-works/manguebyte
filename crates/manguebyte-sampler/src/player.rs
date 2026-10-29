//! One region sounding: the part of a voice that reads a sample.

use manguebyte_core::{AudioBuffer, Sample};
use manguebyte_dsp::event::NoteRef;

use crate::envelope::Envelope;
use crate::instrument::{Instrument, SampleData, Zone};
use crate::interpolate::{Interpolation, MAX_TAPS, cubic_weights};
use crate::region::{FilterKind, OffMode};

/// Samples between updates of what moves while a note plays (LFOs, pitch and
/// filter envelopes): counted per voice from its start, so the output does
/// not depend on how the audio is cut into blocks.
const CONTROL_SAMPLES: u32 = 16;

/// A number in `0..1` from `seed`, the same every time (SplitMix64).
fn unit(seed: u64) -> f64 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
}

/// The waveform of an SFZ 2 LFO at `phase` (0 to 1), -1 to 1.
fn lfo_wave(wave: u8, phase: f64) -> f64 {
    let p = phase - phase.floor();
    match wave {
        1 => (core::f64::consts::TAU * p).sin(),
        2 => {
            if p < 0.75 {
                1.0
            } else {
                -1.0
            }
        }
        3 => {
            if p < 0.5 {
                1.0
            } else {
                -1.0
            }
        }
        4 => {
            if p < 0.25 {
                1.0
            } else {
                -1.0
            }
        }
        5 => {
            if p < 0.125 {
                1.0
            } else {
                -1.0
            }
        }
        6 => 2.0 * p - 1.0,
        7 => 1.0 - 2.0 * p,
        _ => {
            if p < 0.25 {
                4.0 * p
            } else if p < 0.75 {
                2.0 - 4.0 * p
            } else {
                4.0 * p - 4.0
            }
        }
    }
}

/// A voice's filter: one-pole stages, or two-pole state-variable stages
/// (topology-preserving, so the cutoff can move while it plays), one state
/// per channel.
#[derive(Debug, Clone, Copy)]
struct Filter {
    on: bool,
    kind: FilterKind,
    /// `tan(pi * cutoff / rate)`.
    g: f64,
    /// Damping of the last two-pole stage (1 / Q); earlier stages are
    /// Butterworth.
    k: f64,
    /// `[stage * 2 + channel] = (state 1, state 2)`.
    state: [(f64, f64); 6],
}

impl Filter {
    const OFF: Self = Self {
        on: false,
        kind: FilterKind::Lowpass2,
        g: 0.0,
        k: 1.0,
        state: [(0.0, 0.0); 6],
    };

    fn stages(&self) -> usize {
        match self.kind {
            FilterKind::Lowpass4 | FilterKind::Highpass4 => 2,
            FilterKind::Lowpass6 | FilterKind::Highpass6 => 3,
            _ => 1,
        }
    }

    fn set(&mut self, cutoff: f64, sample_rate: f64) {
        let fc = cutoff.clamp(10.0, 0.45 * sample_rate);
        self.g = (core::f64::consts::PI * fc / sample_rate).tan();
    }

    fn process(&mut self, channel: usize, x: f64) -> f64 {
        let g = self.g;
        let mut y = x;
        let stages = self.stages();
        for stage in 0..stages {
            let s = &mut self.state[stage * 2 + channel];
            if matches!(self.kind, FilterKind::Lowpass1 | FilterKind::Highpass1) {
                let v = (y - s.0) * g / (1.0 + g);
                let low = v + s.0;
                s.0 = low + v;
                y = if self.kind == FilterKind::Lowpass1 {
                    low
                } else {
                    y - low
                };
                continue;
            }
            let k = if stage + 1 == stages {
                self.k
            } else {
                core::f64::consts::SQRT_2
            };
            let a1 = 1.0 / (1.0 + g * (g + k));
            let (a2, a3) = (g * a1, g * g * a1);
            let v3 = y - s.1;
            let v1 = a1 * s.0 + a2 * v3;
            let v2 = s.1 + a2 * s.0 + a3 * v3;
            s.0 = 2.0 * v1 - s.0;
            s.1 = 2.0 * v2 - s.1;
            y = match self.kind {
                FilterKind::Highpass2 | FilterKind::Highpass4 | FilterKind::Highpass6 => {
                    y - k * v1 - v2
                }
                FilterKind::Bandpass2 => k * v1,
                FilterKind::Bandreject2 => y - k * v1,
                _ => v2,
            };
        }
        y
    }
}

/// Seconds of the fade a voice gets when it is cut short: by a choke group
/// in `off_mode=fast`, by a polyphony limit or by voice stealing. The SFZ
/// format describes `off_mode=fast` as a 6 ms fade.
pub(crate) const FAST_FADE_SECONDS: f64 = 0.006;

/// One in fixed-point positions: 32 bits of fraction.
const ONE: u64 = 1 << 32;

/// Whether the key that started a voice still holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Hold {
    /// The key is down.
    Key,
    /// The key is up and the sustain pedal holds the note.
    Pedal,
    /// Released, or never held (a release-triggered voice).
    Released,
}

/// Where the pitch wheel stands, as a voice is told.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Bend {
    /// As MIDI sends it, -8192 to 8191; the region's `bend_up` and
    /// `bend_down` turn it into cents.
    Wheel(i16),
    /// Already in cents.
    Cents(f64),
}

impl Bend {
    fn cents(self, zone: &Zone) -> f64 {
        match self {
            Self::Wheel(value) if value >= 0 => f64::from(value) / 8_191.0 * zone.bend_up,
            Self::Wheel(value) => f64::from(value) / -8_192.0 * zone.bend_down,
            Self::Cents(cents) => cents,
        }
    }
}

/// What a voice needs to know to start.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Start {
    pub(crate) zone: usize,
    pub(crate) note: NoteRef,
    /// The key that chose the region, after the instrument's note offset.
    pub(crate) key: u8,
    /// 1 to 127.
    pub(crate) velocity: u8,
    pub(crate) serial: u64,
    pub(crate) age: u64,
    pub(crate) bend: Bend,
    /// Started by the key coming up.
    pub(crate) release_trigger: bool,
    /// Decibels taken off, from `rt_decay`.
    pub(crate) attenuation_db: f64,
    pub(crate) sample_rate: f64,
}

/// A source of a sample's frames for one tap of the interpolator.
enum Tap {
    Data(usize),
    Tail(usize),
    Zero,
}

/// One region of an instrument being played.
#[derive(Debug, Clone)]
pub(crate) struct Player {
    pub(crate) active: bool,
    pub(crate) zone: usize,
    pub(crate) note: NoteRef,
    pub(crate) hold: Hold,
    /// The note event that started it, so that a choke group does not
    /// silence the layers started with it.
    pub(crate) serial: u64,
    /// Lower is older.
    pub(crate) age: u64,
    pub(crate) group: i64,
    pub(crate) off_by: Option<i64>,
    /// Fading out after being cut short.
    pub(crate) dying: bool,
    position: u64,
    step: u64,
    step_frames: f64,
    looping: bool,
    wrapped: bool,
    /// A sustain loop's key came up: the loop is left at the first frame
    /// that reads the same with it as without it.
    leave_loop: bool,
    /// The position went round the loop after the frame just read.
    just_wrapped: bool,
    envelope: Envelope,
    /// Level the fade under way started from.
    fade_from: f64,
    /// Frames the fade lasts, and frames of it left.
    fade_frames: u64,
    fade_left: u64,
    base_gain: f64,
    pan: f64,
    expression_volume: f64,
    expression_pan: f64,
    gain_left: f64,
    gain_right: f64,
    base_cents: f64,
    bend_cents: f64,
    expression_cents: f64,
    rate_ratio: f64,
    stereo: bool,
    sample_rate: f64,
    /// Note velocity, 0 to 1.
    velocity: f64,
    fileg: Envelope,
    pitcheg: Envelope,
    /// Samples since the note started, for the LFOs.
    mod_frames: u64,
    /// Samples to the next modulation update.
    control: u32,
    mod_cents: f64,
    mod_gain: f64,
    mod_pan: f64,
    /// The filter's cutoff before envelopes and LFOs, in hertz.
    cutoff: f64,
    filter: Filter,
}

impl Player {
    pub(crate) const IDLE: Self = Self {
        active: false,
        zone: 0,
        note: NoteRef::new(0, 0),
        hold: Hold::Released,
        serial: 0,
        age: 0,
        group: 0,
        off_by: None,
        dying: false,
        position: 0,
        step: ONE,
        step_frames: 1.0,
        looping: false,
        wrapped: false,
        leave_loop: false,
        just_wrapped: false,
        envelope: Envelope::IDLE,
        fade_from: 1.0,
        fade_frames: 1,
        fade_left: 0,
        base_gain: 0.0,
        pan: 0.0,
        expression_volume: 1.0,
        expression_pan: 0.0,
        gain_left: 0.0,
        gain_right: 0.0,
        base_cents: 0.0,
        bend_cents: 0.0,
        expression_cents: 0.0,
        rate_ratio: 1.0,
        stereo: false,
        sample_rate: 48_000.0,
        velocity: 0.0,
        fileg: Envelope::IDLE,
        pitcheg: Envelope::IDLE,
        mod_frames: 0,
        control: 0,
        mod_cents: 0.0,
        mod_gain: 1.0,
        mod_pan: 0.0,
        cutoff: 0.0,
        filter: Filter::OFF,
    };

    /// Starts playing a region. Returns `false`, leaving the player idle,
    /// when the region has nothing to play.
    pub(crate) fn start(&mut self, instrument: &Instrument, start: &Start) -> bool {
        *self = Self::IDLE;
        let Some(zone) = instrument.zones.get(start.zone) else {
            return false;
        };
        let Some(sample) = zone.sample.and_then(|index| instrument.samples.get(index)) else {
            return false;
        };
        let velocity = f64::from(start.velocity.clamp(1, 127)) / 127.0;
        let curve = velocity * velocity;
        let velocity_gain = match &zone.velcurve {
            Some(table) => f64::from(table[usize::from(start.velocity.clamp(1, 127))]),
            None if zone.veltrack >= 0.0 => 1.0 - zone.veltrack + zone.veltrack * curve,
            None => 1.0 + zone.veltrack * curve,
        };
        self.active = true;
        self.zone = start.zone;
        self.note = start.note;
        self.hold = if start.release_trigger {
            Hold::Released
        } else {
            Hold::Key
        };
        self.serial = start.serial;
        self.age = start.age;
        self.group = zone.group;
        self.off_by = zone.off_by;
        // A region started by a release has no key to hold its loop, so it
        // plays through once.
        self.looping = zone.lp.is_some() && !start.release_trigger;
        self.envelope = Envelope::new(&zone.envelope, velocity, start.sample_rate);
        self.base_gain =
            zone.gain * velocity_gain * 10f64.powf(-start.attenuation_db.max(0.0) / 20.0);
        self.pan = zone.pan;
        self.base_cents = (f64::from(start.key) - zone.keycenter) * zone.keytrack + zone.detune;
        self.bend_cents = start.bend.cents(zone);
        self.rate_ratio = sample.rate / start.sample_rate.max(1.0);
        self.stereo = sample.right.is_some();
        self.sample_rate = start.sample_rate.max(1.0);
        self.velocity = velocity;
        // Draws per note, the same for the same note event and region.
        let draw = |salt: u64| {
            unit(
                start.serial.wrapping_mul(0x2545_F491_4F6C_DD1D)
                    ^ (start.zone as u64).wrapping_mul(0x9E37_79B9)
                    ^ salt,
            )
        };
        let extra = if zone.offset_random > 0 {
            (draw(1) * (zone.offset_random + 1) as f64) as u64
        } else {
            0
        };
        self.position = (zone.offset + extra).min(zone.end) << 32;
        if zone.modulated {
            self.fileg = Envelope::new(&zone.fileg, velocity, self.sample_rate);
            self.pitcheg = Envelope::new(&zone.pitcheg, velocity, self.sample_rate);
            if let Some(cutoff) = zone.filter.cutoff {
                let f = &zone.filter;
                let cents = f.keytrack * (f64::from(start.key) - f64::from(f.keycenter))
                    + f.veltrack * velocity
                    + f.random * draw(2);
                self.cutoff = cutoff * 2f64.powf(cents / 1_200.0);
                // `resonance` is the boost at the cutoff, in decibels; at 0 the
                // filter is Butterworth (Q of 1/√2, no peak).
                let q = 10f64
                    .powf(f.resonance / 20.0)
                    .max(core::f64::consts::FRAC_1_SQRT_2);
                self.filter = Filter {
                    on: true,
                    kind: f.kind,
                    k: 1.0 / q,
                    ..Filter::OFF
                };
            }
            self.modulate(zone);
        }
        self.wrap(zone);
        self.update_pitch();
        self.update_gain();
        true
    }

    /// Works out the LFOs and the pitch and filter envelopes for now.
    fn modulate(&mut self, zone: &Zone) {
        let t = self.mod_frames as f64 / self.sample_rate;
        let mut cents = self.pitcheg.level() * zone.pitcheg_depth;
        let mut cutoff_cents = self.fileg.level() * zone.fileg_depth;
        let (mut db, mut amp, mut pan) = (0.0, 0.0, 0.0);
        for (slot, l) in zone.lfos.iter().enumerate() {
            if l.freq <= 0.0 || t < l.delay {
                continue;
            }
            let fade = if l.fade > 0.0 {
                ((t - l.delay) / l.fade).min(1.0)
            } else {
                1.0
            };
            // SFZ 1 LFOs (slots 0–2) are sines.
            let v = fade
                * lfo_wave(
                    if slot < 3 { 1 } else { l.wave },
                    l.phase + l.freq * (t - l.delay),
                );
            cents += v * l.pitch;
            cutoff_cents += v * l.cutoff;
            db += v * l.volume;
            amp += v * l.amplitude;
            pan += v * l.pan;
        }
        self.mod_cents = cents;
        self.mod_gain = 10f64.powf(db / 20.0) * (1.0 + amp / 100.0).max(0.0);
        self.mod_pan = pan / 100.0;
        if self.filter.on {
            self.filter.set(
                self.cutoff * 2f64.powf(cutoff_cents / 1_200.0),
                self.sample_rate,
            );
        }
    }

    /// The key came up (and the pedal, if it was down).
    pub(crate) fn release(&mut self, instrument: &Instrument) {
        self.hold = Hold::Released;
        let Some(zone) = instrument.zones.get(self.zone) else {
            return;
        };
        if zone.one_shot {
            return;
        }
        self.envelope.release();
        self.fileg.release();
        self.pitcheg.release();
        self.end_sustain(zone);
    }

    /// A sustain loop stops repeating, once the reading allows it (see
    /// [`try_leave_loop`](Self::try_leave_loop)).
    fn end_sustain(&mut self, zone: &Zone) {
        if self.looping && zone.lp.as_ref().is_some_and(|lp| lp.sustain_only) {
            self.leave_loop = true;
        }
    }

    /// Let go of for good, as by an all-notes-off: released as its key
    /// coming up would release it, pedal or not, unless that would take
    /// less than `shortest` seconds or not end it at all (a one-shot, or a
    /// region started by a release, which plays through); then it fades out
    /// over `shortest` in a straight line instead. A voice already in its
    /// release is left to it.
    pub(crate) fn let_go(&mut self, instrument: &Instrument, shortest: f64, sample_rate: f64) {
        let Some(zone) = instrument.zones.get(self.zone) else {
            self.stop();
            return;
        };
        if self.envelope.is_released() {
            return;
        }
        let plays_through = zone.one_shot || self.hold == Hold::Released;
        let shortest_samples = (shortest.max(0.0) * sample_rate).round();
        if plays_through || (self.envelope.release_samples() as f64) < shortest_samples {
            self.hold = Hold::Released;
            self.fade_out(shortest, sample_rate);
        } else {
            self.release(instrument);
        }
    }

    /// Silenced by a choke group, as the region's `off_mode` says.
    pub(crate) fn off(&mut self, instrument: &Instrument, sample_rate: f64) {
        let Some(zone) = instrument.zones.get(self.zone) else {
            self.stop();
            return;
        };
        match zone.off_mode {
            OffMode::Fast => self.fade_out(FAST_FADE_SECONDS, sample_rate),
            OffMode::Time => self.fade_out(zone.off_time, sample_rate),
            OffMode::Normal => {
                self.hold = Hold::Released;
                self.envelope.release();
                self.fileg.release();
                self.pitcheg.release();
                self.end_sustain(zone);
            }
        }
    }

    /// Fades out in a straight line over `seconds`, or faster if a fade is
    /// already under way that ends sooner.
    pub(crate) fn fade_out(&mut self, seconds: f64, sample_rate: f64) {
        let frames = (seconds.max(0.0) * sample_rate).round().clamp(1.0, 1e12) as u64;
        if !self.dying {
            self.fade_from = 1.0;
            self.fade_frames = frames;
            self.fade_left = frames;
        } else if frames < self.fade_left {
            self.fade_from = self.fade();
            self.fade_frames = frames;
            self.fade_left = frames;
        }
        self.dying = true;
    }

    /// The fade's level now: from where it started down to zero, in a
    /// straight line.
    fn fade(&self) -> f64 {
        if self.dying {
            self.fade_from * self.fade_left as f64 / self.fade_frames.max(1) as f64
        } else {
            1.0
        }
    }

    /// Stops at once.
    pub(crate) fn stop(&mut self) {
        *self = Self::IDLE;
    }

    /// The gain the voice has now, for choosing which to cut.
    pub(crate) fn loudness(&self) -> f64 {
        self.envelope.level() * self.fade() * self.base_gain
    }

    /// Whether the envelope is in its release, or the voice is fading.
    pub(crate) fn is_releasing(&self) -> bool {
        self.dying || self.envelope.is_released()
    }

    pub(crate) fn set_bend(&mut self, instrument: &Instrument, bend: Bend) {
        if let Some(zone) = instrument.zones.get(self.zone) {
            self.bend_cents = bend.cents(zone);
            self.update_pitch();
        }
    }

    /// Per-note tuning, in semitones.
    pub(crate) fn set_tuning(&mut self, semitones: f64) {
        self.expression_cents = semitones * 100.0;
        self.update_pitch();
    }

    /// Per-note gain factor.
    pub(crate) fn set_volume(&mut self, volume: f64) {
        self.expression_volume = volume.max(0.0);
        self.update_gain();
    }

    /// Per-note pan, -1 to 1, added to the region's.
    pub(crate) fn set_pan(&mut self, pan: f64) {
        self.expression_pan = pan;
        self.update_gain();
    }

    fn update_pitch(&mut self) {
        let cents = self.base_cents + self.bend_cents + self.expression_cents + self.mod_cents;
        let ratio = (2f64.powf(cents / 1_200.0) * self.rate_ratio).clamp(1.0 / 65_536.0, 256.0);
        let ratio = if ratio.is_finite() { ratio } else { 1.0 };
        self.step = (ratio * ONE as f64).round() as u64;
        self.step_frames = self.step as f64 / ONE as f64;
    }

    fn update_gain(&mut self) {
        let gain = self.base_gain * self.expression_volume * self.mod_gain;
        let pan = (self.pan + self.expression_pan + self.mod_pan).clamp(-1.0, 1.0);
        if self.stereo {
            // A stereo sample is balanced: one side is turned down.
            self.gain_left = gain * (1.0 - pan).min(1.0);
            self.gain_right = gain * (1.0 + pan).min(1.0);
        } else {
            // A mono sample is placed with a constant-power law that leaves
            // each side at unity in the centre.
            let angle = (pan + 1.0) * core::f64::consts::FRAC_PI_4;
            self.gain_left = gain * core::f64::consts::SQRT_2 * angle.cos();
            self.gain_right = gain * core::f64::consts::SQRT_2 * angle.sin();
        }
    }

    /// Brings a position past the loop's end back into it.
    fn wrap(&mut self, zone: &Zone) {
        if !self.looping {
            return;
        }
        let Some(lp) = zone.lp.as_ref() else {
            return;
        };
        let index = self.position >> 32;
        if index > lp.end {
            let inside = lp.start + (index - lp.start) % lp.len();
            self.position = (inside << 32) | (self.position & (ONE - 1));
            self.wrapped = true;
            self.just_wrapped = true;
        }
    }

    /// Leaves a released sustain loop if the taps of this read, `count`
    /// frames from `first`, all read the same frames with the loop as
    /// without it: then the switch changes nothing in the output, even in
    /// the middle of a crossfade. A loop too short for that is left on the
    /// first read after it goes round.
    fn try_leave_loop(&mut self, zone: &Zone, sample: &SampleData, first: i64, count: usize) {
        let (low, high) = self.plain_range(zone, sample);
        let last = first + count as i64 - 1;
        let fits = high - low + 1 >= count as i64;
        if zone.lp.is_none() || (first >= low && last <= high) || (!fits && self.just_wrapped) {
            self.looping = false;
            self.leave_loop = false;
        }
    }

    /// Adds the voice's sound to `buffer`. Stops the voice, on the exact
    /// frame, when its sound ends.
    pub(crate) fn render<S: Sample>(
        &mut self,
        instrument: &Instrument,
        interpolation: Interpolation,
        buffer: &mut AudioBuffer<S>,
    ) {
        if !self.active {
            return;
        }
        let frames = buffer.num_frames();
        match buffer.num_channels() {
            0 => self.run(instrument, interpolation, frames, |_, _, _| {}),
            1 => {
                let out = buffer.channel_mut(0);
                self.run(instrument, interpolation, frames, |frame, left, right| {
                    if let Some(sample) = out.get_mut(frame) {
                        *sample += S::from_f64(0.5 * (left + right));
                    }
                });
            }
            _ => {
                let (out_left, out_right) = buffer.channel_pair_mut(0, 1);
                self.run(instrument, interpolation, frames, |frame, left, right| {
                    if let Some(sample) = out_left.get_mut(frame) {
                        *sample += S::from_f64(left);
                    }
                    if let Some(sample) = out_right.get_mut(frame) {
                        *sample += S::from_f64(right);
                    }
                });
            }
        }
    }

    fn run(
        &mut self,
        instrument: &Instrument,
        interpolation: Interpolation,
        frames: usize,
        mut emit: impl FnMut(usize, f64, f64),
    ) {
        let Some(zone) = instrument.zones.get(self.zone) else {
            self.stop();
            return;
        };
        let Some(sample) = zone.sample.and_then(|index| instrument.samples.get(index)) else {
            self.stop();
            return;
        };
        let mut weights = [0.0; MAX_TAPS];
        for frame in 0..frames {
            let index = self.position >> 32;
            if !self.looping && index > zone.end {
                self.stop();
                return;
            }
            let level = self.envelope.next();
            if self.envelope.is_done() {
                self.stop();
                return;
            }
            if zone.modulated {
                self.fileg.next();
                self.pitcheg.next();
                if self.control == 0 {
                    self.modulate(zone);
                    self.update_pitch();
                    self.update_gain();
                    self.control = CONTROL_SAMPLES;
                }
                self.control -= 1;
                self.mod_frames += 1;
            }
            let mut gain = level;
            if self.dying {
                if self.fade_left == 0 {
                    self.stop();
                    return;
                }
                gain *= self.fade();
                self.fade_left -= 1;
            }
            let fraction_bits = self.position & (ONE - 1);
            let fraction = fraction_bits as f64 / ONE as f64;
            let (first, count) = if self.step == ONE && fraction_bits == 0 {
                // At its own pitch and rate, from a whole frame, the sample
                // is read as it is.
                weights[0] = 1.0;
                (index as i64, 1)
            } else {
                match interpolation {
                    Interpolation::Sinc => {
                        let half =
                            instrument
                                .sinc
                                .weights(fraction, self.step_frames, &mut weights);
                        (index as i64 + 1 - half as i64, 2 * half)
                    }
                    Interpolation::Cubic => {
                        weights[..4].copy_from_slice(&cubic_weights(fraction));
                        (index as i64 - 1, 4)
                    }
                }
            };
            if self.leave_loop {
                self.try_leave_loop(zone, sample, first, count);
            }
            self.just_wrapped = false;
            let (mut left, mut right) = self.gather(zone, sample, first, &weights[..count]);
            if self.filter.on {
                left = self.filter.process(0, left);
                right = if self.stereo {
                    self.filter.process(1, right)
                } else {
                    left
                };
            }
            emit(
                frame,
                left * gain * self.gain_left,
                right * gain * self.gain_right,
            );
            self.position = self.position.wrapping_add(self.step);
            self.wrap(zone);
        }
    }

    /// The weighted sum of the frames from `first` on, through the loop
    /// when the voice is looping.
    #[inline]
    fn gather(&self, zone: &Zone, sample: &SampleData, first: i64, weights: &[f64]) -> (f64, f64) {
        let count = weights.len();
        let (low, high) = self.plain_range(zone, sample);
        if first >= low && first + count as i64 - 1 <= high {
            let start = first as usize;
            if let Some(left) = sample.left.get(start..start + count) {
                let left_sum = dot(left, weights);
                let right_sum = match &sample.right {
                    Some(right) => right
                        .get(start..start + count)
                        .map_or(left_sum, |right| dot(right, weights)),
                    None => left_sum,
                };
                return (left_sum, right_sum);
            }
        }
        let lp = zone.lp.as_ref();
        let mut left_sum = 0.0;
        let mut right_sum = 0.0;
        for (offset, &weight) in weights.iter().enumerate() {
            let (left, right) = match self.map(zone, first + offset as i64) {
                Tap::Data(at) => {
                    let left = sample.left.get(at).copied().unwrap_or(0.0);
                    let right = match &sample.right {
                        Some(right) => right.get(at).copied().unwrap_or(0.0),
                        None => left,
                    };
                    (left, right)
                }
                Tap::Tail(at) => {
                    let Some(lp) = lp else { continue };
                    let left = lp.tail_left.get(at).copied().unwrap_or(0.0);
                    let right = if sample.right.is_some() {
                        lp.tail_right.get(at).copied().unwrap_or(0.0)
                    } else {
                        left
                    };
                    (left, right)
                }
                Tap::Zero => continue,
            };
            left_sum += weight * f64::from(left);
            right_sum += weight * f64::from(right);
        }
        (left_sum, right_sum)
    }

    /// The frames that read straight from the sample data, needing neither
    /// the loop nor the crossfade nor silence for frames outside `offset`
    /// to `end`.
    fn plain_range(&self, zone: &Zone, sample: &SampleData) -> (i64, i64) {
        let last = (sample.frames() as i64 - 1).min(zone.end as i64);
        match zone.lp.as_ref() {
            Some(lp) if self.looping => {
                let low = if self.wrapped { lp.start } else { zone.offset };
                (low as i64, lp.end as i64 - lp.crossfade() as i64)
            }
            _ => (self.first_frame(zone) as i64, last),
        }
    }

    /// The first frame a voice that is not looping hears: `offset`, or the
    /// loop's start if that is earlier and the voice has been round the
    /// loop (a sustain loop that starts before `offset` and was left on
    /// release plays on from inside it).
    fn first_frame(&self, zone: &Zone) -> u64 {
        match zone.lp.as_ref() {
            Some(lp) if self.wrapped => lp.start.min(zone.offset),
            _ => zone.offset,
        }
    }

    /// Where the tap at `frame` reads. Frames before `offset` (before the
    /// loop first goes round, see [`first_frame`](Self::first_frame)) and
    /// after `end` are silence, so that a region cut from a longer file
    /// never hears its neighbours.
    fn map(&self, zone: &Zone, frame: i64) -> Tap {
        if let (true, Some(lp)) = (self.looping, zone.lp.as_ref()) {
            let start = lp.start as i64;
            let end = lp.end as i64;
            let len = lp.len() as i64;
            let mut at = frame;
            if at > end {
                at = start + (at - start) % len;
            } else if self.wrapped && at < start {
                at = end - (start - 1 - at) % len;
            }
            let crossfade = lp.crossfade() as i64;
            if crossfade > 0 && at > end - crossfade {
                return Tap::Tail((at - (end - crossfade + 1)) as usize);
            }
            return if at >= 0 && (self.wrapped || at >= zone.offset as i64) {
                Tap::Data(at as usize)
            } else {
                Tap::Zero
            };
        }
        if frame >= self.first_frame(zone) as i64 && frame <= zone.end as i64 {
            Tap::Data(frame as usize)
        } else {
            Tap::Zero
        }
    }
}

#[inline]
fn dot(frames: &[f32], weights: &[f64]) -> f64 {
    // Four partial sums, so that the additions do not wait on one another.
    let mut sums = [0.0; 4];
    let mut frame_chunks = frames.chunks_exact(4);
    let mut weight_chunks = weights.chunks_exact(4);
    for (frame, weight) in (&mut frame_chunks).zip(&mut weight_chunks) {
        for lane in 0..4 {
            sums[lane] += f64::from(frame[lane]) * weight[lane];
        }
    }
    let mut total = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    for (&frame, &weight) in frame_chunks
        .remainder()
        .iter()
        .zip(weight_chunks.remainder())
    {
        total += f64::from(frame) * weight;
    }
    total
}
