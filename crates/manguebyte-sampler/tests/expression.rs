//! Filters, LFOs, pitch and filter envelopes, velocity curves, random
//! offsets and note polyphony.

mod common;

use common::*;
use manguebyte_sampler::{MemorySource, Sfz};

fn source() -> MemorySource {
    MemorySource::new()
        .with("tone.wav", sine_wav(880.0, 48_000, 2.0, 0.5))
        .with(
            "ramp.wav",
            wav_bytes(
                &[(0..48_000).map(|i| i as f32 / 48_000.0).collect()],
                48_000,
            ),
        )
}

fn play(sfz: &str, key: u8, velocity: f32, frames: usize) -> Vec<f32> {
    let mut source = source();
    let mut sampler = sampler_with(sfz, &mut source, 8, 1);
    render_mono(&mut sampler, frames, &[(0, on(key, velocity))])
}

const TONE: &str = "<global> amp_veltrack=0 pitch_keycenter=69 loop_mode=no_loop";

fn db(x: f64) -> f64 {
    20.0 * x.max(1e-12).log10()
}

#[test]
fn lowpass_and_highpass_filters_cut_what_they_should() {
    let dry = rms(&play(&format!("{TONE} <region> sample=tone.wav"), 69, 1.0, 24_000)[4_800..]);
    let low = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=150 fil_type=lpf_2p"),
        69,
        1.0,
        24_000,
    )[4_800..]);
    let low4 = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=150 fil_type=lpf_4p"),
        69,
        1.0,
        24_000,
    )[4_800..]);
    let high = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=8000 fil_type=hpf_2p"),
        69,
        1.0,
        24_000,
    )[4_800..]);
    let open = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=18000"),
        69,
        1.0,
        24_000,
    )[4_800..]);
    // 880 Hz is 2.55 octaves above 150 Hz: about -30 dB at 12 dB/octave, -60 at 24.
    assert!(db(low / dry) < -25.0, "{}", db(low / dry));
    assert!(
        db(low4 / dry) < db(low / dry) - 20.0,
        "{} {}",
        db(low4 / dry),
        db(low / dry)
    );
    assert!(db(high / dry) < -25.0, "{}", db(high / dry));
    assert!(db(open / dry).abs() < 0.5, "{}", db(open / dry));
    // Resonance at the tone's frequency makes it louder than the dry tone.
    let peak = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=880 resonance=12"),
        69,
        1.0,
        24_000,
    )[4_800..]);
    assert!(db(peak / dry) > 9.0, "{}", db(peak / dry));
}

#[test]
fn filter_follows_keys_velocity_and_its_envelope() {
    // Key tracking: an octave up with 1200 cents/key... 100 cents per key
    // keeps the cutoff at the tone's harmonic distance.
    let low_key = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=400 fil_keytrack=0"),
        81,
        1.0,
        24_000,
    )[4_800..]);
    let tracked = rms(&play(
        &format!("{TONE} <region> sample=tone.wav cutoff=400 fil_keytrack=100 fil_keycenter=69"),
        81,
        1.0,
        24_000,
    )[4_800..]);
    assert!(tracked > 2.0 * low_key, "{tracked} {low_key}");
    // Velocity: soft notes darker.
    let sfz = format!("{TONE} <region> sample=tone.wav cutoff=300 fil_veltrack=2400");
    assert!(
        rms(&play(&sfz, 69, 1.0, 24_000)[4_800..])
            > 3.0 * rms(&play(&sfz, 69, 0.2, 24_000)[4_800..])
    );
    // Envelope: opens and closes over half a second.
    let swept = play(
        &format!(
            "{TONE} <region> sample=tone.wav cutoff=150 fileg_depth=4800 fileg_attack=0 fileg_decay=0.5 fileg_sustain=0"
        ),
        69,
        1.0,
        48_000,
    );
    assert!(rms(&swept[480..2_400]) > 4.0 * rms(&swept[36_000..48_000]));
}

#[test]
fn lfos_move_pitch_and_gain() {
    // Pitch: a 2 Hz LFO, a semitone deep — a quarter period apart the
    // frequency differs by about a semitone.
    for sfz in [
        "<region> sample=tone.wav pitchlfo_freq=2 pitchlfo_depth=100",
        "<region> sample=tone.wav lfo1_freq=2 lfo1_wave=1 lfo1_pitch=100",
    ] {
        let out = play(&format!("{TONE} {sfz}"), 69, 1.0, 48_000);
        let (a, b) = (
            frequency(&out[4_800..7_200], RATE),
            frequency(&out[16_800..19_200], RATE),
        );
        assert!(cents(a, b).abs() > 120.0, "{sfz}: {a} {b}");
    }
    // Gain: 4 Hz, 6 dB.
    let out = play(
        &format!("{TONE} <region> sample=tone.wav amplfo_freq=4 amplfo_depth=6"),
        69,
        1.0,
        24_000,
    );
    let levels: Vec<f64> = (0..8)
        .map(|k| rms(&out[2_400 + k * 1_500..2_400 + k * 1_500 + 600]))
        .collect();
    let (lo, hi) = levels
        .iter()
        .fold((f64::MAX, 0.0f64), |(a, b), &x| (a.min(x), b.max(x)));
    assert!(db(hi / lo) > 6.0, "{levels:?}");
    // Delay: nothing moves before it.
    let out = play(
        &format!("{TONE} <region> sample=tone.wav lfo1_freq=5 lfo1_amplitude=80 lfo1_delay=0.3"),
        69,
        1.0,
        14_400,
    );
    let early: Vec<f64> = (0..4)
        .map(|k| rms(&out[1_200 + k * 2_400..1_200 + k * 2_400 + 1_200]))
        .collect();
    assert!(
        early.windows(2).all(|w| (db(w[0] / w[1])).abs() < 0.5),
        "{early:?}"
    );
}

#[test]
fn pitch_envelope_bends_the_start() {
    let out = play(
        &format!(
            "{TONE} <region> sample=tone.wav pitcheg_depth=1200 pitcheg_decay=0.4 pitcheg_sustain=0"
        ),
        69,
        1.0,
        48_000,
    );
    let (start, end) = (
        frequency(&out[200..1_400], RATE),
        frequency(&out[38_400..43_200], RATE),
    );
    assert!(cents(start, end) > 700.0, "{start} {end}");
    assert!(cents(end, 880.0).abs() < 5.0, "{end}");
}

#[test]
fn velocity_curve_replaces_veltrack() {
    let flat = "<region> sample=tone.wav pitch_keycenter=69 amp_velcurve_1=1 amp_velcurve_127=1";
    let soft = rms(&play(flat, 69, 0.3, 9_600)[2_400..]);
    let hard = rms(&play(flat, 69, 1.0, 9_600)[2_400..]);
    assert!(db(soft / hard).abs() < 0.1, "{soft} {hard}");
    // A curve with only the middle point: straight lines to 0 at 0 and 1 at 127.
    let mid = "<region> sample=tone.wav pitch_keycenter=69 amp_velcurve_64=0.9";
    let at = |v: f32| rms(&play(mid, 69, v, 9_600)[2_400..]);
    assert!((at(64.0 / 127.0) / at(1.0) - 0.9).abs() < 0.02);
}

#[test]
fn offset_random_starts_each_note_somewhere_new_within_range() {
    let mut source = source();
    let mut sampler = sampler_with(
        "<region> sample=ramp.wav pitch_keycenter=60 amp_veltrack=0 offset=4800 offset_random=24000",
        &mut source,
        8,
        1,
    );
    let mut firsts = Vec::new();
    for k in 0..4 {
        let out = render_mono(&mut sampler, 4_800, &[(0, on(60, 1.0)), (100, off(60))]);
        firsts.push(out[0]);
        let _ = k;
    }
    // The ramp's value is its frame / 48000: starts between frames 4800 and 28800.
    assert!(
        firsts.iter().all(|&f| (0.099..=0.601).contains(&f)),
        "{firsts:?}"
    );
    assert!(
        firsts.windows(2).any(|w| (w[0] - w[1]).abs() > 0.01),
        "{firsts:?}"
    );
}

#[test]
fn note_polyphony_fades_the_older_strike() {
    let mut source = source();
    let sfz = "<region> sample=tone.wav pitch_keycenter=69 amp_veltrack=0 note_polyphony=1";
    let mut sampler = sampler_with(sfz, &mut source, 8, 1);
    render_mono(
        &mut sampler,
        9_600,
        &[(0, on(69, 1.0)), (2_400, on(69, 1.0))],
    );
    assert_eq!(sampler.active_voices(), 1);
    let free = "<region> sample=tone.wav pitch_keycenter=69 amp_veltrack=0";
    let mut sampler = sampler_with(free, &mut source, 8, 1);
    render_mono(
        &mut sampler,
        9_600,
        &[(0, on(69, 1.0)), (2_400, on(69, 1.0))],
    );
    assert_eq!(sampler.active_voices(), 2);
}

#[test]
fn the_new_opcodes_and_labels_parse_without_warnings() {
    let sfz = Sfz::parse(
        "<group> group_label=Sustain <region> sample=a.wav region_label=C4 cutoff=1000 resonance=6 fil_type=hpf_4p
         fil_keytrack=50 fil_keycenter=c4 fil_veltrack=1200 fil_random=100 fileg_attack=0.1 fileg_decay=0.3 fileg_sustain=20
         fileg_release=0.2 fileg_depth=2400 pitcheg_attack=0.01 pitcheg_depth=-200 amplfo_freq=3 amplfo_depth=2 pitchlfo_freq=5
         pitchlfo_depth=20 pitchlfo_delay=0.2 pitchlfo_fade=0.4 fillfo_freq=1 fillfo_depth=600 lfo1_freq=4 lfo1_wave=3
         lfo1_phase=0.25 lfo1_pitch=30 lfo1_volume=-3 lfo2_freq=0.5 lfo2_pan=40 lfo2_cutoff=1200 amp_velcurve_0=0.1
         amp_velcurve_100=0.8 offset_random=200 note_polyphony=2",
    )
    .unwrap();
    assert!(sfz.warnings.is_empty(), "{:?}", sfz.warnings);
    let r = &sfz.regions[0];
    assert_eq!(
        (
            r.filter.cutoff,
            r.filter.keycenter,
            r.fileg_depth,
            r.pitcheg_depth
        ),
        (Some(1000.0), 60, 2400.0, -200.0)
    );
    assert_eq!(
        (
            r.lfos[1].pitch,
            r.lfos[3].wave,
            r.lfos[4].pan,
            r.note_polyphony
        ),
        (20.0, 3, 40.0, Some(2))
    );
    // A controller-driven LFO field is still reported.
    assert_eq!(
        Sfz::parse("<region> sample=a.wav lfo1_pitch_oncc1=50 lfo9_freq=2")
            .unwrap()
            .warnings
            .len(),
        2
    );
}
