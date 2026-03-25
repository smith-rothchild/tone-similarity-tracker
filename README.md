# Tone Tracker

Tracks the similarity between a spoken vowel performance and a target tone. Optionally outputs a plot. Originally developed as a final project for LING 575: Speech Processing and Low-Resource Languages, and expanded from there.

## Motivation

Tones are one of the more difficult aspects of learning a language. Unlike vocabulary or grammar, tonal accuracy is hard to self-assess. This tool attempts to fill that gap by extracting the pitch contour from a recording and comparing it directly against a target tone, giving the learner a visual and numerical indication of how close their production was.

The target tone is specified using Chao tone numerals, a standard linguistic notation in which 1 represents the lowest pitch and 5 the highest. Mandarin's third tone, for example, is commonly written as `214`. The system is language-agnostic and can be applied to any tonal language whose tones can be described in the Chao system.

## How It Works

A short baseline recording of the speaker's natural speech calibrates their individual pitch range, mapping their 5th and 95th percentile F0 values to Chao levels 1 and 5. This normalization means the tool works across speakers without manual tuning.

The performance recording is preprocessed with a pre-emphasis filter and a noise gate to reduce the effect of room echo, then passed to Praat for pitch extraction. The F0 contour is trimmed to the voiced region, interpolated over any gaps, and mapped onto the speaker's Chao scale. A target contour is generated from the Chao string using a shape-preserving spline. The two contours are compared using Dynamic Time Warping, which accommodates natural timing variation in the performance.

## Requirements

Python 3.8 or higher.
```bash
pip install -r requirements.txt
```

## Usage
```bash
python tone_similarity.py baseline.wav performance.wav <chao_tone> [output.png]
```

**Arguments**

- `baseline.wav` — a short recording of the speaker talking naturally, used to calibrate their pitch range. A few seconds of normal speech in the target language is sufficient. Must be a WAV file.
- `performance.wav` — the vowel recording to be evaluated. Must be a WAV file.
- `chao_tone` — the target tone as a Chao numeral string, e.g. `214`, `35`, `51`, `55`. Any string of digits 1-5 is valid.
- `output.png` — optional. Path to save the plot. Defaults to `result.png`.

**Example**
```bash
python tone_similarity.py baseline.wav e.wav 214 output.png
```

Chao tone values for common languages can be found in most descriptive grammars or on the relevant Wikipedia phonoly pages/sections. Standard Mandarin values: Tone 1 = `55`, Tone 2 = `35`, Tone 3 = `214`, Tone 4 = `51`.

## Output

The script prints the final score and DTW distance to the terminal and saves a plot showing the target contour (dashed) against the user's performance (solid). The score is in the range 0-100%.

## Limitations

- **Edge detection.** The tool removes unvoiced regions and applies a noise gate to suppress echo, but does not detect the precise onset of the vowel. Recordings with a long lead-in will show a flat region at the left edge of the plot.
- **Scoring.** The score is based on DTW distance passed through an exponential function with an arbitrary sensitivity parameter. It is best used to compare across multiple attempts rather than as a meaningful absolute measure.
- **Idealized contours.** The target contour is generated mathematically from the Chao string. Real tones have language-specific shapes and the same Chao classification can correspond to different acoustic realizations across dialects and speaking contexts.

## Future Work

- Improve edge detection for cleaner vowel onset alignment
- Handling of non-modal phonation such as breathy and creaky voice, which function tonally in some languages
- Automatic tone identification — inferring the intended tone without requiring explicit input
- Expand functionality to full syllables and multi-tonal sequences
- Web or mobile interface
