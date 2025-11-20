#!/usr/bin/env python3
"""
Enhanced Waveform Analysis - Make Differences Visible

Creates multiple views to clearly distinguish waveforms:
1. Time vs X helix visualization (spatial-temporal patterns)
2. Frequency spectrum (FFT analysis, reveals periodic structure)
3. Event rate over time (reveals motion patterns)

EASY CONFIGURATION:
Edit the constants in the CONFIGURATION section below to change:
- Window start time and duration for EACH analysis (helix, frequency, event rate)
- Maximum events for each analysis type
- Frequency analysis parameters (sample rate, max frequency to display)
- Event rate bin size

Author: QTCR-Net Research Team
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib
matplotlib.use('Agg')


# ============================================================================
# CONFIGURATION - Edit these values to change analysis parameters
# ============================================================================

# Window timing parameters - HELIX (Time vs X)
HELIX_START_MS = 1000           # Start time for helix window (milliseconds)
HELIX_DURATION_MS = 500         # Duration for helix window (milliseconds)

# Window timing parameters - FREQUENCY SPECTRUM
FREQUENCY_START_MS = 1000       # Start time for frequency analysis (milliseconds)
FREQUENCY_DURATION_MS = 500     # Duration for frequency analysis (milliseconds)

# Window timing parameters - EVENT RATE
RATE_START_MS = 1000            # Start time for event rate analysis (milliseconds)
RATE_DURATION_MS = 500          # Duration for event rate analysis (milliseconds)

# Event limits (None = no limit)
MAX_EVENTS_HELIX = 100000       # Max events for Time vs X helix plot
MAX_EVENTS_FREQUENCY = 100000   # Max events for frequency spectrum analysis
MAX_EVENTS_RATE = 100000        # Max events for event rate analysis

# Frequency analysis parameters
SAMPLE_RATE_HZ = 1000           # Sampling rate for frequency analysis
MAX_FREQ_DISPLAY_HZ = 100       # Maximum frequency to display on plots

# Event rate binning
EVENT_RATE_BIN_MS = 10          # Time bin size for event rate plots (milliseconds)

# ============================================================================


def load_events_window(csv_path: str, start_ms: float, duration_ms: float, max_events: int = None):
    """Load events from time window."""
    df = pd.read_csv(csv_path, comment='#')
    df.columns = df.columns.str.strip().str.lower()
    if 'timestamp' in df.columns:
        df.rename(columns={'timestamp': 'timestamp_us'}, inplace=True)

    df['timestamp_ms'] = df['timestamp_us'] / 1000.0
    df['timestamp_ms'] = df['timestamp_ms'] - df['timestamp_ms'].min()

    end_ms = start_ms + duration_ms
    df = df[(df['timestamp_ms'] >= start_ms) & (df['timestamp_ms'] < end_ms)].copy()

    if max_events and len(df) > max_events:
        indices = np.linspace(0, len(df)-1, max_events, dtype=int)
        df = df.iloc[indices]

    if df['polarity'].min() < 0:
        df['polarity'] = (df['polarity'] + 1) // 2

    return df[['timestamp_ms', 'x', 'y', 'polarity']]


def analyze_frequency_content(events: pd.DataFrame, sample_rate: float = 1000.0):
    """
    Analyze frequency content of event stream.

    Args:
        events: Event DataFrame
        sample_rate: Sampling rate in Hz

    Returns:
        frequencies, power spectrum
    """
    # Create time series by binning events
    t_min, t_max = events['timestamp_ms'].min(), events['timestamp_ms'].max()
    duration_sec = (t_max - t_min) / 1000.0

    # Bin events into time bins
    num_bins = int(duration_sec * sample_rate)
    bins = np.linspace(t_min, t_max, num_bins)

    # Count events per bin
    event_counts, _ = np.histogram(events['timestamp_ms'], bins=bins)

    # FFT
    frequencies = fftfreq(len(event_counts), 1/sample_rate)
    fft_values = fft(event_counts)
    power = np.abs(fft_values)**2

    # Only positive frequencies
    positive_freq_idx = frequencies > 0

    return frequencies[positive_freq_idx], power[positive_freq_idx]


def plot_comprehensive_comparison(csv_files: dict,
                                  voltage: str = '300mV',
                                  save_dir: str = './enhanced_analysis'):
    """
    Create comprehensive comparison showing multiple aspects.

    Strategy:
    - All columns use 0.5s window for consistent analysis
    - Column 1: Time vs X helix visualization
    - Column 2: Frequency spectrum analysis
    - Column 3: Event rate over time
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    waveform_order = ['sine', 'square', 'triangle', 'burst']

    # Create figure with multiple columns
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    print(f"\n{'='*70}")
    print(f"Comprehensive Waveform Analysis")
    print(f"{'='*70}")
    print(f"Helix window:     {HELIX_DURATION_MS}ms @ {HELIX_START_MS}ms (max {MAX_EVENTS_HELIX:,} events)")
    print(f"Frequency window: {FREQUENCY_DURATION_MS}ms @ {FREQUENCY_START_MS}ms (max {MAX_EVENTS_FREQUENCY:,} events)")
    print(f"Rate window:      {RATE_DURATION_MS}ms @ {RATE_START_MS}ms (max {MAX_EVENTS_RATE:,} events)")

    for i, waveform in enumerate(waveform_order):
        if waveform not in csv_files:
            continue

        csv_path = csv_files[waveform]
        print(f"\n{waveform.upper()}:")

        # Column 1: Time vs X helix visualization
        print(f"  Loading helix window ({HELIX_DURATION_MS}ms @ {HELIX_START_MS}ms)...")
        ax1 = fig.add_subplot(gs[i, 0])
        events_helix = load_events_window(csv_path, HELIX_START_MS, HELIX_DURATION_MS,
                                         max_events=MAX_EVENTS_HELIX)

        on_events = events_helix[events_helix['polarity'] == 1]
        off_events = events_helix[events_helix['polarity'] == 0]

        if len(on_events) > 0:
            ax1.scatter(on_events['timestamp_ms'], on_events['x'],
                       c='blue', s=0.8, alpha=0.5, rasterized=True)
        if len(off_events) > 0:
            ax1.scatter(off_events['timestamp_ms'], off_events['x'],
                       c='red', s=0.8, alpha=0.5, rasterized=True)

        ax1.set_ylabel('X (pixels)', fontsize=10, fontweight='bold')
        ax1.set_title(f'{waveform.upper()} - {HELIX_DURATION_MS}ms helix\n{len(events_helix):,} events',
                     fontsize=11, fontweight='bold')
        ax1.set_ylim(0, 128)
        ax1.grid(True, alpha=0.2)
        if i == 3:
            ax1.set_xlabel('Time (ms)', fontsize=10)

        # Column 2: Frequency analysis
        print(f"  Loading frequency window ({FREQUENCY_DURATION_MS}ms @ {FREQUENCY_START_MS}ms)...")
        events_frequency = load_events_window(csv_path, FREQUENCY_START_MS, FREQUENCY_DURATION_MS,
                                             max_events=MAX_EVENTS_FREQUENCY)

        print(f"  Analyzing frequency content...")
        ax2 = fig.add_subplot(gs[i, 1])

        try:
            freqs, power = analyze_frequency_content(events_frequency, sample_rate=SAMPLE_RATE_HZ)

            # Select frequency range to plot
            freq_mask = freqs <= MAX_FREQ_DISPLAY_HZ
            freqs_to_plot = freqs[freq_mask]
            power_to_plot = power[freq_mask]

            # Plot power spectrum (log scale)
            ax2.semilogy(freqs_to_plot, power_to_plot, linewidth=2)
            ax2.set_ylabel('Power (log scale)', fontsize=10)
            ax2.set_title(f'{waveform.upper()} - Frequency Spectrum',
                         fontsize=11, fontweight='bold')
            ax2.set_xlim(0, MAX_FREQ_DISPLAY_HZ)
            ax2.grid(True, alpha=0.2)
            if i == 3:
                ax2.set_xlabel('Frequency (Hz)', fontsize=10)

            # Find dominant frequency
            dominant_idx = np.argmax(power_to_plot)
            dominant_freq = freqs_to_plot[dominant_idx]
            print(f"  Dominant frequency: {dominant_freq:.1f} Hz")

        except Exception as e:
            print(f"  Frequency analysis failed: {e}")

        # Column 3: Event rate over time
        print(f"  Loading rate window ({RATE_DURATION_MS}ms @ {RATE_START_MS}ms)...")
        events_rate = load_events_window(csv_path, RATE_START_MS, RATE_DURATION_MS,
                                        max_events=MAX_EVENTS_RATE)

        print(f"  Computing event rate...")
        ax3 = fig.add_subplot(gs[i, 2])

        # Bin events into small time windows
        time_bins = np.arange(events_rate['timestamp_ms'].min(),
                             events_rate['timestamp_ms'].max(), EVENT_RATE_BIN_MS)
        event_rates, _ = np.histogram(events_rate['timestamp_ms'], bins=time_bins)

        bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
        ax3.plot(bin_centers, event_rates, linewidth=1.5)
        ax3.set_ylabel(f'Events / {EVENT_RATE_BIN_MS}ms', fontsize=10)
        ax3.set_title(f'{waveform.upper()} - Event Rate',
                     fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.2)
        if i == 3:
            ax3.set_xlabel('Time (ms)', fontsize=10)

    plt.suptitle(f'Comprehensive Waveform Analysis - {voltage}', fontsize=16, fontweight='bold')

    save_path = save_dir / f'comprehensive_{voltage}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved to: {save_path}")
    return save_path


def plot_statistical_features(csv_files: dict, voltage: str = '300mV', save_dir: str = './enhanced_analysis'):
    """
    Show statistical features that the neural network actually learns.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)

    waveform_order = ['sine', 'square', 'triangle', 'burst']

    # Compute features for each waveform
    features = {}

    print(f"\n{'='*70}")
    print(f"Statistical Features (What the Network Learns)")
    print(f"{'='*70}")

    for waveform in waveform_order:
        if waveform not in csv_files:
            continue

        print(f"\n{waveform.upper()}:")
        events = load_events_window(csv_files[waveform], 1000, 5000, max_events=500000)

        # Compute various statistical features
        features[waveform] = {}

        # 1. Inter-event intervals
        iei = np.diff(events['timestamp_ms'].values)
        features[waveform]['iei_mean'] = np.mean(iei)
        features[waveform]['iei_std'] = np.std(iei)
        features[waveform]['iei_cv'] = features[waveform]['iei_std'] / features[waveform]['iei_mean']

        # 2. Spatial distribution
        features[waveform]['x_mean'] = events['x'].mean()
        features[waveform]['x_std'] = events['x'].std()

        # 3. Event rate variability
        time_bins = np.arange(events['timestamp_ms'].min(), events['timestamp_ms'].max(), 50)
        rates, _ = np.histogram(events['timestamp_ms'], bins=time_bins)
        features[waveform]['rate_std'] = np.std(rates)
        features[waveform]['rate_mean'] = np.mean(rates)
        features[waveform]['rate_cv'] = features[waveform]['rate_std'] / features[waveform]['rate_mean']

        # 4. Polarity balance
        on_ratio = (events['polarity'] == 1).mean()
        features[waveform]['on_ratio'] = on_ratio

        print(f"  Inter-event interval: {features[waveform]['iei_mean']:.4f} ± {features[waveform]['iei_std']:.4f} ms")
        print(f"  IEI coefficient of variation: {features[waveform]['iei_cv']:.3f}")
        print(f"  Event rate: {features[waveform]['rate_mean']:.1f} ± {features[waveform]['rate_std']:.1f} events/50ms")
        print(f"  Rate CV: {features[waveform]['rate_cv']:.3f}")
        print(f"  ON event ratio: {on_ratio:.3f}")

    # Plot feature comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    feature_names = ['iei_cv', 'rate_cv', 'x_std', 'on_ratio', 'rate_std', 'rate_mean']
    feature_labels = ['IEI Coefficient of Variation', 'Rate CV', 'Spatial Std Dev',
                     'ON Event Ratio', 'Rate Std Dev', 'Mean Rate']

    for i, (feat_name, feat_label) in enumerate(zip(feature_names, feature_labels)):
        ax = axes[i]

        values = [features[w][feat_name] for w in waveform_order if w in features]
        labels = [w.upper() for w in waveform_order if w in features]
        colors = ['royalblue', 'darkgreen', 'purple', 'darkred']

        bars = ax.bar(labels, values, color=colors[:len(values)])
        ax.set_ylabel(feat_label, fontsize=10)
        ax.set_title(feat_label, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'Statistical Features - {voltage}\n(What the Neural Network Actually Learns)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = save_dir / f'statistical_features_{voltage}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved to: {save_path}")
    return save_path


def main():
    """Main analysis script."""
    data_dir = Path("~/Desktop/QuantumNetwork/data/raw_truncated").expanduser()
    voltage = '300mV'

    # Find CSV files
    csv_files = {}
    for waveform in ['sine', 'square', 'triangle', 'burst']:
        pattern = f"{waveform}-{voltage}.csv"
        matches = list(data_dir.glob(pattern))
        if matches:
            csv_files[waveform] = str(matches[0])

    print("="*70)
    print("Enhanced Waveform Analysis")
    print("="*70)
    print(f"This will create visualizations that clearly show differences!")
    print(f"Processing {len(csv_files)} waveforms at {voltage}")

    # Run comprehensive analysis
    comp_path = plot_comprehensive_comparison(csv_files, voltage)

    # Run statistical feature analysis
    stat_path = plot_statistical_features(csv_files, voltage)

    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nView results:")
    print(f"  xdg-open {comp_path}")
    print(f"  xdg-open {stat_path}")
    print(f"\nKey insights:")
    print(f"  1. Comprehensive analysis shows:")
    print(f"     - Long time windows reveal pattern cycles")
    print(f"     - Frequency spectra show periodic differences")
    print(f"     - Event rate plots reveal motion patterns")
    print(f"  2. Statistical features show:")
    print(f"     - Quantitative differences the network learns")
    print(f"     - These ARE distinguishable by ML models")
    print(f"     - Networks extract features humans don't see in visualizations")


if __name__ == '__main__':
    main()
