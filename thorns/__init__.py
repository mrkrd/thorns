
from spikes import (
    make_trains,

    trains_to_array,

    accumulate,
    accumulate_spikes,
    accumulate_trains,
    accumulate_spike_trains,

    select_trains,
    select,
    sel,

    trim_spike_trains,
    trim_trains,
    trim,

    fold_spike_trains,
    fold_trains,
    fold,
)


from calc import (
    get_duration,

    shuffled_autocorrelogram,
    sac,

    correlation_index,
    ci,

    firing_rate,
    rate,

    psth,

    isih,

    entrainment,

    synchronization_index,
    si,

    count_spikes,
    count,

    period_histogram
)

from plot import (
    plot_raster,
    plot_psth,
    plot_neurogram,
    plot_sac,
    plot_period_histogram,
    plot_signal
)

from mrlib import show

def gcf():
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    return fig
