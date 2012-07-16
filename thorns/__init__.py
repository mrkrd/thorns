
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
)


from calc import (
    calc_shuffled_autocorrelogram,
    calc_sac,

    calc_correlation_index,
    calc_ci,

    calc_firing_rate,

    calc_psth
)

from plot import (
    plot_raster,
    plot_psth,
    plot_neurogram,
    plot_sac
)
