
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

    calc_shuffled_autocorrelogram,
    calc_sac,

    calc_correlation_index,
    calc_ci,

    calc_firing_rate,
    calc_rate,

    calc_psth,

    calc_isih,

    calc_entrainment,

    calc_synchronization_index,
    calc_si,

    count_spikes,
    count
)

from plot import (
    plot_raster,
    plot_psth,
    plot_neurogram,
    plot_sac
)
