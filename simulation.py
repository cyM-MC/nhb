"""Paired populations, facility choices and first-come-first-served service queues."""

import heapq

import numpy as np
import pandas as pd

from decision import choose_facilities


def make_population(origins, config, seed, weights=None):
    # Separate streams keep exogenous populations and choice noise independent.
    population_rng, choice_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    n = config.n_agents
    agents = pd.DataFrame({
        'agent_id': np.arange(n),
        'start_node': population_rng.choice(origins, n, p=weights),
        'threshold': population_rng.uniform(config.threshold_low, config.threshold_high, n),
        'requires_core': population_rng.random(n) < config.requires_core_fraction,
        'departure_min': population_rng.uniform(0, config.departure_window_min, n),
    })
    return agents, choice_rng


def queue_waits(arrivals, choices, facilities, agent_ids):
    """FCFS parallel-server queues, initially empty; deterministic service duration.

    Patients are served even after the departure window ends; that window controls
    demand only. Equal arrivals break ties by agent_id, not input row order.
    """
    arrivals = np.asarray(arrivals, dtype=float)
    choices = np.asarray(choices)
    agent_ids = np.asarray(agent_ids)
    if arrivals.shape != choices.shape or arrivals.shape != agent_ids.shape:
        raise ValueError("Queue arrays must have identical shapes")
    if len(set(agent_ids)) != len(agent_ids) or ((choices < -1) | (choices >= len(facilities))).any():
        raise ValueError("Agent IDs must be unique and choices in range")
    valid = choices >= 0
    if not np.isfinite(arrivals[valid]).all() or (arrivals[valid] < 0).any():
        raise ValueError("Reachable arrivals must be finite and non-negative")
    waits = np.full(len(arrivals), np.nan)
    for j, facility in facilities.iterrows():
        patients = np.flatnonzero(choices == j)
        patients = patients[np.lexsort((agent_ids[patients], arrivals[patients]))]
        servers = [0.0] * int(facility['servers'])
        for i in patients:
            free = heapq.heappop(servers)
            service_start = max(arrivals[i], free)
            waits[i] = service_start - arrivals[i]
            heapq.heappush(servers, service_start + facility['service_minutes'])
    return waits


def run_scenario(population, times, facilities, config, stress, scenario, noise):
    name, friction, exposure, fastest = scenario
    choice, alpha = choose_facilities(
        times, facilities, population['threshold'], stress, population['requires_core'],
        config, friction, exposure, noise, fastest)
    recommended = np.zeros(len(population), dtype=bool)
    predicted_gain = np.zeros(len(population))
    if name == 'Adaptive':
        choice, recommended, predicted_gain = adaptive_choices(population, times, facilities, choice, config)
    valid = choice >= 0
    n = len(population)
    selected_time = np.full(n, np.nan)
    selected_time[valid] = times[np.flatnonzero(valid), choice[valid]] / 60
    eligible = (~population['requires_core'].to_numpy()[:, None]
                | (facilities['facility_type'].to_numpy() == 'core')[None, :])
    fastest_time = np.where(eligible, times, np.inf).min(axis=1) / 60
    fastest_time[~np.isfinite(fastest_time)] = np.nan
    waits = queue_waits(population['departure_min'].to_numpy() + selected_time,
                        choice, facilities, population['agent_id'].to_numpy())
    result = population.copy()
    result['scenario'] = name
    result['alpha'] = alpha
    result['recommendation_accepted'] = recommended
    result['predicted_gain_min'] = predicted_gain
    for column, output_column in [('name', 'chosen_facility'), ('facility_type', 'facility_type')]:
        values = np.full(n, None, dtype=object)
        values[valid] = facilities[column].to_numpy()[choice[valid]]
        result[output_column] = values
    result['reachable'] = valid
    result['travel_min'] = selected_time
    result['fastest_travel_min'] = fastest_time
    result['excess_travel_min'] = np.maximum(0, selected_time - fastest_time)
    result['wait_min'] = waits
    result['access_min'] = selected_time + waits  # departure -> start of service
    result['within_target'] = valid & (result['access_min'] <= config.access_target_min)
    return result


def adaptive_choices(population, times, facilities, baseline, config):
    """Online recommendations using only previously committed arrivals.

    Reservation schedules forecast FCFS queues at candidate arrival times. Later
    departures may arrive earlier: actual waits are recomputed after choices,
    so predicted benefits are not guaranteed realized benefits.
    """
    choices = baseline.copy()
    accepted = np.zeros(len(choices), dtype=bool)
    gains = np.zeros(len(choices))
    bookings = [[] for _ in range(len(facilities))]
    departures = population.departure_min.to_numpy()
    ids = population.agent_id.to_numpy()
    core = facilities.facility_type.to_numpy() == 'core'
    # Stable agent-specific pseudo-uniform values; identical across all settings.
    compliance = pd.util.hash_array(ids.astype(np.int64)) / float(2**64)
    order = np.lexsort((ids, departures))
    for i in order:
        original = baseline[i]
        if original < 0:
            continue
        travel = times[i] / 60
        costs = np.full(len(facilities), np.inf)
        for j, facility in facilities.iterrows():
            if not np.isfinite(travel[j]) or (population.requires_core.iloc[i] and not core[j]):
                continue
            if travel[j] > travel[original] + config.recommendation_max_detour_min:
                continue
            arrival = departures[i] + travel[j]
            servers = [0.] * int(facility.servers)
            for booked_arrival, booked_id in sorted(bookings[j]):
                if (booked_arrival, booked_id) > (arrival, ids[i]):
                    break
                free = heapq.heappop(servers)
                heapq.heappush(servers, max(free, booked_arrival) + facility.service_minutes)
            costs[j] = travel[j] + max(0., servers[0] - arrival)
        best = int(costs.argmin())
        gain = costs[original] - costs[best]
        if best != original and gain >= config.recommendation_min_gain_min and compliance[i] < config.recommendation_compliance:
            choices[i] = best
            accepted[i] = True
            gains[i] = gain
        chosen = choices[i]
        bookings[chosen].append((departures[i] + travel[chosen], ids[i]))
    return choices, accepted, gains
