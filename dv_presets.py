"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus dem
OR-Demo-Portfolio, siehe ag_presets.py in agglomerative-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import dv_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _shape_caster(v):
    return v if v in C.SHAPES else C.DEFAULT_SHAPE


def _split_criterion_caster(v):
    return v if v in C.SPLIT_CRITERIA else C.DEFAULT_SPLIT_CRITERION


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "target_k_slider": SettingSpec("tk", int, C.DEFAULT_TARGET_K, C.TARGET_K_MIN, C.TARGET_K_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "spread_imbalance_slider": SettingSpec(
        "si", float, C.DEFAULT_SPREAD_IMBALANCE, C.SPREAD_IMBALANCE_MIN, C.SPREAD_IMBALANCE_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "shape_radio": SettingSpec("shape", _shape_caster, C.DEFAULT_SHAPE),
    "split_criterion_radio": SettingSpec(
        "crit", _split_criterion_caster, C.DEFAULT_SPLIT_CRITERION
    ),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_points, k, target_k, spread, spread_imbalance, seed, shape, split_criterion):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["k"] = str(int(k))
        st.query_params["tk"] = str(int(target_k))
        st.query_params["spread"] = str(spread)
        st.query_params["si"] = str(spread_imbalance)
        st.query_params["seed"] = str(int(seed))
        st.query_params["shape"] = shape
        st.query_params["crit"] = split_criterion
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["target_k_slider"] = p["target_k"]
    st.session_state["spread_slider"] = p["spread"]
    st.session_state["spread_imbalance_slider"] = p["spread_imbalance"]
    st.session_state["shape_radio"] = p["shape"]
    st.session_state["split_criterion_radio"] = p["split_criterion"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
