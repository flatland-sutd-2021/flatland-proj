# builtin modules
import os
import glob
from functools import lru_cache

# external modules
import streamlit as st
import matplotlib.pyplot as plt
from flatland.utils.rendertools import RenderTool

# internal modules
from vis_utils import ENV_RECORDS_PATH
from vis_utils import load_env_record

# display = st.empty()

# fig, ax = plt.subplots()
# ax.plot([1, 2, 3, 4])

# display.write(fig)

# # Page title
# st.write("# Visualize a Single Run")

# # Figure Display
# display = st.empty()
# fig, ax = plt.subplots()
# ax.set_axis_off()
# display.write(fig)

# # Step slider
# st.slider("Step Number:")


@lru_cache
def get_env_step_image(env_record, renderer, step):
    env_record.set_record_step(step)
    renderer.render_env(show_observations=False)
    image = renderer.get_image()

    return image


def single_recorded_run():
    recorded_runs_fullpath = glob.glob(ENV_RECORDS_PATH + "/*.envrecord.pickle")
    recorded_runs_fullpath.sort()
    recorded_runs_names = [os.path.basename(p) for p in recorded_runs_fullpath]

    # title
    st.write("# single_recorded_run")

    run = st.selectbox("Select a run", recorded_runs_names)
    run_idx = recorded_runs_names.index(run)
    run_fullpath = recorded_runs_fullpath[run_idx]

    env_record = load_env_record(run_fullpath)
    record_length = env_record.get_record_length()

    renderer = RenderTool(env_record, gl="PIL")

    # # Figure Display
    display = st.empty()
    fig, ax = plt.subplots()
    ax.set_axis_off()

    step = st.sidebar.slider("Step Number:", min_value=0, max_value=record_length)
    image = get_env_step_image(env_record, renderer, step)

    ax.set_title(f"Step {step}")
    ax.imshow(image)
    display.write(fig)


def compare_recorded_runs():
    st.write("## compare_recorded_run")
    pass


modes = {
    "View a recorded run": single_recorded_run,
    "Compare two recorded runs": compare_recorded_runs,
}

mode_key = st.sidebar.selectbox("Choose a demo", [k for k in modes])
st.sidebar.write("")
mode = modes[mode_key]
mode()
