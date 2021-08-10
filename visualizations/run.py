# builtin modules
import os
import glob

# external modules
import streamlit as st
import matplotlib.pyplot as plt
from flatland.utils.rendertools import RenderTool

# internal modules
from vis_utils import ENV_RECORDS_PATH
from vis_utils import load_env_record


def get_env_step_image(env_record, renderer, step):
    env_record.set_record_step(step)
    renderer.render_env(show_observations=False)
    image = renderer.get_image()

    return image


def single_recorded_run_gif():
    recorded_gifs_fullpath = glob.glob(ENV_RECORDS_PATH + "/*.gif")
    recorded_gifs_fullpath.sort()
    recorded_gifs_names = [
        os.path.basename(p.strip(".gif")) for p in recorded_gifs_fullpath
    ]

    # title
    st.write("# View a single recorded run (GIF)")

    run = st.selectbox("Select a run", recorded_gifs_names)
    gif_fullpath = recorded_gifs_fullpath[recorded_gifs_names.index(run)]

    st.image(gif_fullpath)


def single_recorded_run():
    recorded_runs_fullpath = glob.glob(ENV_RECORDS_PATH + "/*.envrecord.pickle")
    recorded_runs_fullpath.sort()
    recorded_runs_names = [
        os.path.basename(p.strip(".envrecord.pickle")) for p in recorded_runs_fullpath
    ]

    # title
    st.write("# View a single recorded run")

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
    recorded_runs_fullpath = glob.glob(ENV_RECORDS_PATH + "/*.envrecord.pickle")
    recorded_runs_fullpath.sort()
    recorded_runs_names = [
        os.path.basename(p.strip(".envrecord.pickle")) for p in recorded_runs_fullpath
    ]

    st.write("# Compare two recorded runs")

    l_run = st.selectbox("Select a run (left)", recorded_runs_names, key="l_run")
    r_run = st.selectbox("Select a run (right)", recorded_runs_names, key="r_run")

    left, right = st.columns(2)
    with left:
        l_display = left.empty()

    with right:
        r_display = right.empty()

    for display in [l_display, r_display]:
        fig, ax = plt.subplots()
        ax.set_axis_off()
        display.write(fig)

    l_env_record = load_env_record(
        recorded_runs_fullpath[recorded_runs_names.index(l_run)]
    )
    r_env_record = load_env_record(
        recorded_runs_fullpath[recorded_runs_names.index(r_run)]
    )

    records_length = [
        l_env_record.get_record_length(),
        r_env_record.get_record_length(),
    ]

    step = st.sidebar.slider("Step Number:", min_value=0, max_value=max(records_length))

    for display, env_record in [(l_display, l_env_record), (r_display, r_env_record)]:
        renderer = RenderTool(env_record, gl="PIL")

        display_step = max(step, l_env_record.get_record_length())

        image = get_env_step_image(env_record, renderer, display_step)

        fig, ax = plt.subplots()
        ax.set_title(f"Step {display_step}")
        ax.set_axis_off()
        ax.imshow(image)
        display.write(fig)


modes = {
    "View a recorded run": single_recorded_run,
    "Compare two recorded runs": compare_recorded_runs,
    "View a recorded run (GIF)": single_recorded_run_gif,
}

mode_key = st.sidebar.selectbox("Choose a demo", [k for k in modes])
st.sidebar.write("")
mode = modes[mode_key]
mode()
