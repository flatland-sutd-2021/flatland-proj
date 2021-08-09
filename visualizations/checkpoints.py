# k_best_paths: 17 + 17
# root_extra: 15 + k * 9
# rvnn children: 12 * k_branches
# hint: 5
# state_size = (17 + 17) + (15 + 5 * 9) + (12 * 2) + (5)

checkpoints = {
    "ablation-hint-selector-1000": {
        "path": "/ablation-hint-selector-1000/",
        "training_id": "210804003647",
        "description": "",
        "trained_episodes": 1000,
        "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    },
    # "ablation-k-best-paths-1000": {
    #     "path": "/ablation-k-best-paths-1000/",
    #     "training_id": "210803115112",
    #     "description": "",
    #     "trained_episodes": 1000,
    #     "state_size": (15 + 5 * 9) + (12 * 2),
    # },
    # "ablation-k-nearest-agents-1000": {
    #     "path": "/ablation-k-nearest-agents-1000/",
    #     "training_id": "210803003108",
    #     "description": "",
    #     "trained_episodes": 1000,
    #     "state_size": (17 + 17) + (15) + (12 * 2),
    # },
    # "ablation-rvnn-1000": {
    #     "path": "/ablation-rvnn-1000/",
    #     "training_id": "210802144426",
    #     "description": "",
    #     "trained_episodes": 1000,
    #     "state_size": (17 + 17) + (15 + 5 * 9),
    # },
    # "adam-hint-checkpoints": {
    #     "path": "/adam-hint-checkpoints/",
    #     "training_id": "210727044113",
    #     "description": "",
    #     "trained_episodes": 1100,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "argmax-rmsprop-full-selector": {
    #     "path": "/argmax-rmsprop-full-selector/",
    #     "training_id": "210729153343",
    #     "description": "",
    #     "trained_episodes": 800,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "argmax-rmsprop-half-selector": {
    #     "path": "/argmax-rmsprop-half-selector/",
    #     "training_id": "210729232438",
    #     "description": "",
    #     "trained_episodes": 1000,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "rms-half-selector-no-hint-full-state": {
    #     "path": "/rms-half-selector-no-hint-full-state/",
    #     "training_id": "210804032459",
    #     "description": "",
    #     "trained_episodes": 1000,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "rms-half-selector-no-hint-no-k-best-path": {
    #     "path": "/rms-half-selector-no-hint-no-k-best-path/",
    #     "training_id": "210803151317",
    #     "description": "",
    #     "trained_episodes": 900,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "rms-no-selector-no-hint-no-final-penalty": {
    #     "path": "/rms-no-selector-no-hint-no-final-penalty/",
    #     "training_id": "210802141232",
    #     "description": "",
    #     "trained_episodes": 1100,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
    # "rmsprop-hint-checkpoints": {
    #     "path": "/rmsprop-hint-checkpoints/",
    #     "training_id": "210727044101",
    #     "description": "",
    #     "trained_episodes": 1200,
    #     "state_size": (17 + 17) + (15 + 5 * 9) + (12 * 2),
    # },
}
