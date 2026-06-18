# PseudoDen Research

[Development weblog](weblog.md)

Link to video: [Project walkthrough](https://youtu.be/CSjTvONUVlk)

# Introduction

PseudoDen Research is a Python experiment about pursuit behavior in a small snake-and-player arena. The project compares two ways of controlling the snake: a planned `A*` pathfinding baseline and a lightweight machine-learning policy trained from `A*` teacher decisions.

The main objective is not to prove that the learned model is faster than `A*`. Instead, the project asks a more experience-focused question: where does a learned local reaction feel different from a shortest-path planner, and when can that difference create more useful pressure, variation, or uncertainty for the player?

The comparison focuses on four experience-facing ideas:

- **accuracy**: how closely the model matches the `A*` teacher and how reliably each strategy catches the target
- **randomness**: how often the chosen movement direction changes
- **aggressiveness**: whether the snake is closing distance over time
- **response efficiency**: where the algorithm reacts most strongly on the map and across distance bands

![PseudoDen dev view](assets/weblog/runtime_dev_panel.png)
<sub>Fig.1 The Python runtime with obstacles, snake state, Jung functions, and state-machine readout</sub>

# Related technical and/or creative work

The archived browser prototype under `archive/frontend` is based on the original PseudoDen repository, a p5.js game project with procedural animation and a simple pathfinding enemy (pinowine, n.d.; p5.js contributors, n.d.). The current repository keeps that playable idea but turns it into a Python research tool for pathfinding, logging, training, and visual comparison.

`A*` is the main pathfinding baseline. It searches a graph by combining the known cost from the start node with a heuristic estimate of the remaining cost to the goal. In a grid-based map with obstacles, this makes `A*` a good fit for repeatable, inspectable routes (Hart, Nilsson and Raphael, 1968).

The machine-learning route follows imitation learning. `A*` supplies teacher actions, and the model learns a one-step movement decision from local geometry features. This keeps the experiment close to behavior cloning: the learner copies an expert policy, then the project checks where that copy holds up or drifts on held-out routes and maps (Boyu AI, n.d.).

The early model choice was `DecisionTreeClassifier`, because a decision tree is easy to explain and inspect: each movement choice follows a visible set of feature splits. The final implementation uses a CPU-friendly incremental scikit-learn classifier instead, because the runtime only needs one small feature vector and one movement prediction per frame, while route training benefits from repeated partial updates (Pedregosa et al., 2011; scikit-learn developers, n.d.a).

# Summary of design and development process

The development process moved from a translated playable prototype to a measurable research workflow. First, the Python runtime and `A*` baseline made the snake behavior visible and repeatable. Next, the project added a development panel, obstacle maps, and headless simulation so runs could be automated. The model work then shifted from a first supervised classifier idea to route-based imitation training, best-model promotion, map tests, and data visualisation. The full timeline is recorded in the [development weblog](weblog.md), especially the entries on the Python runtime, headless training, model tuning, and held-out map evaluation.

![Training optimization](assets/weblog/training_optimization.png)
<sub>Fig.2 Training and tuning curves used to decide whether the frozen model was stable enough for comparison</sub>

![Training log view](assets/weblog/training_log_console.png)
<sub>Fig.3 Console training logs made model changes visible across repeated runs</sub>

# Summary of final version

The final version is a compact research application rather than a finished game. It has an interactive pyglet window, an `A*` strategy, an imitation-learning strategy, a headless training path, telemetry logs, analysis scripts, and selected figures for documentation.

The interactive app can run with either controller. The development panel keeps the runtime readable by showing the snake state, Jung-function values, and the state-machine readout. The headless commands run repeated episodes without opening the window, which makes it possible to train, test, and compare strategies from stored data.

The model is intentionally lightweight. It uses local geometry features and CPU inference, with `A*` still available as the teacher and comparison baseline. Best-model saving is handled through scikit-learn and joblib so that a promoted model can be reused in later runs (joblib developers, n.d.; scikit-learn developers, n.d.a).

The current scope works well for comparing one snake against scripted player routes on several obstacle maps. It does not claim to solve general game AI, multi-agent pursuit, or deep reinforcement learning. The known weak point is still generalisation in layouts where a global route is essential, especially narrow passages.

# Evaluation

The project accomplished its main objective by making the difference between planned pursuit and learned local pursuit measurable. The result is not that the machine-learning strategy beats `A*` everywhere. Specifically, the learned model has useful experience differences in some map shapes, while `A*` remains stronger when long-range planning matters.

![Cross-map summary](assets/weblog/cross_map_summary.png)
<sub>Fig.4 Cross-map comparison of average distance, catches, and closing rate</sub>

![Cross-map advantage](assets/weblog/cross_map_advantage.png)
<sub>Fig.5 Advantage signals show where the learned model leads or falls behind `A*`</sub>

Held-out map tests show that the learned model performs best on `corridor_gates`, where it reaches lower average distance and more catches than `A*`. On `dense_blocks`, both strategies struggle, but the learned model remains competitive. On `narrow_passages`, `A*` is much stronger because global route planning matters. On `sparse_blocks`, the learned model shows pressure-style closing behavior but does not consistently turn that pressure into catches.

The evaluation also checks imitation-learning risks. Behavior cloning can copy expert actions quickly, but small disagreements may move the learner into states that were not well covered by the teacher data. This is the compounding-error problem described in imitation-learning literature, and the project visualises it through confusion matrices, distance drift, and occupancy gaps (Boyu AI, n.d.; scikit-learn developers, n.d.b).

![Imitation confusion matrix](assets/weblog/imitation_confusion_matrix.png)
<sub>Fig.6 The confusion matrix shows which `A*` teacher actions the learned policy copies cleanly and where it drifts</sub>

![Compounding error curve](assets/weblog/compounding_error_curve.png)
<sub>Fig.7 Distance drift after a disagreement makes the behavior-cloning error chain visible</sub>

![Occupancy gap heatmap](assets/weblog/occupancy_gap_heatmap.png)
<sub>Fig.8 The occupancy gap heatmap compares where the learned model and `A*` spend their state density</sub>

The experience plots help separate useful behavior from raw correctness. The score curve combines closing, distance improvement, capture, and movement-change penalties. The Pareto scatter separates `A*` stability from learned-model reactivity. These views support the main interpretation: the learned model is most useful when the desired effect is local pressure and less predictable pursuit, not guaranteed shortest-path navigation.

![Experience score curve](assets/weblog/experience_score_curve.png)
<sub>Fig.9 The score curve combines closing, distance improvement, capture, and movement-change penalties</sub>

![Data efficiency curve](assets/weblog/data_efficiency_curve.png)
<sub>Fig.10 The data-efficiency proxy uses route-training logs to show how accuracy changes as more teacher samples are seen</sub>

![Experience Pareto scatter](assets/weblog/experience_pareto_scatter.png)
<sub>Fig.11 The Pareto scatter separates `A*` stability from learned-model reactivity in the movement-change and closing-rate space</sub>

# Reflection and conclusion

The final project feels strongest as a research instrument. It turns a small playable prototype into a system that can run, train, log, compare, and explain behavior. The most valuable outcome is that the model is no longer judged by one exciting visible run. It is judged through routes, maps, metrics, and plots that show both its strengths and its failure cases.

The main thing I would improve next is the training distribution. The current model learns from selected routes and random samples, but a stronger next version would collect more recovery states, add harder map variations, and test methods that reduce compounding error. I would also add a small user-facing study, because the core question is experiential: whether the learned pursuit feels more tense, readable, or surprising than `A*` in practice.

Overall, the project reaches a useful final state for submission. It does not replace `A*`; it makes a comparison possible and identifies the model's advantage zone: local pressure, reactive movement, and behavior style.

# Repository structure and instructions for running

## Repository structure

- `main.py` is the command-line entry point for interactive runs, headless checks, training, and analysis
- `src/pseudoden_research` contains the simulation, entities, maps, `A*` strategy, model strategy, telemetry, and tuning logic
- `src/data-analyze` contains log post-processing, route training, map testing, and Seaborn/Matplotlib plotting scripts
- `assets/weblog` contains the tracked images used by the README and weblog
- `archive/frontend` contains the archived p5.js prototype source
- `reports`, `logs`, `models`, and `docs` are local generated outputs or working notes and are intentionally ignored by Git

## Environment setup

Create an environment and install the project dependencies:

```powershell
conda create -n pressure-estimator python=3.11
conda activate pressure-estimator
pip install -r requirements.txt
```

## Interactive runtime

Run the interactive app with either strategy:

```powershell
python main.py --strategy astar
python main.py --strategy ml
```

## Headless training and analysis

Run smoke checks and model training:

```powershell
python main.py --smoke-test
python main.py --train-ml --episodes 20 --frames 600
python main.py --route-train-ml --episodes 12 --frames 600 --route-passes 3 --train-routes box,zigzag,orbit --test-routes diagonal_sweep --random-samples 512
```

Generate comparison data and figures:

```powershell
python main.py --visualize-analysis --episodes 20 --frames 600 --routes box,zigzag,orbit,diagonal_sweep
python main.py --map-test-analysis --episodes 20 --frames 600 --maps sparse_blocks,corridor_gates,dense_blocks,narrow_passages --map-routes diagonal_sweep
```

The analysis commands output CSV files and figures under `reports`. The tracked weblog images under `assets/weblog` are selected final figures copied from those generated outputs.

## Tests

Run the unit tests:

```powershell
python -m unittest discover -s tests
```

# Use of external resources

## Statement on use of AI tools

Assistive AI tools were used during brainstorming and the JavaScript-to-Python prototype translation. The original playable prototype under `archive/frontend` came from the original PseudoDen repository (pinowine, n.d.). The current Python research version was created with LLM assistance to translate and refactor that prototype into Python simulation, model, logging, and analysis code. The generated translation baseline is recorded in this commit: <https://github.com/pinowine/pressure-estimator/commit/28b8b837cc353d890ef51af2ce4405c35efa5175>.

Here's the prompt I passed to `Codex Agent`, using `GPT-5.5` model (translated, original language in Simplified Chinese):
> This project is a game where several hunters chase a player. The current requirement is to conduct a study comparing the changes in pathfinding logic for the snake (hunter) under LLM, Machine Learning, and hard-coded A* algorithms. Therefore, the code needs to be simplified and the tech stack may need to be changed (if necessary), including: 
> 
> 1. Switching the programming language from frontend (js) to Python (if necessary); 
> 2. Removing all scene transitions, leaving only a single looping scene; 
> 3. Implementing a developer mode to display the snake's pathfinding and parameters; 
> 4. Keeping only one snake to interact with the player; 
> 5. Removing all scene elements and tilemaps, leaving only a solid color background, and changing the player to move in any direction (rather than platform-scrolling mode) and the snake to crawl anywhere (rather than being restricted by wall layers).
>

In `Planning Mode` we discussed about technical stack, then decided to choose the current ones (some of them had changed in the process, like the classifier).

## Use of other third-party resources

The project uses pyglet for the interactive Python window (pyglet contributors, n.d.), scikit-learn for the lightweight classifier and evaluation helpers (Pedregosa et al., 2011; scikit-learn developers, n.d.a; scikit-learn developers, n.d.b), pandas for frame and episode log processing (pandas development team, n.d.), and Seaborn/Matplotlib for figures (Waskom, 2021; Waskom, n.d.; Hunter, 2007; Matplotlib development team, n.d.).

Some long non-core analysis blocks are marked in code with `# --- this part comes from ---`. These sections are using references below. `src/data-analyze/diagnostics.py` follows imitation-learning diagnostics such as behavior-cloning error chains and occupancy comparison (Boyu AI, n.d.). `src/data-analyze/route_training.py` follows the supervised behavior-cloning setup and scikit-learn `SGDClassifier.partial_fit` API (scikit-learn developers, n.d.a). `src/data-analyze/plots.py` and `src/data-analyze/map_analysis.py` follow standard pandas, Seaborn, and Matplotlib plotting APIs for heatmaps, line charts, scatter plots, and rectangle overlays (pandas development team, n.d.; Waskom, n.d.; Matplotlib development team, n.d.).

# References

Boyu AI (n.d.) 'Imitation learning', *Hands-on Reinforcement Learning*. Available at: https://hrl.boyuai.com/chapter/3/%E6%A8%A1%E4%BB%BF%E5%AD%A6%E4%B9%A0/.

Hart, P.E., Nilsson, N.J. and Raphael, B. (1968) 'A formal basis for the heuristic determination of minimum cost paths', *IEEE Transactions on Systems Science and Cybernetics*, 4(2), pp. 100-107. Available at: https://doi.org/10.1109/TSSC.1968.300136.

Hunter, J.D. (2007) 'Matplotlib: A 2D graphics environment', *Computing in Science & Engineering*, 9(3), pp. 90-95. Available at: https://doi.org/10.1109/MCSE.2007.55.

joblib developers (n.d.) *joblib.dump documentation*. Available at: https://joblib.readthedocs.io/en/latest/generated/joblib.dump.html.

Matplotlib development team (n.d.) *matplotlib.patches.Rectangle documentation*. Available at: https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html.

p5.js contributors (n.d.) *p5.js*. Available at: https://p5js.org/.

pandas development team (n.d.) *pandas crosstab and reshaping documentation*. Available at: https://pandas.pydata.org/docs/reference/api/pandas.crosstab.html and https://pandas.pydata.org/docs/user_guide/reshaping.html.

Pedregosa, F. et al. (2011) 'Scikit-learn: Machine learning in Python', *Journal of Machine Learning Research*, 12, pp. 2825-2830. Available at: https://jmlr.org/papers/v12/pedregosa11a.html.

pinowine (n.d.) *PseudoDen*. GitHub repository. Available at: https://github.com/pinowine/PseudoDen.

pyglet contributors (n.d.) *pyglet documentation*. Available at: https://pyglet.readthedocs.io/.

scikit-learn developers (n.d.a) *SGDClassifier documentation*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html.

scikit-learn developers (n.d.b) *Confusion matrix documentation*. Available at: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html.

Waskom, M.L. (2021) 'Seaborn: Statistical data visualization', *Journal of Open Source Software*, 6(60), 3021. Available at: https://doi.org/10.21105/joss.03021.

Waskom, M. (n.d.) *Seaborn heatmap, lineplot, and scatterplot API documentation*. Available at: https://seaborn.pydata.org/generated/seaborn.heatmap.html, https://seaborn.pydata.org/generated/seaborn.lineplot.html and https://seaborn.pydata.org/generated/seaborn.scatterplot.html.
