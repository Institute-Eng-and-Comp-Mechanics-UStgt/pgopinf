<!-- PROJECT SHIELDS -->

[![arXiv][arxiv-shield]][arxiv-url]
[![DOI][doi-shield]][doi-url]
[![Documentation Status][docs-shield]][docs-url]
[![MIT License][license-shield]][license-url]

# Petrov-Galerkin Operator Inference
A linear system inference approach with stability-encouraging data projection.

## Reference 
The preprint is available on [arXiv](https://doi.org/).
If you use this project for academic work, please consider citing it

>
    Johannes Rettberg, Jonas Nicodemus, Harsh Sharma, Boris Kramer, Jörg Fehr and Benjamin Unger. 
    Petrov-Galerkin operator inference with application to stability-encouraging identification. Arxiv, 2026.

## Abstract
Data-driven model order reduction methods such as operator inference enable the eﬃcient construction of reduced-order models directly from high-dimensional time-domain data. The associated inference problem is typically formulated as a least-squares problem admitting an eﬃcient closed-form solution. The solution converges to the intrusive Galerkin projection onto a low-dimensional subspace when the reduced order goes to the full order. However, it is well known from intrusive model order reduction for linear time-invariant systems that Petrov–Galerkin projections are often preferable. A suitable choice of test space can additionally preserve important system properties such as stability and passivity. To overcome this limitation, we extend the operator inference framework for linear time-invariant systems to incorporate Petrov–Galerkin projections and provide explicit error expressions and bounds between the intrusive and nonintrusive reduced operators, thus generalizing results from the literature. We demonstrate the proposed approach in the context of dissipative and port-Hamiltonian systems. Furthermore, we introduce a novel convex optimization formulation that explicitly enforces the port-Hamiltonian structure on the inferred operators. The eﬀectiveness of the proposed methods is demonstrated through numerical examples.

 
## Installation

Clone this repository and install it to your local environment as package using pip:
 #TODO
```bash
git clone git@github.com:JohannesRettberg/PortHamiltonianInference.git
cd PortHamiltonianInference
```
Then you can activate the environment in which you want to install the package, and use pip to perform the installation.
```bash
pip install -e .
```

> :warning: **Please note that you need pip version 24.0 to install the repository in editable mode. Either upgrade pip to the latest version or install it without the ```-e``` argument**


## Paper Experiments
### Note on Mosek
In the publication, we used a Mosek solver for solving the convex optimization problem in Experiment 3. Mosek requires a license file which can be obtained for free for academic use. 
You can request an academic license at [https://www.mosek.com/products/academic-licenses/](https://www.mosek.com/products/academic-licenses/) which usually immediatly gives you the required `mosek.lic` file.
The file should be saved under
```
%USERPROFILE%\mosek\mosek.lic           (Windows)
$HOME/mosek/mosek.lic                   (Linux, MacOS)
```
as explained in [https://docs.mosek.com/11.1/install/installation.html](https://docs.mosek.com/11.1/install/installation.html).

Alternatively, you can use free solvers like `SCS` or `CLARABEL`, see also [https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver](https://www.cvxpy.org/tutorial/solvers/index.html#choosing-a-solver).
With these solvers it can not be ensured that the results of the third experiment from the paper are reproduced.

### Running all experiments
To reproduce the results from the paper, run:
```bash
python scripts/reproduce_paper_results.py
```
This script executes all experiments necessary to generate the figures and table.

Note that computing the stability fraction for the poro experiments requires more reduced orders (`r=2,8,...,200`). These can be computed using:
```bash
python scripts/reproduce_paper_results.py --long
```
> [!NOTE] 
> Running all experiments will take up to 1.5h on a standard workstation and around 2h with `--long`.


Afterwards, generate the figures and table data with
```bash
python scripts/build_paper_figures.py
```
The figures will be saved in `results_paper/`.

### Configuration of the Examples
The examples use predefined `ExperimentSpec` classes which can be found under [src/pgopinf/specs/presets/experiments](./src/pgopinf/specs/presets/experiments)
and the different studies are defined by preset `StudySpec` classes in
[src/pgopinf/specs/presets/studies](./src/pgopinf/specs/presets/studies).

## Structure of the Repository

The repository is organized around the general orchestration of experiments and studies through `ExperimentSpec` and `StudySpec`. From there, the sub-specs define the individual quantities used in a workflow: data generation, initial conditions and inputs, reduction, identification, evaluation, and system analysis. The preset experiment and study definitions in `src/pgopinf/specs/presets/` combine these building blocks into ready-to-run examples.

The metrics defined as `EvaluationSpec` will be calculated for individual experiments runs and saved under `results/runs`. Metrics that are defined in `StudyEvaluationSpec` for evaluations on a study run, e.g. over a set of reduced orders, will be saved in `results/studies`.

Further details of the individual methods can be found in the documentation.

## References

[1] Johannes Rettberg, Jonas Nicodemus, Harsh Sharma, Boris Kramer, Jörg Fehr and Benjamin Unger. 
    Petrov-Galerkin operator inference with application to stability-encouraging identification. Arxiv, 2026.

[license-shield]: https://img.shields.io/github/license/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN.svg
[license-url]: https://github.com/Institute-Eng-and-Comp-Mechanics-UStgt/ApHIN/blob/main/LICENSE
[doi-shield]: https://img.shields.io/badge/doi-10.18419%2Fdarus--4446-d45815.svg
[doi-url]: https://doi.org/10.18419/darus-4446
[arxiv-shield]: https://img.shields.io/badge/arXiv-2408.08185-b31b1b.svg
[arxiv-url]: https://doi.org/10.48550/arXiv.2408.08185
[docs-url]: https://Institute-Eng-and-Comp-Mechanics-UStgt.github.io/ApHIN
[docs-shield]: https://img.shields.io/badge/docs-online-blue.svg
